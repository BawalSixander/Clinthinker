import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from app import database, main


class TrainingDatasetTests(unittest.TestCase):
    def with_isolated_storage(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        storage = Path(temporary.name)
        uploads = storage / "uploads"
        patches = (
            patch.object(database, "STORAGE", storage),
            patch.object(database, "UPLOADS", uploads),
            patch.object(database, "DB_PATH", storage / "assistant.sqlite3"),
            patch.object(main, "UPLOADS", uploads),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        database.initialize()
        return uploads

    def test_csv_examples_are_saved_and_used_for_prediction(self):
        self.with_isolated_storage()
        uploaded = UploadFile(
            filename="botany.csv",
            file=io.BytesIO(
                b"topic,text\n"
                b"Botany,photosynthesis chlorophyll plant energy\n"
                b"Botany,plant leaves sunlight carbon dioxide\n"
                b"Astronomy,orbit telescope star galaxy\n"
                b"Astronomy,planet moon nebula telescope\n"
            ),
        )

        dataset = asyncio.run(main.import_training_csv(uploaded, name=None))
        result = main._analyze(
            main.PredictionRequest(
                text="photosynthesis chlorophyll plant",
                training_dataset_id=dataset["id"],
            )
        )

        self.assertEqual(dataset["name"], "botany")
        self.assertEqual(dataset["example_count"], 4)
        self.assertEqual(dataset["topic_count"], 2)
        self.assertEqual(result["training_dataset_name"], "botany")
        self.assertEqual(result["predictions"][0]["topic"], "Botany")
        self.assertEqual(
            {item["topic"] for item in result["predictions"]},
            {"Botany", "Astronomy"},
        )
        with database.connect() as db:
            saved = db.execute(
                "SELECT training_dataset_id, training_dataset_name FROM analyses "
                "WHERE id=?",
                (result["id"],),
            ).fetchone()
        self.assertEqual(saved["training_dataset_id"], dataset["id"])
        self.assertEqual(saved["training_dataset_name"], "botany")

    def test_csv_requires_topic_and_text_columns(self):
        self.with_isolated_storage()
        uploaded = UploadFile(
            filename="invalid.csv",
            file=io.BytesIO(b"label,content\nPlants,green leaves\nStars,night sky\n"),
        )

        with self.assertRaises(HTTPException) as error:
            asyncio.run(main.import_training_csv(uploaded))

        self.assertEqual(error.exception.status_code, 422)
        self.assertIn("'topic' and 'text'", error.exception.detail)
        self.assertEqual(len(main.list_training_datasets()), 1)

    def test_api_upload_select_and_prediction_round_trip(self):
        self.with_isolated_storage()
        content = (
            b"topic,text\n"
            b"Botany,photosynthesis chlorophyll plant energy\n"
            b"Botany,plant leaves sunlight carbon dioxide\n"
            b"Astronomy,orbit telescope star galaxy\n"
            b"Astronomy,planet moon nebula telescope\n"
        )

        with TestClient(main.app) as client:
            uploaded = client.post(
                "/api/training-datasets/csv",
                files={"file": ("custom.csv", content, "text/csv")},
            )
            self.assertEqual(uploaded.status_code, 200, uploaded.text)
            dataset = uploaded.json()
            listed = client.get("/api/training-datasets").json()
            self.assertIn(dataset["id"], {item["id"] for item in listed})

            evaluation = client.post(
                "/api/training-datasets/evaluate",
                json={"training_dataset_id": dataset["id"]},
            )
            self.assertEqual(evaluation.status_code, 200, evaluation.text)
            self.assertEqual(evaluation.json()["training_dataset_name"], "custom")
            self.assertEqual(evaluation.json()["examples_evaluated"], 4)

            response = client.post(
                "/api/predict/stream",
                json={
                    "text": "photosynthesis chlorophyll plant",
                    "training_dataset_id": dataset["id"],
                },
            )
            self.assertEqual(response.status_code, 200)
            result_event = next(
                line.removeprefix("data: ")
                for line in response.text.splitlines()
                if line.startswith("data: ")
                and '"training_dataset_name"' in line
            )
            result = json.loads(result_event)
            self.assertEqual(result["training_dataset_name"], "custom")
            self.assertEqual(result["predictions"][0]["topic"], "Botany")
            self.assertEqual(
                client.get("/api/analysis/latest").json()["training_dataset_id"],
                dataset["id"],
            )

    def test_appearance_setting_is_saved_and_validated(self):
        self.with_isolated_storage()

        with TestClient(main.app) as client:
            self.assertEqual(client.get("/api/health").json()["service"], "Clinthinker©")
            self.assertEqual(
                client.get("/api/settings").json()["appearance"],
                "orange",
            )
            saved = client.put(
                "/api/settings",
                json={"appearance": "regal", "language": "English"},
            )
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(saved.json()["appearance"], "regal")
            self.assertEqual(
                client.get("/api/settings").json()["appearance"],
                "regal",
            )
            question_settings = {
                "training_dataset_id": "builtin",
                "upload_ids": ["upload-1"],
                "topic": "All Topics",
                "question_type": "Identification",
                "difficulty": "Easy",
                "number_of_questions": 7,
                "questions_per_topic": 2,
                "similarity_threshold": 0.9,
                "priority_weights": {
                    "historical_frequency": 0.25,
                    "course_importance": 0.25,
                    "model_confidence": 0.25,
                    "recent_occurrence": 0.25,
                },
            }
            saved_question_settings = client.put(
                "/api/settings",
                json={"question_generation": question_settings},
            )
            self.assertEqual(saved_question_settings.status_code, 200)
            self.assertEqual(
                saved_question_settings.json()["question_generation"],
                question_settings,
            )
            invalid_question_settings = client.put(
                "/api/settings",
                json={
                    "question_generation": {
                        **question_settings,
                        "priority_weights": {
                            **question_settings["priority_weights"],
                            "recent_occurrence": 0.1,
                        },
                    }
                },
            )
            self.assertEqual(invalid_question_settings.status_code, 422)
            invalid = client.put(
                "/api/settings",
                json={"appearance": "neon"},
            )
            self.assertEqual(invalid.status_code, 422)

    def test_labeled_documents_create_a_deletable_dataset(self):
        uploads = self.with_isolated_storage()
        documents = [
            ("botany.txt", "photosynthesis leaves and chlorophyll"),
            ("astronomy.txt", "stars galaxies and planetary orbits"),
        ]
        with database.connect() as db:
            for index, (name, text) in enumerate(documents):
                stored_path = uploads / f"{index}.txt"
                stored_path.write_text(text, encoding="utf-8")
                db.execute(
                    "INSERT INTO uploads(id, name, stored_path, bytes, text_chars) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (f"upload-{index}", name, str(stored_path), len(text), len(text)),
                )

        dataset = main.create_document_dataset(
            main.DocumentDatasetRequest(
                name="Study topics",
                examples=[
                    main.LabeledUpload(upload_id="upload-0", topic="Botany"),
                    main.LabeledUpload(upload_id="upload-1", topic="Astronomy"),
                ],
            )
        )

        self.assertEqual(dataset["example_count"], 2)
        self.assertEqual(dataset["topic_count"], 2)
        self.assertEqual(main.delete_training_dataset(dataset["id"]), {"deleted": dataset["id"]})
        with self.assertRaises(HTTPException):
            main.delete_training_dataset("builtin")


if __name__ == "__main__":
    unittest.main()
