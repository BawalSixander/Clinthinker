import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import database


class DatabaseMigrationTests(unittest.TestCase):
    def test_model_calibration_table_initialization_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            with (
                patch.object(database, "STORAGE", storage),
                patch.object(database, "UPLOADS", storage / "uploads"),
                patch.object(database, "DB_PATH", storage / "assistant.sqlite3"),
            ):
                database.initialize()
                database.initialize()
                with database.connect() as db:
                    columns = [
                        row["name"]
                        for row in db.execute(
                            "PRAGMA table_info(model_calibration)"
                        ).fetchall()
                    ]

        self.assertEqual(
            columns,
            [
                "dataset_id",
                "dataset_hash",
                "svm_temperature",
                "ensemble_temperature",
                "confidence_threshold",
                "min_margin",
                "disagreement_abstain",
                "target_precision",
                "low_reliability",
                "sklearn_version",
                "created_at",
            ],
        )

    def test_backfills_legacy_saved_questions_into_idempotent_sessions(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            uploads = storage / "uploads"
            db_path = storage / "assistant.sqlite3"
            connection = sqlite3.connect(db_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE generated_questions (
                        id TEXT PRIMARY KEY,
                        question_text TEXT NOT NULL,
                        question_type TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        difficulty TEXT NOT NULL,
                        choices TEXT NOT NULL,
                        correct_answer TEXT NOT NULL,
                        priority_score REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        source_dataset_id TEXT NOT NULL,
                        source_dataset_name TEXT NOT NULL,
                        source_material_id TEXT,
                        source_material_name TEXT NOT NULL,
                        source_reference TEXT NOT NULL,
                        created_at TEXT
                    )
                    """
                )
                question_rows = [
                    ("q1", "Question one", "Multiple Choice", "Networking", "Moderate", "[]", "A"),
                    ("q2", "Question two", "Multiple Choice", "Networking", "Moderate", "[]", "B"),
                    ("q3", "Question three", "True or False", "Databases", "Easy", "[]", "True"),
                ]
                for index, (question_id, text, qtype, topic, difficulty, choices, answer) in enumerate(question_rows):
                    created_at = (
                        "2026-10-18 09:00:00" if index == 0
                        else "2026-10-18 13:00:00" if index == 1
                        else None
                    )
                    connection.execute(
                        "INSERT INTO generated_questions VALUES("
                        "?, ?, ?, ?, ?, ?, ?, 0.5, 0.8, 'builtin', 'Built-in examples', "
                        "NULL, 'lesson.txt', 'source excerpt', ?)",
                        (
                            question_id, text, qtype, topic, difficulty,
                            choices, answer, created_at,
                        ),
                    )
                connection.commit()
            finally:
                connection.close()

            with (
                patch.object(database, "STORAGE", storage),
                patch.object(database, "UPLOADS", uploads),
                patch.object(database, "DB_PATH", db_path),
            ):
                database.initialize()
                database.initialize()
                with database.connect() as db:
                    feedback_table = db.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name='question_feedback'"
                    ).fetchone()
                    sessions = db.execute(
                        "SELECT id, title, question_count FROM saved_sessions ORDER BY id"
                    ).fetchall()
                    questions = db.execute(
                        "SELECT id, question_text, session_id FROM generated_questions ORDER BY id"
                    ).fetchall()

            self.assertEqual(len(sessions), 2)
            self.assertIsNotNone(feedback_table)
            self.assertEqual(
                {session["title"]: session["question_count"] for session in sessions},
                {"Earlier questions": 1, "Questions from 2026-10-18": 2},
            )
            self.assertEqual(
                [(row["id"], row["question_text"]) for row in questions],
                [("q1", "Question one"), ("q2", "Question two"), ("q3", "Question three")],
            )
            self.assertEqual(questions[0]["session_id"], questions[1]["session_id"])
            self.assertNotEqual(questions[1]["session_id"], questions[2]["session_id"])

    def test_rebases_uploaded_material_paths_after_storage_folder_rename(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary) / "current-storage"
            uploads = storage / "uploads"
            uploads.mkdir(parents=True)
            upload_id = "migrated-upload"
            expected_file = uploads / f"{upload_id}.txt"
            expected_file.write_text("local course material", encoding="utf-8")
            db_path = storage / "assistant.sqlite3"
            connection = sqlite3.connect(db_path)
            try:
                connection.execute(
                    "CREATE TABLE uploads (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
                    "stored_path TEXT NOT NULL, bytes INTEGER NOT NULL, text_chars INTEGER NOT NULL, "
                    "uploaded_at TEXT NOT NULL DEFAULT (datetime('now')))"
                )
                connection.execute(
                    "INSERT INTO uploads(id, name, stored_path, bytes, text_chars) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (
                        upload_id,
                        "lesson.txt",
                        str(Path(temporary) / "old-storage" / "uploads" / f"{upload_id}.txt"),
                        21,
                        21,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            with (
                patch.object(database, "STORAGE", storage),
                patch.object(database, "UPLOADS", uploads),
                patch.object(database, "DB_PATH", db_path),
            ):
                database.initialize()
                with database.connect() as db:
                    row = db.execute(
                        "SELECT stored_path FROM uploads WHERE id=?", (upload_id,)
                    ).fetchone()

            self.assertEqual(Path(row["stored_path"]).resolve(), expected_file.resolve())

    def test_migrates_legacy_subject_column_without_losing_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            storage = Path(temporary)
            uploads = storage / "uploads"
            db_path = storage / "assistant.sqlite3"
            connection = sqlite3.connect(db_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE analyses (
                        id TEXT PRIMARY KEY,
                        subject TEXT NOT NULL,
                        predictions TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO analyses(id, subject, predictions) VALUES(?, ?, ?)",
                    ("legacy-id", "Database Systems", json.dumps([{"topic": "SQL"}])),
                )
                connection.commit()
            finally:
                connection.close()

            with (
                patch.object(database, "STORAGE", storage),
                patch.object(database, "UPLOADS", uploads),
                patch.object(database, "DB_PATH", db_path),
            ):
                database.initialize()
                with database.connect() as connection:
                    row = connection.execute(
                        "SELECT source_label, predictions, method, "
                        "training_dataset_id, training_dataset_name "
                        "FROM analyses WHERE id='legacy-id'"
                    ).fetchone()

            self.assertEqual(row["source_label"], "Uploaded materials")
            self.assertEqual(json.loads(row["predictions"]), [{"topic": "SQL"}])
            self.assertEqual(row["method"], "TF-IDF + SVM")
            self.assertEqual(row["training_dataset_id"], "builtin")
            self.assertEqual(row["training_dataset_name"], "Built-in examples")


if __name__ == "__main__":
    unittest.main()
