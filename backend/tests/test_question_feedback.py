import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import database, main


class QuestionFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        storage = Path(self.temporary.name)
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
        self.previous_batches = main._question_batches.copy()
        main._question_batches.clear()
        self.addCleanup(self._restore_batches)
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)

    def _restore_batches(self):
        main._question_batches.clear()
        main._question_batches.update(self.previous_batches)

    def test_saved_question_feedback_updates_global_profile_and_can_be_reset(self):
        with database.connect() as db:
            db.execute(
                "INSERT INTO saved_sessions("
                "id, title, topic, source_file, difficulty, question_type, question_count"
                ") VALUES('session-1', 'Saved set', 'Networks', 'lesson.txt', "
                "'Moderate', 'Multiple Choice', 1)"
            )
            db.execute(
                "INSERT INTO generated_questions("
                "id, question_text, question_type, topic, difficulty, choices, "
                "correct_answer, priority_score, confidence_score, source_dataset_id, "
                "source_dataset_name, source_material_name, source_reference, session_id"
                ") VALUES('question-1', 'Which concept is described by this passage?', "
                "'Multiple Choice', 'Networks', 'Moderate', '[]', 'router', 0.5, 0.8, "
                "'builtin', 'Built-in examples', 'lesson.txt', 'source passage', 'session-1')"
            )

        rated = self.client.post(
            "/api/questions/feedback",
            json={"question_id": "question-1", "rating": 1},
        )
        self.assertEqual(rated.status_code, 200, rated.text)
        self.assertEqual(rated.json()["ratings_recorded"], 1)
        self.assertEqual(rated.json()["positive_ratings"], 1)

        saved = self.client.get("/api/questions/saved").json()
        question = saved["sessions"][0]["questions"][0]
        self.assertEqual(question["feedback_rating"], 1)

        updated = self.client.post(
            "/api/questions/feedback",
            json={"question_id": "question-1", "rating": -1},
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["ratings_recorded"], 1)
        self.assertEqual(updated.json()["negative_ratings"], 1)

        cleared = self.client.delete("/api/questions/feedback")
        self.assertEqual(cleared.status_code, 200, cleared.text)
        self.assertEqual(cleared.json()["ratings_recorded"], 0)
        self.assertIsNone(
            self.client.get("/api/questions/saved").json()["sessions"][0]["questions"][0][
                "feedback_rating"
            ]
        )

    def test_generated_question_can_be_rated_before_saving(self):
        main._question_batches["generation-1"] = {
            "result": {
                "questions": [{
                    "id": "generated-1",
                    "question_text": "Identify the concept from this passage: sample.",
                    "question_type": "Identification",
                    "difficulty": "Easy",
                }]
            }
        }

        response = self.client.post(
            "/api/questions/feedback",
            json={
                "generation_id": "generation-1",
                "question_id": "generated-1",
                "rating": 1,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["ratings_recorded"], 1)
        self.assertIn(
            {"feature": "question_type", "value": "Identification", "positive": 1, "negative": 0},
            response.json()["learned_preferences"],
        )

    def test_feedback_requires_existing_question_and_valid_rating(self):
        missing = self.client.post(
            "/api/questions/feedback",
            json={"question_id": "missing", "rating": 1},
        )
        invalid = self.client.post(
            "/api/questions/feedback",
            json={"question_id": "missing", "rating": 0},
        )

        self.assertEqual(missing.status_code, 404)
        self.assertEqual(invalid.status_code, 422)


if __name__ == "__main__":
    unittest.main()
