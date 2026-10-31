from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
STORAGE = Path(os.environ.get("TOPIC_ASSISTANT_DATA_DIR", ROOT / "storage"))
UPLOADS = STORAGE / "uploads"
DB_PATH = STORAGE / "assistant.sqlite3"

DEFAULT_SETTINGS = {
    "language": "English",
    "appearance": "orange",
    "question_generation": {
        "training_dataset_id": "builtin",
        "upload_ids": None,
        "topic": "All Topics",
        "question_type": "Multiple Choice",
        "difficulty": "Moderate",
        "number_of_questions": 10,
        "questions_per_topic": 3,
        "similarity_threshold": 0.82,
        "priority_weights": {
            "historical_frequency": 0.40,
            "course_importance": 0.30,
            "model_confidence": 0.20,
            "recent_occurrence": 0.10,
        },
    },
}


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    STORAGE.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS uploads (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                text_chars INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                source_label TEXT NOT NULL DEFAULT 'Uploaded materials',
                predictions TEXT NOT NULL,
                method TEXT NOT NULL DEFAULT 'TF-IDF + SVM + Naive Bayes ensemble',
                training_dataset_id TEXT NOT NULL DEFAULT 'builtin',
                training_dataset_name TEXT NOT NULL DEFAULT 'Built-in examples',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS training_datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS training_examples (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL REFERENCES training_datasets(id) ON DELETE CASCADE,
                topic TEXT NOT NULL,
                text TEXT NOT NULL,
                source_material_id TEXT,
                source_material_name TEXT,
                source_reference TEXT
            );
            CREATE TABLE IF NOT EXISTS generated_questions (
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
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                session_id TEXT
            );
            CREATE TABLE IF NOT EXISTS saved_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                topic TEXT NOT NULL,
                source_file TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                question_type TEXT NOT NULL,
                question_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS question_feedback (
                question_id TEXT PRIMARY KEY,
                rating INTEGER NOT NULL CHECK(rating IN (-1, 1)),
                question_type TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                wording_pattern TEXT NOT NULL,
                question_length TEXT NOT NULL DEFAULT 'medium',
                answer_length TEXT NOT NULL DEFAULT 'single-term',
                rated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        question_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(generated_questions)").fetchall()
        }
        if "session_id" not in question_columns:
            db.execute("ALTER TABLE generated_questions ADD COLUMN session_id TEXT")
        _backfill_saved_sessions(db)
        example_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(training_examples)").fetchall()
        }
        if "source_material_id" not in example_columns:
            db.execute("ALTER TABLE training_examples ADD COLUMN source_material_id TEXT")
        if "source_material_name" not in example_columns:
            db.execute("ALTER TABLE training_examples ADD COLUMN source_material_name TEXT")
        if "source_reference" not in example_columns:
            db.execute("ALTER TABLE training_examples ADD COLUMN source_reference TEXT")
        feedback_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(question_feedback)").fetchall()
        }
        if "question_length" not in feedback_columns:
            db.execute(
                "ALTER TABLE question_feedback ADD COLUMN question_length TEXT NOT NULL "
                "DEFAULT 'medium'"
            )
        if "answer_length" not in feedback_columns:
            db.execute(
                "ALTER TABLE question_feedback ADD COLUMN answer_length TEXT NOT NULL "
                "DEFAULT 'single-term'"
            )
        upload_root = UPLOADS.resolve()
        for upload in db.execute(
            "SELECT id, stored_path FROM uploads"
        ).fetchall():
            stored_path = Path(upload["stored_path"])
            suffix = stored_path.suffix.lower()
            if suffix not in {".pdf", ".docx", ".txt"}:
                continue
            try:
                if stored_path.resolve().parent == upload_root and stored_path.is_file():
                    continue
            except OSError:
                pass
            rebased_path = (UPLOADS / f"{upload['id']}{suffix}").resolve()
            if rebased_path.parent == upload_root and rebased_path.is_file():
                db.execute(
                    "UPDATE uploads SET stored_path=? WHERE id=?",
                    (str(rebased_path), upload["id"]),
                )
        analysis_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(analyses)").fetchall()
        }
        if "method" not in analysis_columns:
            db.execute(
                "ALTER TABLE analyses ADD COLUMN method TEXT NOT NULL "
                "DEFAULT 'TF-IDF + SVM'"
            )
        if "training_dataset_id" not in analysis_columns:
            db.execute(
                "ALTER TABLE analyses ADD COLUMN training_dataset_id TEXT NOT NULL "
                "DEFAULT 'builtin'"
            )
        if "training_dataset_name" not in analysis_columns:
            db.execute(
                "ALTER TABLE analyses ADD COLUMN training_dataset_name TEXT NOT NULL "
                "DEFAULT 'Built-in examples'"
            )
        if "subject" in analysis_columns and "source_label" not in analysis_columns:
            db.execute("ALTER TABLE analyses RENAME COLUMN subject TO source_label")
            db.execute(
                "UPDATE analyses SET source_label='Uploaded materials'"
            )
        db.execute(
            "INSERT OR IGNORE INTO settings(id, value) VALUES(1, ?)",
            (json.dumps(DEFAULT_SETTINGS),),
        )


def _backfill_saved_sessions(db: sqlite3.Connection) -> None:
    rows = db.execute(
        "SELECT id, created_at, topic, source_material_name, difficulty, question_type "
        "FROM generated_questions WHERE session_id IS NULL "
        "ORDER BY created_at, rowid"
    ).fetchall()
    groups: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        created_at = row["created_at"]
        try:
            date_key = datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            ).date().isoformat()
        except (AttributeError, TypeError, ValueError):
            date_key = ""
        group_key = date_key if date_key.strip() else "earlier"
        groups.setdefault(group_key, []).append(row)

    for group_key, questions in groups.items():
        session_id = f"legacy-{group_key}"
        first = questions[0]
        created_at = next(
            (
                row["created_at"]
                for row in questions
                if isinstance(row["created_at"], str) and row["created_at"].strip()
            ),
            "1970-01-01 00:00:00",
        )
        db.execute(
            "INSERT OR IGNORE INTO saved_sessions("
            "id, title, created_at, topic, source_file, difficulty, question_type, question_count"
            ") VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                "Earlier questions" if group_key == "earlier" else f"Questions from {group_key}",
                created_at,
                first["topic"] or "All Topics",
                first["source_material_name"] or "Unknown source",
                first["difficulty"] or "Moderate",
                first["question_type"] or "Multiple Choice",
                len(questions),
            ),
        )
        db.executemany(
            "UPDATE generated_questions SET session_id=? WHERE id=? AND session_id IS NULL",
            [(session_id, row["id"]) for row in questions],
        )
        db.execute(
            "UPDATE saved_sessions SET question_count=("
            "SELECT COUNT(*) FROM generated_questions WHERE session_id=?"
            ") WHERE id=?",
            (session_id, session_id),
        )


def get_settings() -> dict:
    with connect() as db:
        row = db.execute("SELECT value FROM settings WHERE id=1").fetchone()
    stored = json.loads(row["value"])
    settings = {
        key: stored.get(key, value)
        for key, value in DEFAULT_SETTINGS.items()
        if key != "question_generation"
    }
    question_generation = dict(DEFAULT_SETTINGS["question_generation"])
    saved_question_generation = stored.get("question_generation")
    if isinstance(saved_question_generation, dict):
        question_generation.update(saved_question_generation)
    priority_weights = dict(DEFAULT_SETTINGS["question_generation"]["priority_weights"])
    saved_weights = question_generation.get("priority_weights")
    if isinstance(saved_weights, dict):
        priority_weights.update(saved_weights)
    question_generation["priority_weights"] = priority_weights
    settings["question_generation"] = question_generation
    return settings


def save_settings(changes: dict) -> dict:
    current = get_settings()
    current.update(
        {key: value for key, value in changes.items() if key in DEFAULT_SETTINGS}
    )
    with connect() as db:
        db.execute(
            "INSERT INTO settings(id, value) VALUES(1, ?) "
            "ON CONFLICT(id) DO UPDATE SET value=excluded.value",
            (json.dumps(current),),
        )
    return current
