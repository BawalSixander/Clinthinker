from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from sklearn import __version__ as sklearn_version

from . import database
from .database import UPLOADS, connect, get_settings, initialize, save_settings
from .ml import (
    BUILTIN_TRAINING_EXAMPLE_COUNT,
    BUILTIN_TRAINING_EXAMPLES,
    BUILTIN_TRAINING_TOPICS,
    PREDICTION_METHOD,
    augment_training_examples_with_exam_passages,
    classify_with_policy,
    dataset_hash,
    evaluate_classifier,
    get_decision_policy,
    predict,
)
from .parsing import MAX_UPLOAD_BYTES, extract_lesson_features, extract_text, read_limited
from .questions import (
    DEFAULT_PRIORITY_WEIGHTS,
    DEFAULT_SIMILARITY_THRESHOLD,
    extract_exam_stem_templates,
    generate_questions,
    question_preference_features,
    rank_lesson_likelihood,
)

APP_NAME = "Clinthinker©"
APP_VERSION = "2.0.1"
UPDATE_MANIFEST_URL = "https://seanlordtrillo-ops.github.io/Clinthinker/version.json"
app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
)

_bundled_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
WEB_DIR = Path(
    os.environ.get(
        "TOPIC_ASSISTANT_WEB_DIR",
        _bundled_root / "frontend_dist"
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[2] / "frontend" / "dist",
    )
)


@app.on_event("startup")
def startup() -> None:
    initialize()


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500000)
    training_dataset_id: str = Field(default="builtin", min_length=1, max_length=100)


class LabeledUpload(BaseModel):
    upload_id: str = Field(min_length=1, max_length=100)
    topic: str = Field(min_length=1, max_length=100)


class EvaluateDatasetRequest(BaseModel):
    training_dataset_id: str = Field(min_length=1, max_length=100)


class DocumentDatasetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    examples: list[LabeledUpload] = Field(min_length=2, max_length=500)


class PriorityWeights(BaseModel):
    historical_frequency: float = Field(default=0.40, ge=0, le=1)
    course_importance: float = Field(default=0.30, ge=0, le=1)
    model_confidence: float = Field(default=0.20, ge=0, le=1)
    recent_occurrence: float = Field(default=0.10, ge=0, le=1)

    @model_validator(mode="after")
    def weights_must_total_one(self):
        if abs(sum(self.model_dump().values()) - 1.0) > 0.001:
            raise ValueError("Priority weights must add up to 100%.")
        return self


class QuestionGenerationSettings(BaseModel):
    training_dataset_id: str = Field(default="builtin", min_length=1, max_length=100)
    upload_ids: list[str] | None = None
    topic: str = Field(default="All Topics", min_length=1, max_length=100)
    question_type: Literal["Multiple Choice", "True or False", "Identification"] = "Multiple Choice"
    difficulty: Literal["Easy", "Moderate", "Difficult"] = "Moderate"
    number_of_questions: int = Field(default=10, ge=1, le=50)
    questions_per_topic: int = Field(default=3, ge=1, le=50)
    similarity_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD, ge=0.5, le=1)
    priority_weights: PriorityWeights = Field(default_factory=PriorityWeights)


class SettingsUpdate(BaseModel):
    language: str | None = Field(default=None, max_length=40)
    appearance: Literal["orange", "pink", "regal"] | None = None
    question_generation: QuestionGenerationSettings | None = None


class GenerateQuestionsRequest(BaseModel):
    training_dataset_id: str = Field(min_length=1, max_length=100)
    upload_ids: list[str] = Field(min_length=1)
    lesson_upload_id: str | None = Field(default=None, min_length=1, max_length=100)
    lesson_upload_ids: list[str] = Field(default_factory=list)
    lesson_only: bool = False
    use_exam_style: bool = True
    topic: str = Field(default="All Topics", min_length=1, max_length=100)
    question_type: Literal["Multiple Choice", "True or False", "Identification"]
    difficulty: Literal["Easy", "Moderate", "Difficult"]
    number_of_questions: int = Field(default=10, ge=1, le=50)
    questions_per_topic: int = Field(default=3, ge=1, le=50)
    similarity_threshold: float = Field(
        default=DEFAULT_SIMILARITY_THRESHOLD, ge=0.5, le=1
    )
    priority_weights: PriorityWeights = Field(default_factory=PriorityWeights)


class SaveQuestionsRequest(BaseModel):
    generation_id: str = Field(min_length=1, max_length=100)
    question_ids: list[str] = Field(min_length=1, max_length=50)


class QuestionFeedbackRequest(BaseModel):
    question_id: str = Field(min_length=1, max_length=100)
    rating: int = Field(ge=-1, le=1)
    generation_id: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def rating_must_be_positive_or_negative(self):
        if self.rating not in {-1, 1}:
            raise ValueError("Choose either positive or negative feedback.")
        return self


class RenameSavedSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


_question_batches: OrderedDict[str, dict] = OrderedDict()
_MAX_QUESTION_BATCHES = 8


def _analyze(request: PredictionRequest, progress_callback=None) -> dict:
    dataset_id = request.training_dataset_id
    if dataset_id == "builtin":
        dataset_name = "Built-in examples"
        training_examples = None
    else:
        with connect() as db:
            dataset = db.execute(
                "SELECT name FROM training_datasets WHERE id=?", (dataset_id,)
            ).fetchone()
            if not dataset:
                raise HTTPException(status_code=404, detail="Training dataset not found.")
            rows = db.execute(
                "SELECT text, topic FROM training_examples "
                "WHERE dataset_id=? ORDER BY rowid",
                (dataset_id,),
            ).fetchall()
        training_examples = [(row["text"], row["topic"]) for row in rows]
        dataset_name = dataset["name"]
    predictions = predict(
        request.text,
        training_examples=training_examples,
        progress_callback=progress_callback,
    )
    decision_policy = get_decision_policy(
        {"id": dataset_id, "examples": (
            BUILTIN_TRAINING_EXAMPLES if training_examples is None else training_examples
        )}
    )
    classification = classify_with_policy(
        request.text,
        training_examples=training_examples,
        dataset_id=dataset_id,
        decision_policy=decision_policy,
    )
    predictions = [
        {
            **prediction,
            "margin": classification["margin"],
            "models_agree": classification["models_agree"],
            "abstain_reason": classification["abstain_reason"],
        }
        for prediction in predictions
    ]
    analysis_id = str(uuid.uuid4())
    with connect() as db:
        db.execute(
            "INSERT INTO analyses("
            "id, source_label, predictions, method, training_dataset_id, "
            "training_dataset_name) VALUES(?, ?, ?, ?, ?, ?)",
            (
                analysis_id,
                "Uploaded materials",
                json.dumps(predictions),
                PREDICTION_METHOD,
                dataset_id,
                dataset_name,
            ),
        )
    return {
        "id": analysis_id,
        "label": "Uploaded materials",
        "method": PREDICTION_METHOD,
        "training_dataset_id": dataset_id,
        "training_dataset_name": dataset_name,
        "predictions": predictions,
        **classification,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": APP_NAME, "storage": "local", "version": APP_VERSION}


def _version_parts(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value):
        return None
    return tuple(int(part) for part in value.split("."))


def _fetch_update_manifest() -> dict:
    with urllib.request.urlopen(UPDATE_MANIFEST_URL, timeout=5) as response:
        manifest = json.loads(response.read().decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Invalid update manifest.")
    return manifest


def _valid_installer_url(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("https://github.com/seanlordtrillo-ops/Clinthinker/")
    )


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value) is not None


@app.get("/api/updates/check")
def check_for_updates() -> dict:
    try:
        manifest = _fetch_update_manifest()
        version = manifest.get("version")
        installer_url = manifest.get("installer_url")
        sha256 = manifest.get("sha256")
        current_parts = _version_parts(APP_VERSION)
        remote_parts = _version_parts(version)
        if (
            current_parts is None
            or remote_parts is None
            or remote_parts <= current_parts
            or not _valid_installer_url(installer_url)
            or (sha256 is not None and not _valid_sha256(sha256))
        ):
            return {"available": False}
        return {
            "available": True,
            "version": version,
            "changelog": manifest.get("changelog", ""),
        }
    except Exception:
        return {"available": False}


@app.post("/api/updates/install")
def install_update() -> dict:
    try:
        manifest = _fetch_update_manifest()
        remote_version = _version_parts(manifest.get("version"))
        current_version = _version_parts(APP_VERSION)
        if remote_version is None or current_version is None or remote_version <= current_version:
            raise HTTPException(status_code=409, detail="No newer update is available.")
        installer_url = manifest.get("installer_url")
        if not _valid_installer_url(installer_url):
            raise HTTPException(status_code=422, detail="The installer URL is invalid.")
        sha256 = manifest.get("sha256")
        if sha256 is not None and not _valid_sha256(sha256):
            raise HTTPException(status_code=422, detail="The installer checksum is invalid.")

        update_directory = Path(tempfile.mkdtemp(prefix="clinthinker-update-"))
        installer_path = update_directory / "Clinthinker-Setup.exe"
        digest = hashlib.sha256()
        with urllib.request.urlopen(installer_url, timeout=30) as response:
            with installer_path.open("wb") as installer_file:
                while chunk := response.read(64 * 1024):
                    installer_file.write(chunk)
                    digest.update(chunk)
        if sha256 is not None and digest.hexdigest().casefold() != sha256.casefold():
            raise HTTPException(status_code=422, detail="The installer checksum does not match.")

        if not database.DB_PATH.is_file():
            raise FileNotFoundError("The local database file was not found.")
        backup_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup_path = database.STORAGE / f"assistant-{backup_time}.sqlite3"
        shutil.copy2(database.DB_PATH, backup_path)
        subprocess.Popen([str(installer_path)], cwd=str(update_directory))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail="Update failed.") from error
    return {"started": True}


def _store_upload(name: str, content: bytes, deduplicate: bool = False) -> dict:
    text = extract_text(name, content)
    if deduplicate:
        with connect() as db:
            candidates = db.execute(
                "SELECT id, name, stored_path, bytes, text_chars, uploaded_at "
                "FROM uploads WHERE bytes=? ORDER BY uploaded_at DESC",
                (len(content),),
            ).fetchall()
        for candidate in candidates:
            path = Path(candidate["stored_path"]).resolve()
            if path.parent == UPLOADS.resolve() and path.is_file():
                if path.read_bytes() == content:
                    return {
                        "id": candidate["id"],
                        "name": candidate["name"],
                        "bytes": candidate["bytes"],
                        "text_chars": candidate["text_chars"],
                        "text": text,
                        "duplicate": True,
                    }
    upload_id = str(uuid.uuid4())
    destination = UPLOADS / f"{upload_id}{Path(name).suffix.lower()}"
    destination.write_bytes(content)
    try:
        with connect() as db:
            db.execute(
                "INSERT INTO uploads(id, name, stored_path, bytes, text_chars) "
                "VALUES(?, ?, ?, ?, ?)",
                (upload_id, name, str(destination), len(content), len(text)),
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {
        "id": upload_id,
        "name": name,
        "bytes": len(content),
        "text_chars": len(text),
        "text": text,
        "duplicate": False,
    }


async def _receive_upload(file: UploadFile, deduplicate: bool = False) -> dict:
    content = await read_limited(file)
    name = Path((file.filename or "upload").replace("\\", "/")).name
    return await run_in_threadpool(_store_upload, name, content, deduplicate)


@app.post("/api/upload")
async def upload_material(file: UploadFile = File(...)) -> dict:
    return await _receive_upload(file)


@app.post("/api/lessons/upload")
async def upload_lesson(file: UploadFile = File(...)) -> dict:
    lesson = await _receive_upload(file, deduplicate=True)
    lesson.pop("text")
    return lesson


@app.get("/api/uploads")
def list_uploads() -> list[dict]:
    with connect() as db:
        rows = db.execute(
            "SELECT id, name, bytes, text_chars, uploaded_at FROM uploads "
            "ORDER BY uploaded_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/training-datasets")
def list_training_datasets() -> list[dict]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT d.id, d.name, d.source_type, d.created_at,
                   COUNT(e.id) AS example_count,
                   COUNT(DISTINCT e.topic) AS topic_count
            FROM training_datasets d
            LEFT JOIN training_examples e ON e.dataset_id=d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC, d.rowid DESC
            """
        ).fetchall()
    return [
        {
            "id": "builtin",
            "name": "Built-in examples",
            "source_type": "built-in",
            "example_count": BUILTIN_TRAINING_EXAMPLE_COUNT,
            "topic_count": len(BUILTIN_TRAINING_TOPICS),
            "topics": BUILTIN_TRAINING_TOPICS,
        },
        *[dict(row) for row in rows],
    ]


@app.post("/api/training-datasets/evaluate")
def evaluate_training_dataset(request: EvaluateDatasetRequest) -> dict:
    if request.training_dataset_id == "builtin":
        dataset_name = "Built-in examples"
        examples = list(BUILTIN_TRAINING_EXAMPLES)
    else:
        with connect() as db:
            dataset = db.execute(
                "SELECT name FROM training_datasets WHERE id=?",
                (request.training_dataset_id,),
            ).fetchone()
            if not dataset:
                raise HTTPException(status_code=404, detail="Training dataset not found.")
            rows = db.execute(
                "SELECT text, topic FROM training_examples "
                "WHERE dataset_id=? ORDER BY rowid",
                (request.training_dataset_id,),
            ).fetchall()
        dataset_name = dataset["name"]
        examples = [(row["text"], row["topic"]) for row in rows]
    try:
        report = evaluate_classifier(examples)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    policy = report["decision_policy"]
    with connect() as db:
        db.execute(
            "INSERT INTO model_calibration("
            "dataset_id, dataset_hash, svm_temperature, ensemble_temperature, "
            "confidence_threshold, min_margin, disagreement_abstain, "
            "target_precision, low_reliability, sklearn_version) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(dataset_id) DO UPDATE SET "
            "dataset_hash=excluded.dataset_hash, "
            "svm_temperature=excluded.svm_temperature, "
            "ensemble_temperature=excluded.ensemble_temperature, "
            "confidence_threshold=excluded.confidence_threshold, "
            "min_margin=excluded.min_margin, "
            "disagreement_abstain=excluded.disagreement_abstain, "
            "target_precision=excluded.target_precision, "
            "low_reliability=excluded.low_reliability, "
            "sklearn_version=excluded.sklearn_version, "
            "created_at=datetime('now')",
            (
                request.training_dataset_id,
                dataset_hash(examples),
                report["calibration"]["svm_temperature"],
                report["calibration"]["ensemble_temperature"],
                policy["confidence_threshold"],
                policy["min_margin"],
                int(policy["disagreement_abstain"]),
                policy["target_precision"],
                int(policy["low_reliability"]),
                sklearn_version,
            ),
        )
    return {
        **report,
        "training_dataset_id": request.training_dataset_id,
        "training_dataset_name": dataset_name,
    }


@app.get("/api/training-datasets/{dataset_id}/topics")
def list_training_topics(dataset_id: str) -> dict:
    if dataset_id == "builtin":
        return {"topics": BUILTIN_TRAINING_TOPICS}
    with connect() as db:
        dataset = db.execute(
            "SELECT id FROM training_datasets WHERE id=?", (dataset_id,)
        ).fetchone()
        if not dataset:
            raise HTTPException(status_code=404, detail="Training dataset not found.")
        rows = db.execute(
            "SELECT DISTINCT topic FROM training_examples "
            "WHERE dataset_id=? ORDER BY topic COLLATE NOCASE",
            (dataset_id,),
        ).fetchall()
    return {"topics": [row["topic"] for row in rows]}


def _validate_training_examples(examples: list[tuple[str, str]]) -> None:
    if not examples:
        raise HTTPException(status_code=422, detail="The dataset contains no labeled examples.")
    topics = {topic.strip() for _, topic in examples if topic.strip()}
    if any(not text.strip() or not topic.strip() for text, topic in examples):
        raise HTTPException(
            status_code=422,
            detail="Every training row needs non-empty text and a topic label.",
        )
    if len(topics) < 2:
        raise HTTPException(
            status_code=422,
            detail="Training data must contain at least two different topic labels.",
        )


def _save_training_dataset(
    name: str,
    source_type: str,
    examples: list[tuple[str, str]],
    example_sources: list[dict | None] | None = None,
) -> dict:
    dataset_id = str(uuid.uuid4())
    with connect() as db:
        db.execute(
            "INSERT INTO training_datasets(id, name, source_type) VALUES(?, ?, ?)",
            (dataset_id, name, source_type),
        )
        db.executemany(
            "INSERT INTO training_examples("
            "id, dataset_id, topic, text, source_material_id, source_material_name, source_reference"
            ") VALUES(?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    str(uuid.uuid4()), dataset_id, topic.strip(), text.strip(),
                    (example_sources[index] or {}).get("id") if example_sources else None,
                    (example_sources[index] or {}).get("name") if example_sources else None,
                    "Labeled uploaded course material" if example_sources else None,
                )
                for index, (text, topic) in enumerate(examples)
            ],
        )
    return {
        "id": dataset_id,
        "name": name,
        "source_type": source_type,
        "example_count": len(examples),
        "topic_count": len({topic.strip() for _, topic in examples}),
    }


@app.post("/api/training-datasets/csv")
async def import_training_csv(
    file: UploadFile = File(...),
    name: str | None = Form(default=None, max_length=100),
) -> dict:
    filename = Path((file.filename or "training-data.csv").replace("\\", "/")).name
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=415, detail="Choose a CSV training dataset.")
    content = await read_limited(file, MAX_UPLOAD_BYTES)
    try:
        decoded = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded), strict=True, restkey="_extra")
        header_values = reader.fieldnames or []
        normalized_headers = [header.strip().lower() for header in header_values if header and header.strip()]
    except (UnicodeDecodeError, csv.Error) as error:
        raise HTTPException(
            status_code=422, detail="The CSV must be a valid UTF-8 file."
        ) from error
    if len(normalized_headers) != len(set(normalized_headers)):
        raise HTTPException(status_code=422, detail="The CSV cannot contain duplicate column names.")
    headers = {
        header.strip().lower(): header
        for header in header_values
        if header and header.strip()
    }
    if "topic" not in headers or "text" not in headers:
        raise HTTPException(
            status_code=422,
            detail="CSV must have a header row with 'topic' and 'text' columns.",
        )
    try:
        examples = []
        for row in reader:
            if row.get("_extra") is not None:
                raise HTTPException(
                    status_code=422,
                    detail="A CSV row has more values than its header columns.",
                )
            examples.append(
                ((row.get(headers["text"]) or ""), (row.get(headers["topic"]) or ""))
            )
            if len(examples) > 50000:
                raise HTTPException(
                    status_code=413, detail="CSV datasets are limited to 50,000 rows."
                )
    except csv.Error as error:
        raise HTTPException(status_code=422, detail="Could not parse the CSV rows.") from error
    _validate_training_examples(examples)
    dataset_name = (name or Path(filename).stem).strip()
    if not dataset_name:
        raise HTTPException(status_code=422, detail="Enter a name for this dataset.")
    return _save_training_dataset(dataset_name, "csv", examples)


@app.post("/api/training-datasets/documents")
def create_document_dataset(request: DocumentDatasetRequest) -> dict:
    if len({example.upload_id for example in request.examples}) != len(request.examples):
        raise HTTPException(status_code=422, detail="Select each document only once.")
    topics = {example.topic.strip() for example in request.examples if example.topic.strip()}
    if len(topics) < 2:
        raise HTTPException(
            status_code=422,
            detail="Assign at least two different topic labels to your documents.",
        )
    examples = []
    example_sources = []
    with connect() as db:
        for item in request.examples:
            row = db.execute(
                "SELECT name, stored_path FROM uploads WHERE id=?", (item.upload_id,)
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="A selected document was not found.")
            path = Path(row["stored_path"]).resolve()
            if path.parent != UPLOADS.resolve():
                raise HTTPException(status_code=500, detail="Stored upload path is invalid.")
            try:
                text = extract_text(row["name"], path.read_bytes())
            except OSError as error:
                raise HTTPException(
                    status_code=500, detail=f"Could not read {row['name']}: {error}"
                ) from error
            examples.append((text, item.topic))
            example_sources.append({"id": item.upload_id, "name": row["name"]})
    _validate_training_examples(examples)
    dataset_name = request.name.strip()
    if not dataset_name:
        raise HTTPException(status_code=422, detail="Enter a name for this dataset.")
    return _save_training_dataset(dataset_name, "documents", examples, example_sources)


@app.delete("/api/training-datasets/{dataset_id}")
def delete_training_dataset(dataset_id: str) -> dict:
    if dataset_id == "builtin":
        raise HTTPException(status_code=409, detail="Built-in examples cannot be deleted.")
    with connect() as db:
        db.execute(
            "DELETE FROM model_calibration WHERE dataset_id=?",
            (dataset_id,),
        )
        cursor = db.execute("DELETE FROM training_datasets WHERE id=?", (dataset_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Training dataset not found.")
    return {"deleted": dataset_id}


@app.get("/api/uploads/{upload_id}/text")
def read_uploaded_text(upload_id: str) -> dict:
    with connect() as db:
        row = db.execute(
            "SELECT name, stored_path FROM uploads WHERE id=?", (upload_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Uploaded file not found.")
    path = Path(row["stored_path"]).resolve()
    if path.parent != UPLOADS.resolve():
        raise HTTPException(status_code=500, detail="Stored upload path is invalid.")
    try:
        text = extract_text(row["name"], path.read_bytes())
    except OSError as error:
        raise HTTPException(
            status_code=500, detail=f"Could not read saved upload: {error}"
        ) from error
    return {"id": upload_id, "name": row["name"], "text": text}


@app.delete("/api/uploads/{upload_id}")
def delete_upload(upload_id: str) -> dict:
    with connect() as db:
        row = db.execute(
            "SELECT stored_path FROM uploads WHERE id=?", (upload_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Uploaded file not found.")
        path = Path(row["stored_path"]).resolve()
        if path.parent != UPLOADS.resolve():
            raise HTTPException(status_code=500, detail="Stored upload path is invalid.")
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            raise HTTPException(
                status_code=500, detail=f"Could not delete upload: {error}"
            ) from error
        db.execute("DELETE FROM uploads WHERE id=?", (upload_id,))
    return {"deleted": upload_id}


@app.post("/api/predict/stream")
async def predict_stream(request: PredictionRequest) -> StreamingResponse:
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Course material text is required.")

    async def events():
        loop = asyncio.get_running_loop()
        updates: asyncio.Queue[dict] = asyncio.Queue()

        def report(update: dict) -> None:
            loop.call_soon_threadsafe(updates.put_nowait, update)

        task = asyncio.create_task(run_in_threadpool(_analyze, request, report))
        try:
            while not task.done() or not updates.empty():
                try:
                    update = await asyncio.wait_for(updates.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                yield f"event: progress\ndata: {json.dumps(update)}\n\n"
            result = await task
        except (ValueError, HTTPException) as error:
            message = error.detail if isinstance(error, HTTPException) else str(error)
            yield f"event: error\ndata: {json.dumps({'detail': message})}\n\n"
            return
        yield f"event: result\ndata: {json.dumps(result)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/analysis/latest")
def latest_analysis() -> dict | None:
    with connect() as db:
        row = db.execute(
            "SELECT id, source_label, predictions, method, training_dataset_id, "
            "training_dataset_name, created_at FROM analyses "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "label": row["source_label"],
        "method": row["method"],
        "training_dataset_id": row["training_dataset_id"],
        "training_dataset_name": row["training_dataset_name"],
        "predictions": json.loads(row["predictions"]),
        "created_at": row["created_at"],
    }


@app.get("/api/settings")
def read_settings() -> dict:
    return get_settings()


@app.put("/api/settings")
def update_settings(request: SettingsUpdate) -> dict:
    return save_settings(request.model_dump(exclude_none=True))


def database_recent_question_topics() -> list[dict]:
    with connect() as db:
        return [
            dict(row)
            for row in db.execute(
                "SELECT topic FROM generated_questions "
                "WHERE created_at >= datetime('now', '-30 days') "
                "ORDER BY created_at DESC, rowid DESC LIMIT 100"
            ).fetchall()
        ]


@app.post("/api/questions/generate")
def generate_practice_questions(request: GenerateQuestionsRequest) -> dict:
    if len(set(request.upload_ids)) != len(request.upload_ids):
        raise HTTPException(status_code=422, detail="Select each course material only once.")
    if len(set(request.lesson_upload_ids)) != len(request.lesson_upload_ids):
        raise HTTPException(status_code=422, detail="Select each lesson only once.")
    lesson_ids = list(dict.fromkeys(
        request.lesson_upload_ids
        + ([request.lesson_upload_id] if request.lesson_upload_id else [])
    ))
    if any(lesson_id not in request.upload_ids for lesson_id in lesson_ids):
        raise HTTPException(status_code=422, detail="Include every lesson in the source selection.")
    if lesson_ids and set(request.upload_ids) != set(lesson_ids):
        raise HTTPException(
            status_code=422,
            detail="Lesson generation accepts only selected lesson files as content sources.",
        )
    if request.lesson_only and not lesson_ids:
        raise HTTPException(status_code=422, detail="Select at least one lesson.")
    try:
        with connect() as db:
            if request.training_dataset_id == "builtin":
                dataset_name = "Built-in examples"
                examples = list(BUILTIN_TRAINING_EXAMPLES)
                dataset_rows = []
            else:
                dataset = db.execute(
                    "SELECT name FROM training_datasets WHERE id=?",
                    (request.training_dataset_id,),
                ).fetchone()
                if not dataset:
                    raise HTTPException(status_code=404, detail="Training dataset not found.")
                dataset_name = dataset["name"]
                dataset_rows = db.execute(
                    "SELECT text, topic, source_material_id, source_material_name "
                    "FROM training_examples WHERE dataset_id=? ORDER BY rowid",
                    (request.training_dataset_id,),
                ).fetchall()
                examples = [(row["text"], row["topic"]) for row in dataset_rows]
            upload_rows = []
            for upload_id in request.upload_ids:
                upload = db.execute(
                    "SELECT id, name, stored_path FROM uploads WHERE id=?", (upload_id,)
                ).fetchone()
                if not upload:
                    raise HTTPException(status_code=404, detail="A selected course material was not found.")
                upload_rows.append(dict(upload))
            lesson_uploads = [
                upload for upload in upload_rows if upload["id"] in lesson_ids
            ]
            lesson_texts = {}
            lesson_features = {
                "emphasized_terms": [],
                "heading_terms": [],
                "list_terms": [],
            }
            for lesson_upload in lesson_uploads:
                lesson_path = Path(lesson_upload["stored_path"]).resolve()
                if lesson_path.parent != UPLOADS.resolve():
                    raise HTTPException(status_code=500, detail="The saved lesson path is invalid.")
                try:
                    lesson_content = lesson_path.read_bytes()
                    lesson_texts[lesson_upload["id"]] = extract_text(
                        lesson_upload["name"], lesson_content
                    )
                    features = extract_lesson_features(
                        lesson_upload["name"], lesson_content
                    )
                    for key in lesson_features:
                        lesson_features[key].extend(features[key])
                except HTTPException:
                    raise
                except OSError as error:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Could not read lesson {lesson_upload['name']}: {error}",
                    ) from error
            saved_texts = [
                row["question_text"]
                for row in db.execute(
                    "SELECT question_text FROM generated_questions"
                ).fetchall()
            ]
            preference_feedback = [
                dict(row)
                for row in db.execute(
                    "SELECT rating, question_type, difficulty, wording_pattern "
                    "FROM question_feedback"
                ).fetchall()
            ]
        saved_texts.extend(
            item["question_text"]
            for batch in _question_batches.values()
            for item in batch["result"]["questions"]
        )
        decision_policy = get_decision_policy(
            {"id": request.training_dataset_id, "examples": examples}
        )

        source_upload_rows = list(upload_rows)
        if lesson_ids:
            with connect() as db:
                exam_rows = db.execute(
                    "SELECT id, name, stored_path FROM uploads"
                ).fetchall()
            existing_upload_ids = set(lesson_ids)
            source_upload_rows.extend(
                dict(row) for row in exam_rows
                if row["id"] not in existing_upload_ids
                and Path(row["name"]).suffix.lower() == ".pdf"
                and re.search(r"(?:^|[_\s-])exam(?:[_\s.-]|$)", row["name"], re.I)
            )

        sources = []
        source_characters = 0
        exam_sources = []
        skipped_exam_files = []
        exam_characters = 0
        for upload in source_upload_rows:
            is_lesson = upload["id"] in lesson_ids
            is_exam = bool(
                re.search(r"(?:^|[_\s-])exam(?:[_\s.-]|$)", upload["name"], re.I)
                and Path(upload["name"]).suffix.lower() == ".pdf"
                and not is_lesson
            )
            path = Path(upload["stored_path"]).resolve()
            if path.parent != UPLOADS.resolve():
                raise HTTPException(status_code=500, detail="A saved course material could not be read.")
            try:
                text = lesson_texts[upload["id"]] if is_lesson else extract_text(
                    upload["name"], path.read_bytes()
                )
            except HTTPException as error:
                if lesson_ids and is_exam:
                    skipped_exam_files.append(upload["name"])
                    continue
                raise HTTPException(
                    status_code=error.status_code,
                    detail=f"Could not extract readable text from {upload['name']}.",
                ) from error
            except OSError as error:
                raise HTTPException(
                    status_code=500, detail=f"Could not read saved course material {upload['name']}."
                ) from error
            if lesson_ids and is_exam:
                if exam_characters + len(text) <= 500_000:
                    exam_sources.append(
                        {"id": upload["id"], "name": upload["name"], "text": text}
                    )
                    exam_characters += len(text)
                else:
                    skipped_exam_files.append(upload["name"])
                continue
            source_characters += len(text)
            if source_characters > 500_000:
                raise HTTPException(
                    status_code=413,
                    detail="Selected course materials exceed the 500,000 character analysis limit.",
                )
            sources.append({
                "id": upload["id"],
                "name": upload["name"],
                "text": text,
                "kind": "lesson" if is_lesson else "course",
            })
        if not lesson_ids:
            for index, row in enumerate(dataset_rows, start=1):
                sources.append(
                    {
                        "id": row["source_material_id"],
                        "name": row["source_material_name"] or f"{dataset_name} · Example {index}",
                        "text": row["text"],
                        "kind": "training",
                    }
                )

        training_metrics = {}
        classifier_examples = examples
        exam_passages_added = 0
        if lesson_ids:
            classifier_examples, exam_passages_added = (
                augment_training_examples_with_exam_passages(
                    examples,
                    exam_sources,
                    training_metrics,
                    dataset_id=request.training_dataset_id,
                    decision_policy=decision_policy,
                )
            )
        style_templates = (
            extract_exam_stem_templates(
                exam_sources,
                classifier_examples,
                training_metrics,
                dataset_id=request.training_dataset_id,
                decision_policy=decision_policy,
            )
            if lesson_ids and request.use_exam_style else []
        )
        result = generate_questions(
            requested_count=request.number_of_questions,
            question_type=request.question_type,
            difficulty=request.difficulty,
            topic_filter=request.topic,
            per_topic=request.questions_per_topic,
            training_examples=classifier_examples,
            dataset_id=request.training_dataset_id,
            dataset_name=dataset_name,
            sources=sources,
            lesson_mode=bool(lesson_ids),
            style_templates=style_templates,
            training_metrics=training_metrics,
            saved_question_texts=saved_texts,
            preference_feedback=preference_feedback,
            similarity_threshold=request.similarity_threshold,
            priority_weights=request.priority_weights.model_dump(),
            decision_policy=decision_policy,
        )
        if lesson_ids:
            lesson_text = "\n\n".join(
                lesson_texts[upload["id"]] for upload in lesson_uploads
            )
            recent_topics = [row["topic"] for row in database_recent_question_topics()]
            lesson_ranking = rank_lesson_likelihood(
                result["questions"],
                lesson_text,
                classifier_examples,
                exam_sources=exam_sources,
                recent_question_topics=recent_topics,
                emphasized_terms=lesson_features["emphasized_terms"],
                heading_terms=lesson_features["heading_terms"],
                list_terms=lesson_features["list_terms"],
                training_metrics=training_metrics,
                dataset_id=request.training_dataset_id,
                decision_policy=decision_policy,
            )
            result.update(lesson_ranking)
            result["skipped_exam_files"] = skipped_exam_files
            result["classifier_training"] = {
                "mode": "on-demand during generation; fitted models are cached in memory",
                "fit_seconds": round(training_metrics.get("fit_seconds", 0.0), 3),
                "fit_count": training_metrics.get("fit_count", 0),
                "cache_hits": training_metrics.get("cache_hits", 0),
                "exam_passages_added": exam_passages_added,
                "minimum_exam_label_confidence": decision_policy["confidence_threshold"],
            }
            result["style_reference_enabled"] = request.use_exam_style
            result["style_reference_files"] = len(exam_sources) if request.use_exam_style else 0
            result["style_template_count"] = len(style_templates)
            lesson_names = [upload["name"] for upload in lesson_uploads]
            result["lesson_names"] = lesson_names
            result["lesson_name"] = lesson_names[0] if len(lesson_names) == 1 else None
        result["preference_learning"] = {
            "ratings_recorded": len(preference_feedback),
            "applied": bool(preference_feedback),
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    generation_id = str(uuid.uuid4())
    result["generation_id"] = generation_id
    _question_batches[generation_id] = {
        "result": result,
        "dataset_id": request.training_dataset_id,
        "upload_ids": request.upload_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "topic": request.topic,
        "source_file": (
            ", ".join(upload["name"] for upload in lesson_uploads)
            if lesson_ids else ", ".join(upload["name"] for upload in upload_rows)
        ),
        "difficulty": request.difficulty,
        "question_type": request.question_type,
    }
    _question_batches.move_to_end(generation_id)
    while len(_question_batches) > _MAX_QUESTION_BATCHES:
        _question_batches.popitem(last=False)
    public_result = {**result}
    public_result["questions"] = [
        {
            key: value for key, value in item.items()
            if key not in {"correct_answer", "support_excerpt", "source_content"}
        }
        for item in result["questions"]
    ]
    return public_result


@app.get("/api/questions/preferences")
def get_question_preferences() -> dict:
    with connect() as db:
        rows = db.execute(
            "SELECT question_type, difficulty, wording_pattern, question_length, "
            "answer_length, rating "
            "FROM question_feedback"
        ).fetchall()
    grouped: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        for feature, value in (
            ("question_type", row["question_type"]),
            ("difficulty", row["difficulty"]),
            ("wording_pattern", row["wording_pattern"]),
            ("question_length", row["question_length"]),
            ("answer_length", row["answer_length"]),
        ):
            counts = grouped.setdefault(
                (feature, value), {"positive": 0, "negative": 0}
            )
            counts["positive" if row["rating"] > 0 else "negative"] += 1
    learned = [
        {"feature": feature, "value": value, **counts}
        for (feature, value), counts in grouped.items()
        if counts["positive"] > counts["negative"]
    ]
    learned.sort(
        key=lambda item: (
            -(item["positive"] - item["negative"]),
            item["feature"],
            item["value"],
        )
    )
    return {
        "ratings_recorded": len(rows),
        "positive_ratings": sum(row["rating"] > 0 for row in rows),
        "negative_ratings": sum(row["rating"] < 0 for row in rows),
        "learned_preferences": learned,
    }


@app.post("/api/questions/feedback")
def save_question_feedback(request: QuestionFeedbackRequest) -> dict:
    question = None
    if request.generation_id:
        batch = _question_batches.get(request.generation_id)
        if batch:
            question = next(
                (
                    item for item in batch["result"]["questions"]
                    if item["id"] == request.question_id
                ),
                None,
            )
    if question is None:
        with connect() as db:
            row = db.execute(
                "SELECT id, question_text, question_type, difficulty, correct_answer "
                "FROM generated_questions WHERE id=?",
                (request.question_id,),
            ).fetchone()
        question = dict(row) if row else None
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found.")
    features = question_preference_features(question)
    with connect() as db:
        db.execute(
            "INSERT INTO question_feedback("
            "question_id, rating, question_type, difficulty, wording_pattern, "
            "question_length, answer_length"
            ") VALUES(?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(question_id) DO UPDATE SET "
            "rating=excluded.rating, question_type=excluded.question_type, "
            "difficulty=excluded.difficulty, wording_pattern=excluded.wording_pattern, "
            "question_length=excluded.question_length, answer_length=excluded.answer_length, "
            "rated_at=datetime('now')",
            (
                request.question_id,
                request.rating,
                features["question_type"],
                features["difficulty"],
                features["wording_pattern"],
                features["question_length"],
                features["answer_length"],
            ),
        )
    return {
        **get_question_preferences(),
        "question_id": request.question_id,
        "rating": request.rating,
    }


@app.delete("/api/questions/feedback")
def clear_question_feedback() -> dict:
    with connect() as db:
        db.execute("DELETE FROM question_feedback")
    return get_question_preferences()


@app.post("/api/questions/answers")
def reveal_question_answers(request: SaveQuestionsRequest) -> dict:
    batch = _question_batches.get(request.generation_id)
    if not batch:
        raise HTTPException(status_code=410, detail="Generate the questions again to reveal their answer key.")
    questions = {item["id"]: item for item in batch["result"]["questions"]}
    if not set(request.question_ids).issubset(questions):
        raise HTTPException(status_code=422, detail="The selected generated questions are invalid.")
    return {
        "answers": {
            question_id: questions[question_id]["correct_answer"]
            for question_id in request.question_ids
        }
    }


@app.post("/api/questions/save")
def save_practice_questions(request: SaveQuestionsRequest) -> dict:
    batch = _question_batches.get(request.generation_id)
    if not batch:
        raise HTTPException(
            status_code=410,
            detail="This generated set is no longer available. Generate the questions again before saving.",
        )
    requested_ids = set(request.question_ids)
    questions = {item["id"]: item for item in batch["result"]["questions"]}
    if len(request.question_ids) != len(requested_ids) or not requested_ids.issubset(questions):
        raise HTTPException(status_code=422, detail="The selected generated questions are invalid.")

    try:
        with connect() as db:
            if batch["dataset_id"] != "builtin" and not db.execute(
                "SELECT 1 FROM training_datasets WHERE id=?", (batch["dataset_id"],)
            ).fetchone():
                raise HTTPException(status_code=409, detail="The source training dataset was deleted.")
            placeholders = ",".join("?" for _ in batch["upload_ids"])
            existing_uploads = {
                row["id"]
                for row in db.execute(
                    f"SELECT id FROM uploads WHERE id IN ({placeholders})",
                    tuple(batch["upload_ids"]),
                ).fetchall()
            }
            if existing_uploads != set(batch["upload_ids"]):
                raise HTTPException(status_code=409, detail="A source course material was deleted.")
            selected = [questions[item] for item in request.question_ids]
            placeholders = ",".join("?" for _ in selected)
            duplicates = db.execute(
                f"SELECT 1 FROM generated_questions WHERE question_text IN "
                f"({placeholders}) LIMIT 1",
                tuple(item["question_text"] for item in selected),
            ).fetchone()
            if duplicates:
                raise HTTPException(status_code=409, detail="One or more questions were already saved.")
            db.execute(
                "INSERT OR IGNORE INTO saved_sessions("
                "id, title, created_at, topic, source_file, difficulty, question_type, question_count"
                ") VALUES(?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    request.generation_id,
                    f"{batch['topic']} practice questions",
                    batch["created_at"],
                    batch["topic"],
                    batch["source_file"],
                    batch["difficulty"],
                    batch["question_type"],
                ),
            )
            for item in selected:
                if (
                    item["support_excerpt"].casefold() not in item["source_reference"].casefold()
                    or item["support_excerpt"].casefold() not in item["source_content"].casefold()
                ):
                    raise HTTPException(status_code=422, detail="A generated question failed source validation.")
                db.execute(
                    "INSERT INTO generated_questions("
                    "id, question_text, question_type, topic, difficulty, choices, "
                    "correct_answer, priority_score, confidence_score, source_dataset_id, "
                    "source_dataset_name, source_material_id, source_material_name, source_reference, "
                    "session_id"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        item["id"], item["question_text"], item["question_type"],
                        item["topic"], item["difficulty"], json.dumps(item["choices"]),
                        item["correct_answer"], item["priority_score"], item["confidence_score"],
                        item["source_dataset_id"], item["source_dataset_name"],
                        item["source_material_id"], item["source_material_name"],
                        item["source_reference"],
                        request.generation_id,
                    ),
                )
            db.execute(
                "UPDATE saved_sessions SET question_count=("
                "SELECT COUNT(*) FROM generated_questions WHERE session_id=?"
                ") WHERE id=?",
                (request.generation_id, request.generation_id),
            )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail="Could not save generated questions locally.") from error
    return {"saved_count": len(requested_ids), "question_ids": request.question_ids}


@app.get("/api/questions/saved")
def list_saved_questions(
    include_answers: bool = False,
    session_id: str | None = None,
) -> dict:
    with connect() as db:
        sessions = db.execute(
            "SELECT id, title, created_at, topic, source_file, difficulty, "
            "question_type, question_count FROM saved_sessions "
            "ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
        rows = db.execute(
            "SELECT q.id, q.question_text, q.question_type, q.topic, q.difficulty, q.choices, "
            "correct_answer, priority_score, confidence_score, source_dataset_name, "
            "source_material_name, source_reference, created_at, session_id, "
            "f.rating AS feedback_rating "
            "FROM generated_questions q LEFT JOIN question_feedback f ON f.question_id=q.id "
            "ORDER BY q.created_at, q.rowid"
        ).fetchall()
    questions_by_session: dict[str, list[dict]] = {}
    for row in rows:
        item = dict(row)
        item["choices"] = json.loads(item["choices"])
        if not include_answers or (session_id and item["session_id"] != session_id):
            item.pop("correct_answer")
        questions_by_session.setdefault(item.pop("session_id"), []).append(item)
    return {
        "sessions": [
            {
                **dict(session),
                "questions": questions_by_session.get(session["id"], []),
            }
            for session in sessions
        ]
    }


@app.put("/api/questions/sessions/{session_id}")
def rename_saved_session(session_id: str, request: RenameSavedSessionRequest) -> dict:
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Session title cannot be blank.")
    with connect() as db:
        cursor = db.execute(
            "UPDATE saved_sessions SET title=? WHERE id=?",
            (title, session_id),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Saved question session not found.")
    return {"id": session_id, "title": title}


@app.delete("/api/questions/sessions/{session_id}")
def delete_saved_session(session_id: str) -> dict:
    with connect() as db:
        cursor = db.execute("DELETE FROM saved_sessions WHERE id=?", (session_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Saved question session not found.")
        db.execute("DELETE FROM generated_questions WHERE session_id=?", (session_id,))
    return {"deleted": session_id}


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="frontend")
