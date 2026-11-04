from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np
from nltk.stem import PorterStemmer
from scipy.optimize import minimize_scalar
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

_stemmer = PorterStemmer()
_stop_words = set(ENGLISH_STOP_WORDS)
PREDICTION_METHOD = "TF-IDF + SVM + Naive Bayes ensemble"

_SAMPLE_ROWS = [
    ("normalization functional dependency candidate key relation 1nf 2nf 3nf database", "Database Normalization"),
    ("split tables remove repeating groups partial dependencies primary key 2nf", "Database Normalization"),
    ("third normal form transitive dependency atomic values relational schema", "Database Normalization"),
    ("normalize relation lossless decomposition dependency preservation database design", "Database Normalization"),
    ("sql select where join group by aggregate query relational database", "SQL Queries"),
    ("inner join foreign key tables select records matching sql", "SQL Queries"),
    ("create table insert update delete transaction sql query", "SQL Queries"),
    ("having clause count sum group by order by database query", "SQL Queries"),
    ("acid transaction atomicity consistency isolation durability database", "Transactions and Concurrency"),
    ("serializable schedule concurrent transaction lock database isolation", "Transactions and Concurrency"),
    ("deadlock wait die timestamp ordering concurrency control transaction", "Transactions and Concurrency"),
    ("commit rollback dirty read phantom read isolation level database", "Transactions and Concurrency"),
    ("process scheduling cpu context switch operating system round robin", "Process Management"),
    ("thread process synchronization mutex semaphore operating system", "Process Management"),
    ("deadlock resource allocation operating system processes", "Process Management"),
    ("process state ready running blocked scheduler operating system", "Process Management"),
    ("virtual memory page table address translation page fault operating system", "Memory Management"),
    ("paging segmentation page replacement least recently used memory", "Memory Management"),
    ("thrashing working set demand paging virtual memory operating system", "Memory Management"),
    ("memory allocation fragmentation contiguous paging operating system", "Memory Management"),
    ("tcp handshake sequence acknowledgement reliable transport network", "Network Protocols"),
    ("ip routing packet forwarding subnet network layer protocol", "Network Protocols"),
    ("dns resolves domain names http request response protocol", "Network Protocols"),
    ("udp tcp ports congestion flow control network transport", "Network Protocols"),
    ("public key private key encryption digital signature cryptography", "Network Security"),
    ("firewall intrusion detection authentication confidentiality security", "Network Security"),
    ("hash function salt password authentication secure network", "Network Security"),
    ("tls certificate certificate authority secure communication encryption", "Network Security"),
    ("html semantic elements css layout responsive web development", "Web Development"),
    ("javascript dom events asynchronous fetch api web development", "Web Development"),
    ("http client server rest api route json web application", "Web Development"),
    ("flexbox grid accessibility responsive design browser web", "Web Development"),
]
BUILTIN_TRAINING_EXAMPLE_COUNT = len(_SAMPLE_ROWS)
BUILTIN_TRAINING_EXAMPLES = tuple(_SAMPLE_ROWS)
BUILTIN_TRAINING_TOPICS = sorted({label for _, label in _SAMPLE_ROWS})
LEGACY_DECISION_POLICY = {
    "svm_temperature": 1.0,
    "ensemble_temperature": 1.0,
    "confidence_threshold": 0.45,
    "min_margin": 0.0,
    "disagreement_abstain": False,
    "target_precision": 0.80,
    "low_reliability": False,
}


def clean_text(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower())
    return " ".join(
        _stemmer.stem(token)
        for token in tokens
        if token not in _stop_words and len(token) > 1
    )


def dataset_hash(examples: Sequence[tuple[str, str]]) -> str:
    rows = sorted((str(topic), str(text)) for text, topic in examples)
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_decision_policy(dataset: Mapping | str) -> dict:
    if isinstance(dataset, str):
        dataset_id = dataset
        examples = None
    else:
        dataset_id = str(dataset.get("id") or dataset.get("dataset_id") or "")
        examples = dataset.get("examples")
    if not dataset_id:
        return dict(LEGACY_DECISION_POLICY)
    if examples is None:
        if dataset_id == "builtin":
            examples = BUILTIN_TRAINING_EXAMPLES
        else:
            from .database import connect

            with connect() as db:
                rows = db.execute(
                    "SELECT text, topic FROM training_examples "
                    "WHERE dataset_id=? ORDER BY rowid",
                    (dataset_id,),
                ).fetchall()
            examples = [(row["text"], row["topic"]) for row in rows]

    from .database import connect

    try:
        with connect() as db:
            row = db.execute(
                "SELECT dataset_hash, svm_temperature, ensemble_temperature, "
                "confidence_threshold, min_margin, disagreement_abstain, "
                "target_precision, low_reliability, sklearn_version "
                "FROM model_calibration WHERE dataset_id=?",
                (dataset_id,),
            ).fetchone()
    except sqlite3.OperationalError as error:
        if "no such table: model_calibration" in str(error).lower():
            return dict(LEGACY_DECISION_POLICY)
        raise
    if (
        row is None
        or row["dataset_hash"] != dataset_hash(examples)
        or row["sklearn_version"] != sklearn_version
    ):
        return dict(LEGACY_DECISION_POLICY)
    return {
        "svm_temperature": float(row["svm_temperature"]),
        "ensemble_temperature": float(row["ensemble_temperature"]),
        "confidence_threshold": float(row["confidence_threshold"]),
        "min_margin": float(row["min_margin"]),
        "disagreement_abstain": bool(row["disagreement_abstain"]),
        "target_precision": float(row["target_precision"]),
        "low_reliability": bool(row["low_reliability"]),
    }


@lru_cache(maxsize=2)
def _fit_models(examples: tuple[tuple[str, str], ...]):
    return _fit_uncached_models(examples)


def _get_models(examples: tuple[tuple[str, str], ...]):
    if len(examples) > 2000:
        return _fit_uncached_models(examples)
    return _fit_models(examples)


def _fit_uncached_models(examples: tuple[tuple[str, str], ...]):
    processed = [
        (clean_text(text), label.strip())
        for text, label in examples
        if text.strip() and label.strip()
    ]
    processed = [(text, label) for text, label in processed if text]
    if len(processed) < 2 or len({label for _, label in processed}) < 2:
        raise ValueError("Training data needs readable examples from at least two topics.")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=12000)
    features = vectorizer.fit_transform([text for text, _ in processed])
    labels = [label for _, label in processed]
    svm = SVC(
        kernel="linear", class_weight="balanced", decision_function_shape="ovr"
    )
    naive_bayes = MultinomialNB(alpha=0.5)
    svm.fit(features, labels)
    naive_bayes.fit(features, labels)
    return vectorizer, svm, naive_bayes


def _validate_temperature(temperature: float) -> float:
    value = float(temperature)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Temperature must be a finite positive number.")
    return value


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted_scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(shifted_scores)
    return weights / weights.sum(axis=1, keepdims=True)


def apply_ensemble_temperature(
    p_ens: np.ndarray,
    T_ens: float = 1.0,
) -> np.ndarray:
    temperature = _validate_temperature(T_ens)
    probabilities = np.asarray(p_ens, dtype=float)
    if probabilities.ndim != 2 or not probabilities.size:
        raise ValueError("Ensemble probabilities must be a non-empty 2D array.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("Ensemble probabilities must be finite and non-negative.")
    if np.any(probabilities.sum(axis=1) <= 0):
        raise ValueError("Each probability row must have a positive sum.")
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
    return _softmax(log_probabilities / temperature)


def fit_temperature(
    logits_or_log_probs: np.ndarray,
    true_label_indices: Sequence[int],
) -> float:
    scores = np.asarray(logits_or_log_probs, dtype=float)
    labels = np.asarray(true_label_indices)
    if scores.ndim != 2 or not scores.size or labels.size == 0:
        raise ValueError("Scores and true labels must be non-empty.")
    if labels.ndim != 1 or scores.shape[0] != labels.shape[0]:
        raise ValueError("Scores and true labels must have matching sample counts.")
    if scores.shape[1] < 2:
        raise ValueError("Scores must contain at least two classes.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("True label indices must be integers.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores must contain only finite values.")
    if np.any(labels < 0) or np.any(labels >= scores.shape[1]):
        raise ValueError("True label indices are outside the score columns.")

    def mean_log_loss(log_temperature: float) -> float:
        scaled_scores = scores / np.exp(log_temperature)
        scaled_scores -= scaled_scores.max(axis=1, keepdims=True)
        log_probabilities = scaled_scores - np.log(
            np.exp(scaled_scores).sum(axis=1, keepdims=True)
        )
        return float(-log_probabilities[np.arange(len(labels)), labels].mean())

    result = minimize_scalar(
        mean_log_loss,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"Temperature fitting failed: {result.message}")
    return float(np.exp(result.x))


def _score_components(
    texts_or_features,
    svm,
    naive_bayes,
    vectorizer=None,
    svm_temperature: float = 1.0,
    ensemble_temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    svm_temperature = _validate_temperature(svm_temperature)
    ensemble_temperature = _validate_temperature(ensemble_temperature)
    features = (
        vectorizer.transform(texts_or_features)
        if vectorizer is not None
        else texts_or_features
    )
    svm_scores = np.asarray(svm.decision_function(features))
    if len(svm.classes_) == 2:
        svm_scores = np.column_stack((-svm_scores, svm_scores))
    svm_probabilities = _softmax(svm_scores / svm_temperature)
    bayes_probabilities = naive_bayes.predict_proba(features)
    ensemble_probabilities = (svm_probabilities + bayes_probabilities) / 2
    if ensemble_temperature != 1.0:
        ensemble_probabilities = apply_ensemble_temperature(
            ensemble_probabilities,
            ensemble_temperature,
        )
    return svm.classes_, svm_probabilities, bayes_probabilities, ensemble_probabilities


def _ensemble_probabilities(features, svm, naive_bayes) -> np.ndarray:
    return _score_components(features, svm, naive_bayes)[3]


def _align_class_columns(
    scores: np.ndarray,
    source_classes: Sequence[str],
    target_classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]
    aligned = np.zeros((scores.shape[0], len(target_classes)), dtype=float)
    class_mask = np.zeros(len(target_classes), dtype=bool)
    target_indices = {label: index for index, label in enumerate(target_classes)}
    for source_index, label in enumerate(source_classes):
        target_index = target_indices.get(label)
        if target_index is not None:
            aligned[:, target_index] = scores[:, source_index]
            class_mask[target_index] = True
    return aligned, class_mask


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _probabilities_from_margins(
    margins: np.ndarray,
    class_mask: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    probabilities = _softmax(margins / _validate_temperature(temperature))
    probabilities[:, ~class_mask] = 0.0
    return _normalize_probabilities(probabilities)


def _cross_fitting_indices(fold_ids: np.ndarray):
    for fold_id in np.unique(fold_ids):
        scored_indices = np.flatnonzero(fold_ids == fold_id)
        fit_indices = np.flatnonzero(fold_ids != fold_id)
        assert not np.intersect1d(fit_indices, scored_indices).size
        yield fit_indices, scored_indices


def _fit_ensemble_temperature(
    ensemble_probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    return fit_temperature(
        np.log(np.clip(ensemble_probabilities, 1e-12, 1.0)),
        labels,
    )


def _comparison_metrics(
    probabilities: np.ndarray,
    true_indices: np.ndarray,
) -> dict:
    predicted_indices = np.argmax(probabilities, axis=1)
    true_one_hot = np.eye(probabilities.shape[1])[true_indices]
    clipped_probabilities = np.clip(probabilities, 1e-12, 1.0)
    log_loss = -np.log(
        clipped_probabilities[np.arange(len(true_indices)), true_indices]
    ).mean()
    _, _, macro_f1, _ = precision_recall_fscore_support(
        true_indices,
        predicted_indices,
        labels=np.arange(probabilities.shape[1]),
        average="macro",
        zero_division=0,
    )
    confidence = probabilities.max(axis=1)
    correct = (predicted_indices == true_indices).astype(float)
    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        in_bin = (confidence >= lower) & (
            confidence <= upper if bin_index == 9 else confidence < upper
        )
        if np.any(in_bin):
            ece += float(np.mean(in_bin)) * abs(
                float(confidence[in_bin].mean())
                - float(correct[in_bin].mean())
            )
    return {
        "accuracy": float(np.mean(predicted_indices == true_indices)),
        "macro_f1": float(macro_f1),
        "log_loss": float(log_loss),
        "brier_score": float(np.mean(np.sum((probabilities - true_one_hot) ** 2, axis=1))),
        "expected_calibration_error": float(ece),
    }


def _reliability_bins(
    probabilities: np.ndarray,
    true_indices: np.ndarray,
) -> list[dict]:
    confidence = probabilities.max(axis=1)
    correct = (np.argmax(probabilities, axis=1) == true_indices).astype(float)
    bins = []
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        in_bin = (confidence >= lower) & (
            confidence <= upper if bin_index == 9 else confidence < upper
        )
        count = int(in_bin.sum())
        bins.append({
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_confidence": (
                float(confidence[in_bin].mean()) if count else None
            ),
            "accuracy": float(correct[in_bin].mean()) if count else None,
        })
    return bins


def _select_decision_policy(
    probabilities: np.ndarray,
    true_indices: np.ndarray,
    *,
    target_precision: float = 0.80,
    force_low_reliability: bool = False,
) -> tuple[list[dict], dict]:
    confidence = probabilities.max(axis=1)
    predicted = np.argmax(probabilities, axis=1)
    margins = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
    correct = predicted == true_indices
    sweep = []
    for threshold_index in range(66):
        threshold = round(0.30 + threshold_index * 0.01, 2)
        accepted = confidence >= threshold
        count = int(accepted.sum())
        sweep.append({
            "threshold": threshold,
            "coverage": float(count / len(true_indices)),
            "precision": (
                float(correct[accepted].mean()) if count else None
            ),
        })
    qualifying = [
        item for item in sweep
        if item["coverage"] >= 0.20
        and item["precision"] is not None
        and item["precision"] >= target_precision
    ]
    low_reliability = force_low_reliability or not qualifying
    confidence_threshold = (
        qualifying[0]["threshold"]
        if qualifying and not force_low_reliability
        else LEGACY_DECISION_POLICY["confidence_threshold"]
    )
    min_margin = 0.0
    if not low_reliability:
        candidates = []
        for margin_threshold in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
            accepted = (
                (confidence >= confidence_threshold)
                & (margins >= margin_threshold)
            )
            count = int(accepted.sum())
            precision = float(correct[accepted].mean()) if count else None
            coverage = count / len(true_indices)
            if (
                coverage >= 0.20
                and precision is not None
                and precision >= target_precision
            ):
                candidates.append((coverage, -margin_threshold, margin_threshold))
        if candidates:
            min_margin = max(candidates)[2]
    policy = {
        "svm_temperature": None,
        "ensemble_temperature": None,
        "confidence_threshold": confidence_threshold,
        "min_margin": min_margin,
        "disagreement_abstain": True,
        "target_precision": float(target_precision),
        "low_reliability": bool(low_reliability),
    }
    return sweep, policy


def evaluate_classifier(
    examples: Sequence[tuple[str, str]],
    *,
    max_examples: int = 10000,
) -> dict:
    usable_examples = []
    for text, topic in examples:
        cleaned_text = clean_text(text)
        if text.strip() and topic.strip() and cleaned_text:
            usable_examples.append((text.strip(), topic.strip(), cleaned_text))
    excluded_examples = len(examples) - len(usable_examples)
    if len(usable_examples) > max_examples:
        raise ValueError(
            f"Evaluation is limited to {max_examples:,} readable labeled examples."
        )
    labels = [topic for _, topic, _ in usable_examples]
    label_counts = Counter(labels)
    if len(label_counts) < 2:
        raise ValueError("Evaluation requires at least two topic labels.")
    folds = min(5, min(label_counts.values()))
    if folds < 2:
        raise ValueError(
            "Each topic needs at least two readable examples for stratified evaluation."
        )

    label_names = sorted(label_counts, key=str.casefold)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    predicted_labels = [""] * len(labels)
    true_indices = np.asarray([label_names.index(label) for label in labels])
    svm_margins = np.zeros((len(labels), len(label_names)), dtype=float)
    nb_scores = np.zeros_like(svm_margins)
    svm_class_masks = np.zeros((len(labels), len(label_names)), dtype=bool)
    fold_ids = np.full(len(labels), -1, dtype=int)
    cleaned_texts = [text for _, _, text in usable_examples]
    for fold_id, (train_indices, test_indices) in enumerate(
        splitter.split(cleaned_texts, labels)
    ):
        fold_examples = tuple(
            (usable_examples[index][0], labels[index]) for index in train_indices
        )
        vectorizer, svm, naive_bayes = _fit_uncached_models(fold_examples)
        test_features = vectorizer.transform(
            [cleaned_texts[index] for index in test_indices]
        )
        probabilities = _ensemble_probabilities(test_features, svm, naive_bayes)
        fold_predictions = [
            str(svm.classes_[index])
            for index in np.argmax(probabilities, axis=1)
        ]
        for index, prediction in zip(test_indices, fold_predictions):
            predicted_labels[index] = prediction
        fold_margins = np.asarray(svm.decision_function(test_features))
        if len(svm.classes_) == 2:
            fold_margins = np.column_stack((-fold_margins, fold_margins))
        aligned_margins, class_mask = _align_class_columns(
            fold_margins,
            svm.classes_,
            label_names,
        )
        aligned_nb_scores, _ = _align_class_columns(
            naive_bayes.predict_proba(test_features),
            naive_bayes.classes_,
            label_names,
        )
        svm_margins[test_indices] = aligned_margins
        nb_scores[test_indices] = _normalize_probabilities(aligned_nb_scores)
        svm_class_masks[test_indices] = class_mask
        fold_ids[test_indices] = fold_id

    svm_temperature = fit_temperature(svm_margins, true_indices)
    cross_fitted_temperatures = []
    svm_probabilities = np.vstack([
        _probabilities_from_margins(
            svm_margins[index:index + 1],
            svm_class_masks[index],
        )
        for index in range(len(labels))
    ])
    ensemble_probabilities = _normalize_probabilities(
        (svm_probabilities + nb_scores) / 2
    )
    ensemble_temperature = _fit_ensemble_temperature(
        ensemble_probabilities,
        true_indices,
    )
    if folds >= 3:
        for fit_indices, _ in _cross_fitting_indices(fold_ids):
            cross_fitted_temperatures.append(
                _fit_ensemble_temperature(
                    ensemble_probabilities[fit_indices],
                    true_indices[fit_indices],
                )
            )
    nb_predictions = np.argmax(nb_scores, axis=1)
    svm_predictions = np.argmax(svm_probabilities, axis=1)
    ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
    calibrated_probabilities = None
    if folds >= 3:
        calibrated_probabilities = np.zeros_like(ensemble_probabilities)
        for (_, scored_indices), fold_ensemble_temperature in zip(
            _cross_fitting_indices(fold_ids),
            cross_fitted_temperatures,
        ):
            calibrated_probabilities[scored_indices] = apply_ensemble_temperature(
                ensemble_probabilities[scored_indices],
                fold_ensemble_temperature,
            )
    policy_probabilities = (
        calibrated_probabilities
        if calibrated_probabilities is not None
        else ensemble_probabilities
    )
    threshold_sweep, decision_policy = _select_decision_policy(
        policy_probabilities,
        true_indices,
        force_low_reliability=calibrated_probabilities is None,
    )
    decision_policy["svm_temperature"] = svm_temperature
    decision_policy["ensemble_temperature"] = ensemble_temperature

    warnings = [
        f"Topic '{label}' has fewer than 20 readable examples."
        for label, count in label_counts.items()
        if count < 20
    ]
    if folds < 3:
        warnings.append(
            "Fewer than 3 folds are available; calibrated comparison metrics were skipped."
        )
    comparison = {
        "svm_only": _comparison_metrics(
            svm_probabilities,
            true_indices,
        ),
        "nb_only": _comparison_metrics(nb_scores, true_indices),
        "ensemble": _comparison_metrics(ensemble_probabilities, true_indices),
        "ensemble_calibrated": (
            _comparison_metrics(calibrated_probabilities, true_indices)
            if calibrated_probabilities is not None
            else None
        ),
    }
    ensemble_correct = ensemble_predictions == true_indices
    agreement = svm_predictions == nb_predictions
    agree_count = int(agreement.sum())
    disagree_count = int((~agreement).sum())

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted_labels,
        labels=label_names,
        average="macro",
        zero_division=0,
    )
    return {
        "method": PREDICTION_METHOD,
        "validation": "Stratified cross-validation",
        "folds": folds,
        "examples_evaluated": len(labels),
        "examples_excluded": excluded_examples,
        "topics": label_names,
        "metrics": {
            "accuracy": float(accuracy_score(labels, predicted_labels)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        },
        "confusion_matrix": {
            "labels": label_names,
            "values": confusion_matrix(
                labels, predicted_labels, labels=label_names
            ).tolist(),
        },
        "disclaimer": (
            "Cross-validation estimates performance on this labeled dataset. "
            "It does not measure accuracy on future exams or guarantee exam topics."
        ),
        "comparison": comparison,
        "calibration": {
            "svm_temperature": svm_temperature,
            "ensemble_temperature": ensemble_temperature,
            "reliability_bins": {
                "ensemble": _reliability_bins(
                    ensemble_probabilities,
                    true_indices,
                ),
                "ensemble_calibrated": (
                    _reliability_bins(calibrated_probabilities, true_indices)
                    if calibrated_probabilities is not None
                    else None
                ),
            },
        },
        "component_scores": {
            "svm_only": {
                "mean": float(svm_probabilities.max(axis=1).mean()),
                "median": float(np.median(svm_probabilities.max(axis=1))),
            },
            "nb_only": {
                "mean": float(nb_scores.max(axis=1).mean()),
                "median": float(np.median(nb_scores.max(axis=1))),
            },
        },
        "disagreement": {
            "agreement_rate": float(agreement.mean()),
            "accuracy_when_agree": (
                float(ensemble_correct[agreement].mean()) if agree_count else None
            ),
            "accuracy_when_disagree": (
                float(ensemble_correct[~agreement].mean()) if disagree_count else None
            ),
            "agreement_count": agree_count,
            "disagreement_count": disagree_count,
        },
        "warnings": warnings,
        "threshold_sweep": threshold_sweep,
        "decision_policy": decision_policy,
        "low_reliability": decision_policy["low_reliability"],
    }


def predict_distribution(
    text: str,
    training_examples: Sequence[tuple[str, str]] | None = None,
    progress_callback=None,
) -> list[dict]:
    return predict_distributions(
        [text],
        training_examples=training_examples,
        progress_callback=progress_callback,
    )[0]


def predict_distributions(
    texts: Sequence[str],
    training_examples: Sequence[tuple[str, str]] | None = None,
    progress_callback=None,
    training_metrics: dict | None = None,
) -> list[list[dict]]:
    def report(phase: str, label: str) -> None:
        if progress_callback:
            progress_callback({"phase": phase, "label": label})

    queries = [clean_text(text) for text in texts]
    valid_indices = [index for index, query in enumerate(queries) if query]
    if not valid_indices:
        return [[] for _ in queries]
    examples = tuple(
        (sample_text, topic)
        for sample_text, topic in (
            _SAMPLE_ROWS if training_examples is None else training_examples
        )
    )
    report("cleaning", "Text cleaning complete")
    report("features", "Feature extraction complete")
    fit_started = time.perf_counter()
    cache_hits_before = _fit_models.cache_info().hits
    try:
        vectorizer, svm, naive_bayes = _get_models(examples)
    except ValueError as error:
        if "empty vocabulary" in str(error).lower():
            raise ValueError("Training data has no readable words after text cleaning.") from error
        raise
    fit_elapsed = time.perf_counter() - fit_started
    cache_hit = (
        len(examples) <= 2000
        and _fit_models.cache_info().hits > cache_hits_before
    )
    if training_metrics is not None:
        training_metrics["fit_seconds"] = (
            training_metrics.get("fit_seconds", 0.0)
            + (0.0 if cache_hit else fit_elapsed)
        )
        training_metrics["fit_count"] = (
            training_metrics.get("fit_count", 0) + (0 if cache_hit else 1)
        )
        training_metrics["cache_hits"] = (
            training_metrics.get("cache_hits", 0) + int(cache_hit)
        )

    class_labels, _, _, probabilities = _score_components(
        [queries[index] for index in valid_indices],
        svm,
        naive_bayes,
        vectorizer,
    )
    report("analysis", "SVM and Naive Bayes analysis complete")
    predictions = [
        [
            {"topic": label, "confidence": float(score)}
            for label, score in sorted(
                zip(class_labels, row),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]
        for row in probabilities
    ]
    results = [[] for _ in queries]
    for index, prediction in zip(valid_indices, predictions):
        results[index] = prediction
    return results


def classify_with_policy(
    text: str,
    training_examples: Sequence[tuple[str, str]] | None = None,
    dataset_id: str = "builtin",
    decision_policy: Mapping | None = None,
    precomputed_distribution: Sequence[Mapping] | None = None,
) -> dict:
    examples = tuple(
        (sample_text, topic)
        for sample_text, topic in (
            BUILTIN_TRAINING_EXAMPLES
            if training_examples is None
            else training_examples
        )
    )
    policy = dict(decision_policy) if decision_policy is not None else get_decision_policy(
        {"id": dataset_id, "examples": examples}
    )
    if precomputed_distribution is not None and policy == LEGACY_DECISION_POLICY:
        if not precomputed_distribution:
            return {
                "topic": "Unclassified",
                "confidence": 0.0,
                "margin": 0.0,
                "models_agree": True,
                "abstain_reason": "low_confidence",
            }
        best = precomputed_distribution[0]
        cleaned = clean_text(text)
        if cleaned:
            vectorizer, svm, naive_bayes = _get_models(examples)
            _, svm_scores, nb_scores, _ = _score_components(
                [cleaned],
                svm,
                naive_bayes,
                vectorizer,
            )
            models_agree = bool(
                np.argmax(svm_scores[0]) == np.argmax(nb_scores[0])
            )
        else:
            models_agree = False
        confidence = float(best["confidence"])
        margin = confidence - float(
            precomputed_distribution[1]["confidence"]
            if len(precomputed_distribution) > 1
            else 0.0
        )
        reason = (
            "low_confidence"
            if confidence < float(policy["confidence_threshold"])
            else None
        )
        return {
            "topic": "Unclassified" if reason else str(best["topic"]),
            "confidence": confidence,
            "margin": margin,
            "models_agree": models_agree,
            "abstain_reason": reason,
        }
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "topic": "Unclassified",
            "confidence": 0.0,
            "margin": 0.0,
            "models_agree": False,
            "abstain_reason": "low_confidence",
        }
    vectorizer, svm, naive_bayes = _get_models(examples)
    class_labels, svm_scores, nb_scores, ensemble_scores = _score_components(
        [cleaned],
        svm,
        naive_bayes,
        vectorizer,
        ensemble_temperature=float(policy["ensemble_temperature"]),
    )
    ensemble_row = ensemble_scores[0]
    top_two = np.argsort(ensemble_row)[-2:][::-1]
    top_index, second_index = int(top_two[0]), int(top_two[1])
    confidence = float(ensemble_row[top_index])
    margin = confidence - float(ensemble_row[second_index])
    models_agree = bool(
        np.argmax(svm_scores[0]) == np.argmax(nb_scores[0])
    )
    abstain_reason = None
    if confidence < float(policy["confidence_threshold"]):
        abstain_reason = "low_confidence"
    elif margin < float(policy["min_margin"]):
        abstain_reason = "small_margin"
    elif policy["disagreement_abstain"] and not models_agree:
        abstain_reason = "model_disagreement"
    return {
        "topic": (
            "Unclassified"
            if abstain_reason is not None
            else str(class_labels[top_index])
        ),
        "confidence": confidence,
        "margin": margin,
        "models_agree": models_agree,
        "abstain_reason": abstain_reason,
    }


def augment_training_examples_with_exam_passages(
    training_examples: Sequence[tuple[str, str]],
    exam_sources: Sequence[dict],
    training_metrics: dict | None = None,
    dataset_id: str = "builtin",
    decision_policy: Mapping | None = None,
) -> tuple[list[tuple[str, str]], int]:
    passages = []
    for source in exam_sources:
        for passage in re.split(r"(?<=[.!?])\s+|\r?\n+", str(source.get("text", ""))):
            passage = re.sub(r"^\s*(?:\d+[.)]|[A-D][.)])\s*", "", passage).strip()
            if 24 <= len(passage) <= 500 and clean_text(passage):
                passages.append(passage)
            if len(passages) >= 2000:
                break
        if len(passages) >= 2000:
            break
    if not passages:
        return list(training_examples), 0

    distributions = predict_distributions(
        passages,
        training_examples,
        training_metrics=training_metrics,
    )
    policy = dict(decision_policy) if decision_policy is not None else get_decision_policy(
        {"id": dataset_id, "examples": training_examples}
    )
    augmented = list(training_examples)
    seen = {(text.casefold(), topic.casefold()) for text, topic in augmented}
    added = 0
    for passage, distribution in zip(passages, distributions):
        classified = classify_with_policy(
            passage,
            training_examples,
            dataset_id,
            policy,
            distribution,
        )
        key = (passage.casefold(), classified["topic"].casefold())
        if classified["topic"] != "Unclassified" and key not in seen:
            augmented.append((passage, classified["topic"]))
            seen.add(key)
            added += 1
    return augmented, added


def predict(
    text: str,
    training_examples: Sequence[tuple[str, str]] | None = None,
    progress_callback=None,
) -> list[dict]:
    if progress_callback:
        progress_callback({"phase": "extraction", "label": "Text extraction complete"})
        progress_callback({"phase": "analysis", "label": "SVM and Naive Bayes analysis in progress"})
    ranked = predict_distribution(
        text,
        training_examples=training_examples,
        progress_callback=progress_callback,
    )
    return [
        {"topic": item["topic"], "confidence": round(item["confidence"], 4)}
        for item in ranked[:5]
    ]
