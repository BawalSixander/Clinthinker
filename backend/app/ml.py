from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from nltk.stem import PorterStemmer
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
MINIMUM_TRAINING_CONFIDENCE = 0.45


def clean_text(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower())
    return " ".join(
        _stemmer.stem(token)
        for token in tokens
        if token not in _stop_words and len(token) > 1
    )


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


def _ensemble_probabilities(features, svm, naive_bayes) -> np.ndarray:
    svm_scores = np.asarray(svm.decision_function(features))
    if len(svm.classes_) == 2:
        svm_scores = np.column_stack((-svm_scores, svm_scores))
    svm_scores -= svm_scores.max(axis=1, keepdims=True)
    svm_weights = np.exp(svm_scores)
    svm_probabilities = svm_weights / svm_weights.sum(axis=1, keepdims=True)
    bayes_probabilities = naive_bayes.predict_proba(features)
    return (svm_probabilities + bayes_probabilities) / 2


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
    cleaned_texts = [text for _, _, text in usable_examples]
    for train_indices, test_indices in splitter.split(cleaned_texts, labels):
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

    query_features = vectorizer.transform([queries[index] for index in valid_indices])
    probabilities = _ensemble_probabilities(query_features, svm, naive_bayes)
    report("analysis", "SVM and Naive Bayes analysis complete")
    predictions = [
        [
            {"topic": label, "confidence": float(score)}
            for label, score in sorted(
                zip(svm.classes_, row),
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


def augment_training_examples_with_exam_passages(
    training_examples: Sequence[tuple[str, str]],
    exam_sources: Sequence[dict],
    training_metrics: dict | None = None,
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
    augmented = list(training_examples)
    seen = {(text.casefold(), topic.casefold()) for text, topic in augmented}
    added = 0
    for passage, distribution in zip(passages, distributions):
        best = distribution[0]
        key = (passage.casefold(), best["topic"].casefold())
        if (
            best["confidence"] >= MINIMUM_TRAINING_CONFIDENCE
            and key not in seen
        ):
            augmented.append((passage, best["topic"]))
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
