from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import uuid
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ml import (
    clean_text,
    classify_with_policy,
    get_decision_policy,
    predict_distributions,
)

logger = logging.getLogger(__name__)

QUESTION_TYPES = {"Multiple Choice", "True or False", "Identification"}
DIFFICULTIES = {"Easy", "Moderate", "Difficult"}
DEFAULT_PRIORITY_WEIGHTS = {
    "historical_frequency": 0.40,
    "course_importance": 0.30,
    "model_confidence": 0.20,
    "recent_occurrence": 0.10,
}
DEFAULT_SIMILARITY_THRESHOLD = 0.82
_DEFINITION = re.compile(
    r"^\s*(?:(?:a|an|the)\s+)?"
    r"([A-Za-z][A-Za-z0-9/-]*(?:\s+[A-Za-z][A-Za-z0-9/-]*){0,3})\s+"
    r"(?:is defined as|is|are|refers to|means|describes)\s+(.{12,300})$",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_INVALID_TERM_STARTS = {
    "analyze", "because", "compare", "define", "describe", "discuss", "explain",
    "how", "identify", "list", "locally", "select", "state", "what", "when",
    "where", "which", "who", "why",
}
_DISTRACTOR_JUNK = ENGLISH_STOP_WORDS | {
    "all values", "code", "concept", "data", "example", "information", "item",
    "object", "process", "result", "stuff", "system", "thing", "type", "value",
    "values", "way", "you",
}
_PRONOUNS = {
    "he", "her", "hers", "herself", "him", "himself", "his", "i", "it", "its",
    "itself", "me", "my", "myself", "our", "ours", "ourselves", "she", "them",
    "themselves", "they", "us", "we", "you", "your", "yours", "yourself",
    "yourselves",
}
_INCOMPLETE_ENDINGS = re.compile(
    r"\b(?:a|an|and|are|as|because|but|by|for|from|in|is|of|or|that|the|to|"
    r"was|were|which|with)$",
    re.IGNORECASE,
)
_COMPLEXITY_CUES = {
    "analyze", "compare", "contrast", "differentiate", "evaluate",
    "implications", "relationship", "whereas",
}
_HEADER_FIELD_LINE = re.compile(
    r"^\s*(?:student\s+)?(?:name|section|course(?:\s+(?:name|code|number))?|"
    r"class|date|instructor|teacher|student\s*(?:id|number)|id)"
    r"\b\s*(?::|#|-|_{2,})",
    re.IGNORECASE,
)
_EXAM_TITLE_LINE = re.compile(
    r"^\s*(?:(?:midterm|final)\s+)?(?:exam|quiz|test|midterm|final)"
    r"\s*(?:(?:no\.?|#)\s*)?(?:\d+|[ivxlcdm]+)?"
    r"(?:\s*[-:|]\s*[\w &-]+)?\s*$",
    re.IGNORECASE,
)
_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:page\s+)?\d+\s*(?:of\s+\d+|/\s*\d+)?\s*$",
    re.IGNORECASE,
)
_EXAM_STEM_TEMPLATE_CACHE: OrderedDict[str, list[dict]] = OrderedDict()


def _cache_exam_stem_templates(key: str, templates: list[dict]) -> None:
    _EXAM_STEM_TEMPLATE_CACHE[key] = templates
    _EXAM_STEM_TEMPLATE_CACHE.move_to_end(key)
    while len(_EXAM_STEM_TEMPLATE_CACHE) > 4:
        _EXAM_STEM_TEMPLATE_CACHE.popitem(last=False)


def validate_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    values = dict(DEFAULT_PRIORITY_WEIGHTS if weights is None else weights)
    if set(values) != set(DEFAULT_PRIORITY_WEIGHTS):
        raise ValueError("Priority weights must include all four supported score components.")
    if any(
        not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
        for value in values.values()
    ):
        raise ValueError("Priority weights must be non-negative numbers.")
    total = sum(values.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError("Priority weights must add up to 1.0.")
    return {key: float(value) for key, value in values.items()}


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip(" \t-•")
        for sentence in _SENTENCE_END.split(text)
        if 24 <= len(sentence.strip(" \t-•")) <= 500
    ]


def strip_document_headers(text: str) -> tuple[str, int]:
    content_lines = []
    removed_lines = 0
    for line in text.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or _HEADER_FIELD_LINE.match(stripped)
            or _EXAM_TITLE_LINE.match(stripped)
            or _PAGE_NUMBER_LINE.match(stripped)
        ):
            if stripped:
                removed_lines += 1
            continue
        content_lines.append(line)
    return "\n".join(content_lines), removed_lines


def _definition(sentence: str) -> tuple[str, str] | None:
    match = _DEFINITION.match(sentence.strip().rstrip("."))
    if not match:
        return None
    term = re.sub(r"\s+", " ", match.group(1)).strip(" -")
    explanation = re.sub(r"\s+", " ", match.group(2)).strip()
    term_words = term.split()
    if (
        len(term) < 3
        or len(term_words) > 4
        or any(len(word) < 3 and not word.isupper() for word in term_words)
        or term_words[0].casefold() in _INVALID_TERM_STARTS
        or len(explanation) < 12
        or _INCOMPLETE_ENDINGS.search(explanation.rstrip(".!?"))
        or term.casefold() == explanation.casefold()
    ):
        return None
    return term, explanation


def _usable_distractor(
    term: str,
    answer: str,
    question: str,
    cited_passage: str,
    other_answers: set[str],
) -> bool:
    normalized = re.sub(r"\s+", " ", term).strip().casefold()
    words = re.findall(r"[a-z0-9]+", normalized)
    if (
        len(normalized) < 3
        or not words
        or normalized in _DISTRACTOR_JUNK
        or len(words) == 1 and words[0] in _DISTRACTOR_JUNK
        or any(word in _PRONOUNS for word in words)
        or (len(answer.split()) > 1 and len(words) < 2)
        or normalized == answer.casefold()
        or normalized in other_answers
        or normalized in question.casefold()
        or re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", cited_passage.casefold())
    ):
        return False
    return True


def _similar_definition_term(answer: str, distractor: str) -> bool:
    answer_words = re.findall(r"[A-Za-z0-9]+", answer)
    distractor_words = re.findall(r"[A-Za-z0-9]+", distractor)
    return (
        len(answer_words) == len(distractor_words)
        and abs(len(answer) - len(distractor)) <= max(4, len(answer) // 2)
    )


def _negate_statement(sentence: str) -> str | None:
    if re.search(r"\b(?:not|never|no)\b|n't\b", sentence, re.IGNORECASE):
        return None
    match = re.match(
        r"^(.+?\b(?:is|are|was|were|can|could|will|would|should|must|may|might|"
        r"do|does|did)\b)(\s+.*)$",
        sentence,
        re.IGNORECASE,
    )
    if not match:
        return None
    prefix = match.group(1)
    auxiliary = prefix.rsplit(None, 1)[-1]
    if auxiliary.casefold() == "can":
        replacement = (
            "cannot"
            if auxiliary.islower()
            else "CANNOT" if auxiliary.isupper() else "Cannot"
        )
        return f"{prefix[:-len(auxiliary)]}{replacement}{match.group(2)}"
    return f"{prefix} not{match.group(2)}"


def estimate_difficulty(sentence: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", sentence)
    cues = {word.casefold() for word in words} & _COMPLEXITY_CUES
    if len(words) <= 12:
        return "Easy"
    if len(words) >= 36 or len(cues) >= 2:
        return "Difficult"
    return "Moderate"


def topic_priority(
    topics: Sequence[str],
    training_examples: Sequence[tuple[str, str]],
    material_texts: Sequence[str],
    topic_confidence: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> list[dict]:
    weights = validate_weights(weights)
    counts = Counter(topic.strip() for _, topic in training_examples if topic.strip())
    max_frequency = max(counts.values(), default=1)
    all_material = "\n".join(material_texts).casefold()
    material_counts: dict[str, int] = {}
    for topic in topics:
        words = [word for word in re.findall(r"[a-z0-9]+", topic.casefold()) if len(word) > 1]
        material_counts[topic] = sum(
            len(re.findall(rf"\b{re.escape(word)}\b", all_material)) for word in words
        )
    max_material = max(material_counts.values(), default=0)

    ranked = []
    for topic in topics:
        frequency_score = counts[topic] / max_frequency
        material_score = (
            material_counts[topic] / max_material if max_material else 0.0
        )
        confidence_score = max(0.0, min(1.0, float(topic_confidence.get(topic, 0.0))))
        recent_score = 0.5
        score = (
            frequency_score * weights["historical_frequency"]
            + material_score * weights["course_importance"]
            + confidence_score * weights["model_confidence"]
            + recent_score * weights["recent_occurrence"]
        )
        ranked.append(
            {
                "topic": topic,
                "priority_score": round(score, 4),
                "frequency_score": round(frequency_score, 4),
                "course_importance_score": round(material_score, 4),
                "model_confidence_score": round(confidence_score, 4),
                "recent_occurrence_score": recent_score,
                "recent_occurrence_available": False,
            }
        )
    return sorted(ranked, key=lambda item: (-item["priority_score"], item["topic"].casefold()))


def is_duplicate(candidate: str, previous: Sequence[str], threshold: float) -> bool:
    if not previous:
        return False
    documents = [candidate, *previous]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(
        documents
    )
    return bool(cosine_similarity(matrix[0:1], matrix[1:]).max() >= threshold)


def _lesson_passages(text: str) -> list[dict]:
    passages = []
    current = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                passages.append("\n".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        passages.append("\n".join(current))
    if not passages:
        passages = [text.strip()]

    result = []
    for passage in passages:
        for sentence in _SENTENCE_END.split(passage):
            list_item = bool(re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", sentence))
            sentence = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", sentence).strip()
            heading = (
                len(sentence) <= 100
                and not sentence.endswith((".", "?", "!"))
            )
            if (len(sentence) < 24 and not heading and not list_item) or not clean_text(sentence):
                continue
            result.append(
                {
                    "text": sentence,
                    "heading": heading,
                    "definition": _definition(sentence) is not None,
                    "list_item": list_item,
                }
            )
    return result


def extract_exam_stem_templates(
    exam_sources: Sequence[Mapping],
    training_examples: Sequence[tuple[str, str]],
    training_metrics: dict | None = None,
    dataset_id: str = "builtin",
    decision_policy: Mapping | None = None,
) -> list[dict]:
    policy = dict(decision_policy) if decision_policy is not None else get_decision_policy(
        {"id": dataset_id, "examples": training_examples}
    )
    cache_digest = hashlib.sha256()
    for source in exam_sources:
        cache_digest.update(str(source.get("name", "")).encode("utf-8"))
        cache_digest.update(b"\0")
        cache_digest.update(str(source.get("text", "")).encode("utf-8"))
        cache_digest.update(b"\0")
    for text, topic in training_examples:
        cache_digest.update(text.encode("utf-8"))
        cache_digest.update(b"\0")
        cache_digest.update(topic.encode("utf-8"))
        cache_digest.update(b"\0")
    cache_digest.update(repr(sorted(policy.items())).encode("utf-8"))
    cache_key = cache_digest.hexdigest()
    if cache_key in _EXAM_STEM_TEMPLATE_CACHE:
        _EXAM_STEM_TEMPLATE_CACHE.move_to_end(cache_key)
        return [dict(item) for item in _EXAM_STEM_TEMPLATE_CACHE[cache_key]]

    stems = []
    for source in exam_sources:
        for line in re.split(r"\r?\n+", str(source.get("text", ""))):
            line = re.sub(r"^\s*(?:\d+[.)]|[A-D][.)])\s*", "", line).strip()
            if not 10 <= len(line) <= 300:
                continue
            if not (
                "?" in line
                or re.match(
                    r"^(?:true\s+or\s+false|t/f|what|which|who|when|where|why|how|"
                    r"define|identify|name|state|select|choose)\b",
                    line,
                    re.IGNORECASE,
                )
            ):
                continue
            if re.match(r"^(?:true\s+or\s+false|t/f)\b", line, re.IGNORECASE):
                question_type, pattern, template = (
                    "True or False",
                    "true-false",
                    "True or False: {content}",
                )
            elif re.match(r"^(?:define|identify|name|state)\b", line, re.IGNORECASE):
                question_type = "Identification"
                if re.match(r"^define\b", line, re.IGNORECASE):
                    pattern, template = "define", "Define the concept using this passage: {content}"
                else:
                    pattern, template = "identify", "Identify the concept from this passage: {content}"
            else:
                question_type = "Multiple Choice"
                if re.match(r"^which\s+of\s+the\s+following\b", line, re.IGNORECASE):
                    pattern = "which-of-following"
                    template = "Which of the following concepts is described by this passage? {content}"
                elif re.match(r"^what\b", line, re.IGNORECASE):
                    pattern, template = "what", "What concept is described by this passage? {content}"
                else:
                    pattern, template = "which", "Which concept is described by this passage? {content}"
            stems.append((line, question_type, pattern, template))

    stems = [stem for stem in stems if clean_text(stem[0])]
    if not stems:
        _cache_exam_stem_templates(cache_key, [])
        return []
    try:
        distributions = predict_distributions(
            [line for line, _, _, _ in stems],
            training_examples,
            training_metrics=training_metrics,
        )
    except Exception:
        distributions = []
        classified_stems = []
        for stem in stems:
            try:
                prediction = predict_distributions(
                    [stem[0]],
                    training_examples,
                    training_metrics=training_metrics,
                )
            except Exception as error:
                logger.warning("Skipping exam stem after classifier error: %s", error)
                continue
            if prediction and prediction[0]:
                classified_stems.append(stem)
                distributions.append(prediction[0])
        stems = classified_stems
    frequencies: Counter[tuple[str, str, str]] = Counter()
    templates = {}
    for (stem, question_type, pattern, template), distribution in zip(stems, distributions):
        classified = classify_with_policy(
            stem,
            training_examples,
            dataset_id,
            policy,
            distribution,
        )
        if classified["topic"] == "Unclassified":
            continue
        key = (question_type, classified["topic"], pattern)
        frequencies[key] += 1
        templates[key] = template
    templates = [
        {
            "question_type": question_type,
            "topic": topic,
            "pattern": pattern,
            "template": templates[(question_type, topic, pattern)],
            "frequency": frequency,
        }
        for (question_type, topic, pattern), frequency in sorted(
            frequencies.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    _cache_exam_stem_templates(cache_key, templates)
    return [dict(item) for item in templates]


def _style_question(
    question_type: str,
    topic: str,
    content: str,
    style_templates: Sequence[Mapping],
    order: int,
) -> str | None:
    matches = [
        item for item in style_templates
        if item.get("question_type") == question_type
        and item.get("topic") == topic
    ]
    if not matches:
        matches = [
            item for item in style_templates
            if item.get("question_type") == question_type
        ]
    if not matches:
        return None
    matches = sorted(matches, key=lambda item: (-int(item.get("frequency", 0)), item.get("pattern", "")))
    selected = matches[order % len(matches)]
    return str(selected["template"]).replace("{content}", content)


def rank_lesson_likelihood(
    questions: list[dict],
    lesson_text: str,
    training_examples: Sequence[tuple[str, str]],
    *,
    exam_sources: Sequence[Mapping] = (),
    recent_question_topics: Sequence[str] = (),
    emphasized_terms: Sequence[str] = (),
    heading_terms: Sequence[str] = (),
    list_terms: Sequence[str] = (),
    training_metrics: dict | None = None,
    dataset_id: str = "builtin",
    decision_policy: Mapping | None = None,
) -> dict:
    policy = dict(decision_policy) if decision_policy is not None else get_decision_policy(
        {"id": dataset_id, "examples": training_examples}
    )
    topics = sorted({label.strip() for _, label in training_examples if label.strip()})
    passages = _lesson_passages(lesson_text)
    distributions = predict_distributions(
        [passage["text"] for passage in passages],
        training_examples,
        training_metrics=training_metrics,
    ) if passages else []
    frequencies = Counter(
        token.casefold()
        for passage in passages
        for token in clean_text(passage["text"]).split()
    )
    tfidf_weights = dict.fromkeys(range(len(passages)), 0.0)
    if passages:
        passage_matrix = TfidfVectorizer().fit_transform(
            [passage["text"] for passage in passages]
        )
        for index, row in enumerate(passage_matrix):
            weights = row.data
            tfidf_weights[index] = (
                sum(sorted(weights, reverse=True)[:5]) / 5
                if len(weights) else 0.0
            )
    maximum_tfidf = max(tfidf_weights.values(), default=0.0) or 1.0
    maximum_frequency = max(frequencies.values(), default=1)
    emphasized = Counter(
        token.casefold()
        for term in emphasized_terms
        for token in clean_text(term).split()
    )
    heading_phrases = [clean_text(term) for term in heading_terms if clean_text(term)]
    list_phrases = [clean_text(term) for term in list_terms if clean_text(term)]
    emphasis_scores = dict.fromkeys(topics, 0.0)
    topic_passage_counts = Counter()
    emphasis_signal_counts: dict[str, Counter] = {
        topic: Counter() for topic in topics
    }
    analyzed_passages = []
    for passage_index, (passage, distribution) in enumerate(zip(passages, distributions)):
        classification = classify_with_policy(
            passage["text"],
            training_examples,
            dataset_id,
            policy,
            distribution,
        )
        topic = classification["topic"]
        cleaned_passage = clean_text(passage["text"])
        passage_tokens = set(cleaned_passage.split())
        emphasized_in_passage = sorted(passage_tokens & set(emphasized))
        repeated_in_passage = sorted(
            token for token in passage_tokens if frequencies[token] >= 3
        )
        is_heading = passage["heading"] or any(
            phrase in cleaned_passage for phrase in heading_phrases
        )
        is_list = passage["list_item"] or any(
            phrase in cleaned_passage for phrase in list_phrases
        )
        analyzed_passages.append(
            {
                "text": passage["text"],
                "topic": topic,
                "confidence": round(float(classification["confidence"]), 3),
                "margin": round(float(classification["margin"]), 3),
                "models_agree": classification["models_agree"],
                "abstain_reason": classification["abstain_reason"],
                "signals": {
                    "heading": is_heading,
                    "definition": passage["definition"],
                    "list": is_list,
                    "emphasized_terms": emphasized_in_passage,
                    "repeated_terms": repeated_in_passage,
                },
            }
        )
        if topic == "Unclassified":
            continue
        topic_passage_counts[topic] += 1
        emphasis_signal_counts[topic]["headings"] += int(is_heading)
        emphasis_signal_counts[topic]["definitions"] += int(passage["definition"])
        emphasis_signal_counts[topic]["lists"] += int(is_list)
        emphasis_signal_counts[topic]["emphasized_terms"] += len(emphasized_in_passage)
        emphasis_signal_counts[topic]["repeated_terms"] += len(repeated_in_passage)
        tokens = clean_text(passage["text"]).split()
        repeated = sum(max(0, frequencies[token] - 1) for token in set(tokens))
        emphasized_count = sum(emphasized[token] for token in set(tokens))
        emphasis_scores[topic] += (
            0.25 * tfidf_weights[passage_index] / maximum_tfidf
            + (0.20 if is_heading else 0.0)
            + (0.20 if passage["definition"] else 0.0)
            + (0.10 if is_list else 0.0)
            + 0.10 * min(1.0, repeated / max(1, len(set(tokens))))
            + 0.15 * min(1.0, emphasized_count / max(1, len(set(tokens))))
            + 0.10 * sum(frequencies[token] / maximum_frequency for token in set(tokens))
              / max(1, len(set(tokens)))
        )

    maximum_emphasis = max(emphasis_scores.values(), default=0.0)
    if maximum_emphasis:
        emphasis_scores = {
            topic: score / maximum_emphasis
            for topic, score in emphasis_scores.items()
        }

    exam_counts = Counter()
    exam_questions_by_topic: dict[str, list[str]] = {}
    all_exam_question_lines = []
    for source in exam_sources:
        exam_text = str(source.get("text", ""))
        source_passages = _lesson_passages(exam_text)
        source_distributions = predict_distributions(
            [passage["text"] for passage in source_passages],
            training_examples,
        ) if source_passages else []
        file_topics = set()
        for passage, distribution in zip(source_passages, source_distributions):
            classified = classify_with_policy(
                passage["text"],
                training_examples,
                dataset_id,
                policy,
                distribution,
            )
            if classified["topic"] != "Unclassified":
                file_topics.add(classified["topic"])
        exam_counts.update(file_topics)
        for line in exam_text.splitlines():
            line = line.strip(" \t-•")
            if len(line) >= 24 and (
                "?" in line
                or re.match(
                    r"^(?:what|which|who|when|where|why|how|define|explain|describe|"
                    r"compare|list|state|identify|discuss)\b",
                    line,
                    re.IGNORECASE,
                )
            ) and clean_text(line):
                all_exam_question_lines.append(line)
                if len(all_exam_question_lines) >= 500:
                    break
    if all_exam_question_lines:
        exam_question_distributions = predict_distributions(
            all_exam_question_lines, training_examples
        )
        for line, distribution in zip(all_exam_question_lines, exam_question_distributions):
            classified = classify_with_policy(
                line,
                training_examples,
                dataset_id,
                policy,
                distribution,
            )
            if classified["topic"] != "Unclassified":
                exam_questions_by_topic.setdefault(classified["topic"], []).append(line)
    has_exam_evidence = bool(exam_sources)
    maximum_exam_count = max(exam_counts.values(), default=1)
    recent = Counter(topic.casefold() for topic in recent_question_topics)
    style_by_topic = dict.fromkeys(topics, 0.0)
    style_by_question = {}
    if exam_questions_by_topic and questions:
        exam_lines = [
            (topic, line)
            for topic, lines in exam_questions_by_topic.items()
            for line in lines[:200]
        ]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        try:
            matrix = vectorizer.fit_transform(
                [item["question_text"] for item in questions]
                + [line for _, line in exam_lines]
            )
            similarities = cosine_similarity(
                matrix[:len(questions)],
                matrix[len(questions):],
            )
            topic_scores = {topic: [] for topic in topics}
            for question_index, item in enumerate(questions):
                matching = [
                    similarities[question_index, index]
                    for index, (topic, _) in enumerate(exam_lines)
                    if topic == item["topic"]
                ]
                style_by_question[item["id"]] = max(matching, default=0.0)
                if matching:
                    topic_scores[item["topic"]].append(max(matching))
            for topic in topics:
                style_by_topic[topic] = max(topic_scores[topic], default=0.0)
        except ValueError:
            style_by_question = {}

    topic_scores = []
    for topic in topics:
        exam_frequency = exam_counts[topic] / maximum_exam_count
        gap = (
            1.0 if exam_counts[topic] and recent[topic.casefold()] == 0 else 0.0
        )
        if has_exam_evidence:
            score = (
                0.40 * exam_frequency
                + 0.30 * emphasis_scores.get(topic, 0.0)
                + 0.20 * style_by_topic.get(topic, 0.0)
                + 0.10 * gap
            )
        else:
            score = emphasis_scores.get(topic, 0.0)
        explanations = []
        signals = emphasis_signal_counts[topic]
        emphasis_reasons = []
        for key, label in (
            ("headings", "heading"),
            ("definitions", "definition"),
            ("lists", "list"),
        ):
            if signals[key]:
                emphasis_reasons.append(f"{signals[key]} lesson {label}(s)")
        if signals["emphasized_terms"]:
            emphasis_reasons.append("emphasized key terms")
        if signals["repeated_terms"]:
            emphasis_reasons.append("repeated key terms")
        if emphasis_scores.get(topic, 0.0) > 0:
            emphasis_reasons.append("TF-IDF term emphasis")
        if emphasis_reasons:
            explanations.append("Lesson emphasis: " + ", ".join(emphasis_reasons))
        if has_exam_evidence:
            explanations.append(
                f"Appeared in {exam_counts[topic]} past exam file(s)"
                if exam_counts[topic]
                else "Not classified in uploaded past exams"
            )
            if gap:
                explanations.append("No recent saved practice question for this exam topic")
            if style_by_topic.get(topic, 0.0) >= 0.2:
                explanations.append("Lesson questions resemble past-exam wording")
        else:
            explanations.append("Ranked by lesson emphasis; no past exam PDFs found")
        topic_scores.append(
            {
                "topic": topic,
                "likelihood_score": round(score, 4),
                "likelihood": _likelihood_label(score),
                "why": explanations,
                "lesson_passages": topic_passage_counts[topic],
                "exam_files": exam_counts[topic],
            }
        )
    topic_scores.sort(
        key=lambda item: (-item["likelihood_score"], item["topic"].casefold())
    )
    topic_score_by_name = {item["topic"]: item for item in topic_scores}

    for item in questions:
        topic_score = topic_score_by_name.get(item["topic"])
        item["likelihood_score"] = (
            topic_score["likelihood_score"] if topic_score else 0.0
        )
        item["likelihood"] = topic_score["likelihood"] if topic_score else "Low"
        item["likelihood_why"] = list(topic_score["why"]) if topic_score else [
            "Question topic could not be classified from the lesson"
        ]
        style_score = style_by_question.get(item["id"], 0.0)
        if style_score >= 0.2:
            item["likelihood_why"].append("Question wording resembles a past-exam item")
        elif has_exam_evidence:
            item["likelihood_why"].append("No strong past-exam wording match")
        if has_exam_evidence and style_by_question:
            item["likelihood_score"] = round(
                0.8 * item["likelihood_score"] + 0.2 * style_score, 4
            )
            item["likelihood"] = _likelihood_label(item["likelihood_score"])

    return {
        "lesson_analysis": {
            "passages": analyzed_passages,
            "unclassified_count": sum(
                passage["topic"] == "Unclassified" for passage in analyzed_passages
            ),
        },
        "likely_topics": topic_scores,
        "exam_evidence_available": has_exam_evidence,
        "exam_sources": [str(source.get("name", "")) for source in exam_sources],
        "ranking_formula": (
            {
                "past_exam_frequency": 0.40,
                "lesson_emphasis": 0.30,
                "exam_style_similarity": 0.20,
                "coverage_gap": 0.10,
            }
            if has_exam_evidence else {"lesson_emphasis": 1.0}
        ),
        "disclaimer": (
            "Likelihood is an estimate from your materials, not a guarantee of actual exam content."
        ),
    }


def _likelihood_label(score: float) -> str:
    if score >= 0.67:
        return "High"
    if score >= 0.34:
        return "Medium"
    return "Low"


def validate_candidate(
    candidate: Mapping,
    question_type: str,
    other_correct_answers: Sequence[str] = (),
) -> bool:
    question = str(candidate.get("question_text", "")).strip()
    answer = str(candidate.get("correct_answer", "")).strip()
    source = str(candidate.get("source_reference", "")).strip()
    excerpt = str(candidate.get("support_excerpt", "")).strip()
    cited_passage = re.sub(
        r"\s+", " ", str(candidate.get("cited_passage", "")).strip()
    )
    cited_folded = cited_passage.casefold()
    if (
        question_type not in QUESTION_TYPES
        or candidate.get("difficulty") not in DIFFICULTIES
        or not candidate.get("topic")
        or not answer
        or not source
        or not excerpt
        or not cited_passage
        or not 24 <= len(question) <= 1000
        or cited_passage.casefold() not in re.sub(r"\s+", " ", source).casefold()
        or excerpt.casefold() not in str(candidate.get("source_content", "")).casefold()
        or cited_folded not in re.sub(
            r"\s+", " ", str(candidate.get("source_content", ""))
        ).casefold()
    ):
        return False
    choices = candidate.get("choices", [])
    if question_type == "Multiple Choice":
        normalized = [
            re.sub(r"\s+", " ", str(choice)).strip().casefold()
            for choice in choices
        ]
        wrong_choices = [
            str(choice).strip()
            for choice in choices
            if str(choice).strip().casefold() != answer.casefold()
        ]
        other_answers = {
            re.sub(r"\s+", " ", str(item)).strip().casefold()
            for item in other_correct_answers
        }
        return (
            len(normalized) == 4
            and all(normalized)
            and len(set(normalized)) == 4
            and answer.casefold() in normalized
            and re.search(
                rf"(?<!\w){re.escape(answer.casefold())}(?!\w)", cited_folded
            ) is not None
            and all(
                _usable_distractor(
                    choice,
                    answer,
                    question,
                    cited_passage,
                    other_answers,
                )
                for choice in wrong_choices
            )
        )
    if question_type == "Identification":
        return (
            not choices
            and re.search(
                rf"(?<!\w){re.escape(answer.casefold())}(?!\w)", cited_folded
            ) is not None
        )
    if (
        choices
        or answer not in {"True", "False"}
        or not question.startswith("True or False: ")
    ):
        return False
    statement = question.removeprefix("True or False: ").strip()
    if answer == "True":
        return statement == excerpt
    return statement == _negate_statement(excerpt)


def question_preference_features(question: Mapping) -> dict[str, str]:
    text = str(question.get("question_text", "")).strip().casefold()
    word_count = len(text.split())
    question_length = "short" if word_count < 12 else "medium" if word_count < 24 else "long"
    answer_word_count = len(str(question.get("correct_answer", "")).split())
    if text.startswith("true or false:"):
        wording_pattern = "true-or-false"
    elif text.startswith("which of the following"):
        wording_pattern = "which-of-the-following"
    elif text.startswith("what "):
        wording_pattern = "what"
    elif text.startswith("which "):
        wording_pattern = "which"
    elif text.startswith("define "):
        wording_pattern = "define"
    elif text.startswith("identify "):
        wording_pattern = "identify"
    else:
        wording_pattern = "other"
    return {
        "question_type": str(question.get("question_type", "")),
        "difficulty": str(question.get("difficulty", "")),
        "wording_pattern": wording_pattern,
        "question_length": question_length,
        "answer_length": "single-term" if answer_word_count <= 1 else "phrase",
    }


def question_preference_profile(
    feedback: Sequence[Mapping],
) -> dict[tuple[str, str], tuple[int, int]]:
    totals: dict[tuple[str, str], list[int]] = {}
    for entry in feedback:
        for feature, value in question_preference_features(entry).items():
            counts = totals.setdefault((feature, value), [0, 0])
            counts[0] += int(entry["rating"])
            counts[1] += 1
    return {key: (value[0], value[1]) for key, value in totals.items()}


def question_preference_score(
    question: Mapping,
    profile: Mapping[tuple[str, str], tuple[int, int]],
) -> float:
    scores = [
        profile[(feature, value)][0] / (profile[(feature, value)][1] + 2)
        for feature, value in question_preference_features(question).items()
        if (feature, value) in profile
    ]
    return sum(scores) / len(scores) if scores else 0.0


def generate_questions(
    *,
    requested_count: int,
    question_type: str,
    difficulty: str,
    topic_filter: str,
    per_topic: int,
    training_examples: Sequence[tuple[str, str]],
    dataset_id: str,
    dataset_name: str,
    sources: Sequence[Mapping],
    lesson_mode: bool = False,
    style_templates: Sequence[Mapping] = (),
    training_metrics: dict | None = None,
    saved_question_texts: Sequence[str] = (),
    preference_feedback: Sequence[Mapping] = (),
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    priority_weights: Mapping[str, float] | None = None,
    decision_policy: Mapping | None = None,
) -> dict:
    policy = dict(decision_policy) if decision_policy is not None else get_decision_policy(
        {"id": dataset_id, "examples": training_examples}
    )
    if question_type not in QUESTION_TYPES:
        raise ValueError("Choose a supported question type.")
    if difficulty not in DIFFICULTIES:
        raise ValueError("Choose Easy, Moderate, or Difficult difficulty.")
    if not 1 <= requested_count <= 50 or not 1 <= per_topic <= 50:
        raise ValueError("Question count and per-topic limit must be between 1 and 50.")
    if not 0.5 <= similarity_threshold <= 1:
        raise ValueError("Similarity threshold must be between 0.5 and 1.0.")
    weights = validate_weights(priority_weights)
    if not training_examples:
        raise ValueError("The selected training dataset is empty.")
    topics = sorted({label.strip() for _, label in training_examples if label.strip()})
    if len(topics) < 2:
        raise ValueError("The selected training dataset needs at least two topics.")
    material_sources = [
        source for source in sources
        if source.get("kind") in {"course", "lesson"}
    ]
    if not material_sources:
        raise ValueError("Select at least one saved course material source.")

    cleaned_sources = []
    removed_header_lines = 0
    for source in sources:
        source_text, removed = strip_document_headers(str(source.get("text", "")))
        removed_header_lines += removed
        if source_text.strip():
            cleaned_sources.append({**source, "text": source_text})
    sources = cleaned_sources
    source_texts = [str(source.get("text", "")) for source in sources if source.get("text")]
    if not source_texts:
        raise ValueError("No readable course materials or training examples were found.")
    combined_source = "\n".join(source_texts)
    if len(combined_source) > 500_000:
        raise ValueError("Selected source text exceeds the 500,000 character analysis limit.")

    candidate_sources = []
    diagnostics: Counter[str] = Counter()
    raw_sentence_count = 0
    for source in sources:
        raw_sentences = _SENTENCE_END.split(str(source.get("text", "")))
        raw_sentence_count += len(raw_sentences)
        for source_order, raw_sentence in enumerate(raw_sentences):
            sentence = raw_sentence.strip(" \t-•")
            if not 24 <= len(sentence) <= 500:
                diagnostics["outside_sentence_length"] += 1
                continue
            if sentence.endswith("?"):
                diagnostics["already_a_question"] += 1
                continue
            if not clean_text(sentence):
                diagnostics["no_readable_words"] += 1
                continue
            definition = _definition(sentence)
            eligible = estimate_difficulty(sentence) == difficulty or (
                lesson_mode
                and question_type in {"Multiple Choice", "Identification"}
                and definition is not None
            )
            if not eligible:
                diagnostics["different_difficulty"] += 1
                if question_type not in {"Multiple Choice", "Identification"} or not definition:
                    continue
            if question_type in {"Multiple Choice", "Identification"} and not definition:
                if eligible:
                    diagnostics["not_definition"] += 1
                continue
            candidate_sources.append((source, sentence, definition, eligible, source_order))

    distributions = predict_distributions(
        [combined_source, *(sentence for _, sentence, _, _, _ in candidate_sources)],
        training_examples,
        training_metrics=training_metrics,
    )
    aggregate = distributions[0]
    confidences = {entry["topic"]: entry["confidence"] for entry in aggregate}
    priority_topics = [*topics, "Unclassified"] if lesson_mode else topics
    ranked_topics = topic_priority(
        priority_topics, training_examples, [str(s["text"]) for s in material_sources],
        confidences, weights
    )
    priorities = {entry["topic"]: entry for entry in ranked_topics}
    if topic_filter != "All Topics" and topic_filter not in priorities:
        raise ValueError("The selected topic is not part of this training dataset.")
    allowed_topics = (
        [topic_filter] if topic_filter != "All Topics" else [item["topic"] for item in ranked_topics]
    )

    classified = []
    definitions_by_source: dict[tuple[str, str, str], list[str]] = {}
    lesson_definitions_by_source: dict[tuple[str, str], list[str]] = {}
    for candidate, distribution in zip(candidate_sources, distributions[1:]):
        source, sentence, definition, eligible, source_order = candidate
        classification = classify_with_policy(
            sentence,
            training_examples,
            dataset_id,
            policy,
            distribution,
        )
        best = {**distribution[0], **classification}
        if not lesson_mode and best["topic"] == "Unclassified":
            best["topic"] = distribution[0]["topic"]
        source_key = (
            str(source.get("kind", "course")),
            str(source.get("id") or source.get("name") or ""),
        )
        if lesson_mode and definition:
            lesson_definitions_by_source.setdefault(source_key, []).append(
                definition[0]
            )
        if lesson_mode and best["topic"] == "Unclassified":
            diagnostics["low_confidence_unclassified"] += 1
        classified.append((source, sentence, definition, eligible, best, source_order))
        if definition and best["abstain_reason"] is None:
            definitions_by_source.setdefault(
                (best["topic"], *source_key), []
            ).append(definition[0])

    training_definitions_by_topic: dict[str, list[str]] = {}
    if not lesson_mode:
        for text, topic in training_examples:
            for sentence in _SENTENCE_END.split(text):
                definition = _definition(sentence)
                if definition:
                    training_definitions_by_topic.setdefault(topic.strip(), []).append(
                        definition[0]
                    )
    candidates: list[dict] = []
    candidate_distractor_pools: dict[str, list[str]] = {}
    eligible_sentence_count = sum(candidate[3] for candidate in candidate_sources)
    for source, sentence, definition, eligible, best, source_order in classified:
        if not eligible:
            continue
        if best["topic"] not in allowed_topics:
            diagnostics["topic_not_selected"] += 1
            continue
        if best["abstain_reason"] is not None:
            if not lesson_mode:
                diagnostics["below_topic_relevance"] += 1
                continue

        candidate_question_type = question_type
        if question_type == "True or False":
            negated_sentence = _negate_statement(sentence)
            if not negated_sentence:
                diagnostics["no_safe_negation"] += 1
                continue
            question = _style_question(
                question_type, best["topic"], sentence, style_templates, source_order
            ) or f"True or False: {sentence}"
            answer = "True"
            choices: list[str] = []
            excerpt = sentence
        else:
            term, explanation = definition
            if question_type == "Identification":
                question = _style_question(
                    question_type, best["topic"], explanation, style_templates, source_order
                ) or f"Identify the concept described in the source: {explanation}"
                answer = term
                choices = []
            else:
                distractors = []
                seen_terms = {term.casefold()}
                current_source_key = (
                    str(source.get("kind", "course")),
                    str(source.get("id") or source.get("name") or ""),
                )
                same_source = (
                    definitions_by_source.get(
                        (best["topic"], *current_source_key), []
                    )
                    if current_source_key[0] in {"course", "lesson"}
                    else []
                )
                if lesson_mode and best["topic"] == "Unclassified":
                    same_source = [
                        distractor
                        for distractor in lesson_definitions_by_source.get(
                            current_source_key, []
                        )
                        if _similar_definition_term(term, distractor)
                    ]
                other_material_sources = [
                    distractor
                    for (topic, kind, source_id), terms in definitions_by_source.items()
                    if topic == best["topic"]
                    and kind in {"course", "lesson"}
                    and (kind, source_id) != current_source_key
                    for distractor in terms
                ]
                training_definitions = training_definitions_by_topic.get(
                    best["topic"], []
                )
                tiers = [same_source, other_material_sources]
                if not lesson_mode:
                    tiers.append(training_definitions)
                for tier in tiers:
                    for distractor in tier:
                        folded = re.sub(r"\s+", " ", distractor).strip().casefold()
                        if (
                            folded not in seen_terms
                            and _usable_distractor(
                                distractor,
                                term,
                                f"Which concept is described by the source passage? {explanation}",
                                sentence,
                                set(),
                            )
                        ):
                            seen_terms.add(folded)
                            distractors.append(distractor)
                        if len(distractors) == 3 and not lesson_mode:
                            break
                    if len(distractors) == 3 and not lesson_mode:
                        break
                if len(distractors) < 3:
                    if lesson_mode:
                        candidate_question_type = "True or False"
                        question = _style_question(
                            candidate_question_type,
                            best["topic"],
                            sentence,
                            style_templates,
                            source_order,
                        ) or f"True or False: {sentence}"
                        answer = "True"
                        choices = []
                        excerpt = sentence
                    else:
                        diagnostics["not_enough_distinct_distractors"] += 1
                        continue
                else:
                    wrong = distractors[:3]
                    choices = [term, *wrong]
                    seed = int(hashlib.sha256(sentence.encode("utf-8")).hexdigest()[:8], 16)
                    random.Random(seed ^ uuid.uuid4().int).shuffle(choices)
                    question = _style_question(
                        question_type, best["topic"], explanation, style_templates, source_order
                    ) or f"Which concept is described by the source passage? {explanation}"
                    answer = term
            if candidate_question_type != "True or False":
                excerpt = explanation

        source_label = str(source.get("name", "Selected source"))
        reference_text = sentence
        source_metadata = (
            f"Source Dataset: {dataset_name}\n"
            f"Source Material: {source_label}\n"
            f"Source Topic: {best['topic']}\n"
            f"Source Text/Reference: {reference_text}"
        )
        candidate_id = str(uuid.uuid4())
        if lesson_mode and candidate_question_type == "Multiple Choice":
            candidate_distractor_pools[candidate_id] = distractors
        candidate = {
            "id": candidate_id,
            "question_text": question,
            "question_type": candidate_question_type,
            "topic": best["topic"],
            "difficulty": difficulty,
            "choices": choices,
            "correct_answer": answer,
            "priority_score": priorities[best["topic"]]["priority_score"],
            "confidence_score": round(float(best["confidence"]), 4),
            "source_dataset_id": dataset_id,
            "source_dataset_name": dataset_name,
            "source_material_id": source.get("id"),
            "source_material_name": source_label,
            "source_reference": source_metadata,
            "cited_passage": sentence,
            "support_excerpt": excerpt,
            "source_content": str(source.get("text", "")),
            "lesson_order": source_order,
            "created_at": None,
        }
        if not validate_candidate(
            candidate,
            candidate_question_type,
        ):
            if candidate_question_type == "True or False" and lesson_mode:
                diagnostics["no_valid_distractors"] += 1
            diagnostics["candidate_validation_failed"] += 1
            continue
        candidates.append(candidate)

    if preference_feedback:
        preference_profile = question_preference_profile(preference_feedback)
        candidates.sort(
            key=lambda item: (
                -question_preference_score(item, preference_profile),
                -priorities[item["topic"]]["priority_score"],
                item["source_material_name"].casefold(),
            )
        )
    else:
        candidates.sort(
            key=lambda item: (
                -priorities[item["topic"]]["priority_score"],
                item["source_material_name"].casefold(),
            )
        )
    if lesson_mode and len(material_sources) > 1:
        by_source: dict[str, list[dict]] = {
            str(source.get("id") or source.get("name") or ""): []
            for source in material_sources
        }
        for candidate in candidates:
            by_source.setdefault(str(candidate.get("source_material_id") or ""), []).append(
                candidate
            )
        fair_share = math.ceil(requested_count / len(material_sources))
        balanced = []
        for offset in range(fair_share):
            for source in material_sources:
                rows = by_source.get(str(source.get("id") or source.get("name") or ""), [])
                if offset < len(rows):
                    balanced.append(rows[offset])
        balanced_ids = {id(candidate) for candidate in balanced}
        candidates = balanced + [
            candidate for candidate in candidates if id(candidate) not in balanced_ids
        ]
    true_false_offset = (
        int(hashlib.sha256(combined_source.encode("utf-8")).hexdigest()[:1], 16) % 2
    )
    counts: Counter[str] = Counter()
    accepted = []
    accepted_answers: set[str] = set()
    previous = list(saved_question_texts) + [
        text for text, _ in training_examples
        if text.strip().endswith("?")
        or re.match(
            r"^\s*(?:what|which|who|when|where|why|how|define|explain|describe|"
            r"compare|list|state|identify|discuss)\b",
            text,
            re.IGNORECASE,
        )
    ]
    for candidate in candidates:
        if len(accepted) >= requested_count:
            break
        if counts[candidate["topic"]] >= per_topic:
            diagnostics["per_topic_limit_reached"] += 1
            continue
        candidate_answer = str(candidate["correct_answer"]).casefold()
        accepted_distractors = {
            str(choice).casefold()
            for accepted_item in accepted
            for choice in accepted_item.get("choices", [])
            if str(choice).casefold() != str(accepted_item["correct_answer"]).casefold()
        }
        if lesson_mode and candidate["question_type"] == "Multiple Choice":
            options = [
                distractor
                for distractor in candidate_distractor_pools.get(candidate["id"], [])
                if distractor.casefold() not in accepted_answers
                and distractor.casefold() not in accepted_distractors
                and _usable_distractor(
                    distractor,
                    candidate["correct_answer"],
                    candidate["question_text"],
                    candidate["cited_passage"],
                    accepted_answers | accepted_distractors,
                )
            ]
            if candidate_answer in accepted_distractors or len(options) < 3:
                candidate = {
                    **candidate,
                    "question_type": "True or False",
                    "question_text": f"True or False: {candidate['cited_passage']}",
                    "correct_answer": "True",
                    "choices": [],
                    "support_excerpt": candidate["cited_passage"],
                }
            else:
                choices = [candidate["correct_answer"], *options[:3]]
                random.Random(
                    int(hashlib.sha256(candidate["cited_passage"].encode("utf-8")).hexdigest()[:8], 16)
                    ^ uuid.uuid4().int
                ).shuffle(choices)
                candidate = {**candidate, "choices": choices}
        candidate_answer = str(candidate["correct_answer"]).casefold()
        accepted_distractors = {
            str(choice).casefold()
            for accepted_item in accepted
            for choice in accepted_item.get("choices", [])
            if str(choice).casefold() != str(accepted_item["correct_answer"]).casefold()
        }
        if (
            candidate_answer in accepted_distractors
            or any(
                str(choice).casefold() in accepted_answers
                for choice in candidate.get("choices", [])
                if str(choice).casefold() != candidate_answer
            )
        ):
            diagnostics["distractor_is_answer_in_set"] += 1
            continue
        if not validate_candidate(
            candidate,
            candidate["question_type"],
            accepted_answers,
        ):
            diagnostics["candidate_validation_failed"] += 1
            continue
        if candidate["question_type"] == "True or False":
            excerpt = candidate["support_excerpt"]
            negated_excerpt = _negate_statement(excerpt)
            answer = (
                "True"
                if negated_excerpt is None or (len(accepted) + true_false_offset) % 2 == 0
                else "False"
            )
            statement = excerpt if answer == "True" else negated_excerpt
            question_text = _style_question(
                candidate["question_type"], candidate["topic"], statement, style_templates, len(accepted)
            ) or f"True or False: {statement}"
            candidate = {
                **candidate,
                "question_text": question_text,
                "correct_answer": answer,
            }
            if not validate_candidate(candidate, candidate["question_type"]):
                diagnostics["candidate_validation_failed"] += 1
                continue
        if is_duplicate(candidate["question_text"], previous, similarity_threshold):
            diagnostics["too_similar"] += 1
            continue
        accepted.append(candidate)
        accepted_answers.add(str(candidate["correct_answer"]).casefold())
        counts[candidate["topic"]] += 1
        previous.append(candidate["question_text"])
    return {
        "questions": accepted,
        "diagnostics": {
            "source_sentence_count": raw_sentence_count,
            "header_lines_skipped": removed_header_lines,
            "sentences_matching_filters": eligible_sentence_count,
            **(
                {
                    "low_confidence_unclassified": diagnostics["low_confidence_unclassified"],
                    "skipped_low_confidence": diagnostics["below_topic_relevance"],
                    "no_valid_distractors": diagnostics["no_valid_distractors"],
                }
                if lesson_mode else {}
            ),
            "rejections": [
                {"reason": reason, "count": count}
                for reason, count in sorted(diagnostics.items())
                if count and reason != "low_confidence_unclassified"
            ],
        },
        "topic_priorities": ranked_topics,
        "priority_weights": weights,
        "minimum_topic_confidence": float(policy["confidence_threshold"]),
        "requested_count": requested_count,
        "shortfall": max(0, requested_count - len(accepted)),
        "shortfall_reason": (
            (
                "The selected training data did not match source sentences with at least "
                f"{float(policy['confidence_threshold']):.0%} model relevance. Choose a better-matched "
                "dataset or source material."
                if (
                    not candidates
                    and eligible_sentence_count > 0
                    and diagnostics["below_topic_relevance"] >= eligible_sentence_count
                )
                else "Not enough distinct, source-supported questions passed the quality checks."
            )
            if len(accepted) < requested_count
            else None
        ),
        "disclaimer": (
            "Priority indicates topic relevance in the selected sources; it is not a "
            "prediction or guarantee that a topic will appear on an examination."
        ),
    }
