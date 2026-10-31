import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import database, main
from app.questions import (
    _definition,
    _negate_statement,
    _similar_definition_term,
    extract_exam_stem_templates,
    generate_questions,
    estimate_difficulty,
    is_duplicate,
    question_preference_profile,
    question_preference_score,
    rank_lesson_likelihood,
    strip_document_headers,
    topic_priority,
    validate_candidate,
)

TRAINING = [
    ("router routing packets network protocol connection", "Networking"),
    ("routing table forwards packets network layer router", "Networking"),
    ("network protocol communication packet switching", "Networking"),
    ("relational database tables schema keys database", "Databases"),
    ("database relation primary key query transactions", "Databases"),
    ("tables store records relational schema normalization", "Databases"),
]
MATERIAL = (
    "Router is a device that forwards data packets between networks using routing tables. "
    "Routing is the process of selecting a path for data packets across interconnected networks. "
    "Network protocol is a formal set of rules that govern communication between devices. "
    "Packet switching is a method that sends data in small units over a network."
)


class PracticeQuestionUnitTests(unittest.TestCase):
    def generate(self, question_type="Multiple Choice", sources=None, **kwargs):
        return generate_questions(
            requested_count=kwargs.pop("requested_count", 4),
            question_type=question_type,
            difficulty=kwargs.pop("difficulty", "Moderate"),
            topic_filter=kwargs.pop("topic_filter", "All Topics"),
            per_topic=kwargs.pop("per_topic", 4),
            training_examples=kwargs.pop("training_examples", TRAINING),
            dataset_id="dataset-1",
            dataset_name="Study set",
            sources=sources or [{"id": "upload-1", "name": "Networks lesson", "text": MATERIAL, "kind": "course"}],
            **kwargs,
        )

    def test_topic_ranking_uses_configured_weighted_components(self):
        ranked = topic_priority(
            ["Networking", "Databases"],
            TRAINING,
            [MATERIAL],
            {"Networking": 0.9, "Databases": 0.1},
            {
                "historical_frequency": 0.4,
                "course_importance": 0.3,
                "model_confidence": 0.2,
                "recent_occurrence": 0.1,
            },
        )
        self.assertEqual(ranked[0]["topic"], "Networking")
        self.assertFalse(ranked[0]["recent_occurrence_available"])
        self.assertEqual(ranked[0]["recent_occurrence_score"], 0.5)

    def test_question_ratings_prefer_liked_wording_without_using_topic(self):
        feedback = [
            {
                "question_type": "Multiple Choice",
                "difficulty": "Moderate",
                "question_text": "Which concept is described by this passage? sample",
                "rating": 1,
            },
            {
                "question_type": "Multiple Choice",
                "difficulty": "Moderate",
                "question_text": "Which concept is described by this passage? another",
                "rating": 1,
            },
            {
                "question_type": "Multiple Choice",
                "difficulty": "Moderate",
                "question_text": "What concept is described by this passage? sample",
                "rating": -1,
            },
        ]
        profile = question_preference_profile(feedback)
        liked_wording = {
            "question_type": "Multiple Choice",
            "difficulty": "Moderate",
            "topic": "Databases",
            "question_text": "Which concept is described by this passage? database",
        }
        disliked_wording = {
            **liked_wording,
            "topic": "Networking",
            "question_text": "What concept is described by this passage? router",
        }

        self.assertGreater(
            question_preference_score(liked_wording, profile),
            question_preference_score(disliked_wording, profile),
        )

    def test_question_ratings_use_answer_length_without_topic(self):
        profile = question_preference_profile([
            {
                "question_type": "Identification",
                "difficulty": "Moderate",
                "question_text": "Identify the concept described in this passage.",
                "correct_answer": "network protocol",
                "rating": 1,
            },
            {
                "question_type": "Identification",
                "difficulty": "Moderate",
                "question_text": "Identify the concept described in this passage.",
                "correct_answer": "packet switching",
                "rating": 1,
            },
            {
                "question_type": "Identification",
                "difficulty": "Moderate",
                "question_text": "Identify the concept described in this passage.",
                "correct_answer": "router",
                "rating": -1,
            },
        ])
        phrase_answer = {
            "question_type": "Identification",
            "difficulty": "Moderate",
            "question_text": "Identify the concept described in this passage.",
            "correct_answer": "routing protocol",
            "topic": "Databases",
        }
        single_term = {
            **phrase_answer,
            "correct_answer": "router",
            "topic": "Networking",
        }

        self.assertGreater(
            question_preference_score(phrase_answer, profile),
            question_preference_score(single_term, profile),
        )

    def test_lesson_likelihood_uses_lesson_only_without_exam_sources(self):
        result = rank_lesson_likelihood(
            [],
            MATERIAL,
            TRAINING,
            emphasized_terms=["router", "routing"],
        )
        self.assertFalse(result["exam_evidence_available"])
        self.assertEqual(result["ranking_formula"], {"lesson_emphasis": 1.0})
        self.assertTrue(result["likely_topics"])
        self.assertTrue(any(
            "no past exam PDFs" in reason
            for reason in result["likely_topics"][0]["why"]
        ))
        self.assertTrue(result["lesson_analysis"]["passages"])
        self.assertEqual(
            result["disclaimer"],
            "Likelihood is an estimate from your materials, not a guarantee of actual exam content.",
        )

    def test_low_confidence_lesson_passages_are_unclassified(self):
        passage = "Quasar spectroscopy measures unusual stellar emissions across distant galaxies."
        with patch(
            "app.questions.predict_distributions",
            side_effect=lambda texts, examples, **kwargs: [
                [{"topic": "Networking", "confidence": 0.2}]
                for _ in texts
            ],
        ):
            result = rank_lesson_likelihood([], passage, TRAINING)

        self.assertEqual(result["lesson_analysis"]["unclassified_count"], 1)
        self.assertEqual(result["lesson_analysis"]["passages"][0]["topic"], "Unclassified")

    def test_lesson_likelihood_uses_exam_frequency_and_coverage_gap(self):
        question = {
            "id": "question-1",
            "topic": "Networking",
            "question_text": "Which concept is described by the source passage? A router forwards data packets between networks.",
        }
        result = rank_lesson_likelihood(
            [question],
            MATERIAL,
            TRAINING,
            exam_sources=[{
                "name": "Networks_Exam.pdf",
                "text": MATERIAL + "\n" + question["question_text"],
            }],
        )
        networking = next(
            item for item in result["likely_topics"] if item["topic"] == "Networking"
        )
        self.assertTrue(result["exam_evidence_available"])
        self.assertEqual(networking["exam_files"], 1)
        self.assertIn("No recent saved practice question", " ".join(networking["why"]))
        self.assertIn("past-exam wording", " ".join(networking["why"]))
        self.assertEqual(
            result["ranking_formula"]["past_exam_frequency"],
            0.40,
        )

    def test_multiple_choice_generation_is_source_supported_and_answer_hidden_in_display_contract(self):
        result = self.generate()
        self.assertGreater(len(result["questions"]), 0)
        item = result["questions"][0]
        self.assertEqual(len(item["choices"]), 4)
        self.assertEqual(len({choice.casefold() for choice in item["choices"]}), 4)
        self.assertTrue(item["correct_answer"] in item["choices"])
        self.assertIn(item["support_excerpt"], item["source_content"])
        self.assertTrue(validate_candidate(item, "Multiple Choice"))

    def test_true_false_and_identification_types_are_valid(self):
        true_false = self.generate("True or False")
        identification = self.generate("Identification")
        self.assertTrue(true_false["questions"])
        answers = {item["correct_answer"] for item in true_false["questions"]}
        self.assertEqual(answers, {"True", "False"})
        self.assertTrue(
            all(validate_candidate(item, "True or False") for item in true_false["questions"])
        )
        false_question = next(
            item for item in true_false["questions"] if item["correct_answer"] == "False"
        )
        self.assertIn(" not ", false_question["question_text"])
        self.assertIn(false_question["support_excerpt"], false_question["source_content"])
        self.assertIsNone(_negate_statement("The source does not define this term."))
        self.assertIsNone(_negate_statement("The source isn't available in this example."))
        self.assertEqual(
            _negate_statement("The device can support this network protocol."),
            "The device cannot support this network protocol.",
        )
        self.assertEqual(
            _negate_statement("The device does support this network protocol."),
            "The device does not support this network protocol.",
        )
        self.assertEqual(
            _negate_statement("The device did support this network protocol."),
            "The device did not support this network protocol.",
        )
        self.assertTrue(identification["questions"])
        self.assertEqual(identification["questions"][0]["choices"], [])
        self.assertTrue(validate_candidate(identification["questions"][0], "Identification"))

    def test_true_false_candidate_rejects_unsupported_or_mismatched_claims(self):
        item = self.generate("True or False")["questions"][0]
        self.assertFalse(
            validate_candidate(
                {**item, "question_text": "True or False: an unsupported claim"},
                "True or False",
            )
        )
        self.assertFalse(
            validate_candidate(
                {**item, "correct_answer": "False"},
                "True or False",
            )
        )

    def test_difficulty_and_source_validation_reject_invalid_candidates(self):
        item = self.generate()["questions"][0]
        self.assertFalse(validate_candidate({**item, "difficulty": "Impossible"}, "Multiple Choice"))
        self.assertFalse(
            validate_candidate({**item, "source_content": "unrelated unsupported text"}, "Multiple Choice")
        )
        self.assertFalse(
            validate_candidate({**item, "correct_answer": "not an option"}, "Multiple Choice")
        )
        unrelated_passage = (
            "This unrelated cited passage is long enough to be accepted as a passage."
        )
        self.assertFalse(
            validate_candidate(
                {
                    **item,
                    "cited_passage": unrelated_passage,
                    "source_reference": item["source_reference"].replace(
                        item["cited_passage"], unrelated_passage
                    ),
                    "source_content": f'{item["source_content"]} {unrelated_passage}',
                },
                "Multiple Choice",
            )
        )
        self.assertFalse(
            validate_candidate(
                {**item, "choices": [item["choices"][0], "You", *item["choices"][2:]]},
                "Multiple Choice",
            )
        )
        stem_distractor_choices = list(item["choices"])
        stem_distractor_index = next(
            index for index, choice in enumerate(stem_distractor_choices)
            if choice.casefold() != item["correct_answer"].casefold()
        )
        stem_distractor_choices[stem_distractor_index] = "source passage"
        self.assertFalse(
            validate_candidate(
                {**item, "choices": stem_distractor_choices},
                "Multiple Choice",
            )
        )
        wrong_answer = next(
            choice for choice in item["choices"]
            if choice.casefold() != item["correct_answer"].casefold()
        )
        self.assertFalse(
            validate_candidate(
                item,
                "Multiple Choice",
                other_correct_answers=[wrong_answer],
            )
        )
        passage_word = next(
            word for word in item["cited_passage"].split()
            if word.casefold() not in item["correct_answer"].casefold()
            and len(word.strip(".,;:")) > 2
        ).strip(".,;:")
        self.assertFalse(
            validate_candidate(
                {**item, "choices": [item["correct_answer"], passage_word, *[
                    choice for choice in item["choices"]
                    if choice.casefold() not in {
                        item["correct_answer"].casefold(), passage_word.casefold()
                    }
                ][:2]]},
                "Multiple Choice",
            )
        )
        self.assertEqual(estimate_difficulty("A router forwards data between networks."), "Easy")
        self.assertEqual(
            estimate_difficulty(
                "A router is a device that forwards data packets between networks using routing tables."
            ),
            "Moderate",
        )
        self.assertIsNone(
            _definition("Which structure needs elements to be shifted when an item is inserted.")
        )
        self.assertIsNone(_definition("n offline website is available without internet access."))

    def test_duplicate_detection_flags_near_identical_question(self):
        self.assertTrue(
            is_duplicate(
                "Which concept is described by the source passage? A router forwards data packets.",
                ["Which concept is described by the source passage? A router forwards data packets"],
                0.8,
            )
        )
        self.assertFalse(is_duplicate("An unrelated topic in botany and plant growth", [], 0.8))

    def test_empty_small_and_unsupported_sources_report_insufficient_evidence(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.generate(training_examples=[])
        with self.assertRaisesRegex(ValueError, "at least two topics"):
            self.generate(training_examples=[("a long enough example about networks", "Networking")])
        with self.assertRaisesRegex(ValueError, "course material"):
            self.generate(sources=[{"name": "training only", "text": MATERIAL, "kind": "training"}])

    def test_shortfalls_are_reported_instead_of_filling_with_random_questions(self):
        result = self.generate(requested_count=40)
        self.assertGreater(result["shortfall"], 0)
        self.assertIsNotNone(result["shortfall_reason"])
        self.assertGreater(result["diagnostics"]["source_sentence_count"], 0)
        self.assertTrue(result["diagnostics"]["rejections"])

    def test_multiple_choice_distractors_are_distinct_and_defined_in_selected_sources(self):
        result = self.generate()
        defined_terms = {
            definition[0].casefold()
            for sentence in MATERIAL.split(". ")
            if (definition := _definition(sentence.strip().rstrip(".")))
        }
        for item in result["questions"]:
            self.assertEqual(len(item["choices"]), 4)
            self.assertEqual(len({choice.casefold() for choice in item["choices"]}), 4)
            self.assertTrue(
                all(choice.casefold() in defined_terms for choice in item["choices"])
            )
            self.assertEqual(item["topic"], "Networking")

    def test_lesson_mode_does_not_use_training_examples_as_distractors(self):
        lesson = (
            "Router is a device that forwards packets between networks using a routing table."
        )
        training = [
            *TRAINING,
            ("Gateway is a device that connects separate networks across boundaries.", "Networking"),
            ("Firewall is a system that filters network traffic according to security rules.", "Networking"),
            ("Switch is a device that forwards frames within a local network segment.", "Networking"),
        ]
        result = self.generate(
            sources=[{"id": "lesson-1", "name": "One lesson", "text": lesson, "kind": "lesson"}],
            training_examples=training,
            lesson_mode=True,
            requested_count=1,
            per_topic=1,
        )
        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["questions"][0]["question_type"], "True or False")
        self.assertTrue(
            validate_candidate(result["questions"][0], "True or False")
        )

    def test_lesson_low_confidence_definitions_are_unclassified_and_supported(self):
        lesson = (
            "Router is a network device that forwards data packets. "
            "Gateway is a network device that connects separate networks. "
            "Firewall is a security system that filters network traffic. "
            "Switch is a network device that forwards frames locally. "
            "Server is a computer system that provides network services."
        )
        with patch(
            "app.questions.predict_distributions",
            side_effect=lambda texts, examples, **kwargs: [
                [{"topic": "Networking", "confidence": 0.2}]
                for _ in texts
            ],
        ):
            result = self.generate(
                sources=[{
                    "id": "lesson-1",
                    "name": "One lesson",
                    "text": lesson,
                    "kind": "lesson",
                }],
                lesson_mode=True,
                difficulty="Easy",
                requested_count=4,
                per_topic=4,
            )

        self.assertEqual(len(result["questions"]), 4)
        self.assertTrue(
            all(item["topic"] == "Unclassified" for item in result["questions"])
        )
        for item in result["questions"]:
            self.assertTrue(validate_candidate(item, item["question_type"]))
            if item["question_type"] == "Multiple Choice":
                self.assertIn(item["correct_answer"], item["cited_passage"])
                self.assertTrue(
                    all(choice.casefold() in lesson.casefold() for choice in item["choices"])
                )
                self.assertTrue(
                    all(
                        _similar_definition_term(item["correct_answer"], choice)
                        for choice in item["choices"]
                        if choice.casefold() != item["correct_answer"].casefold()
                    )
                )
                self.assertFalse(
                    any(choice.casefold() in {"you", "code"} for choice in item["choices"])
                )
        self.assertEqual(result["diagnostics"]["skipped_low_confidence"], 0)
        self.assertGreater(result["diagnostics"]["low_confidence_unclassified"], 0)

    def test_lesson_mode_difficulty_keeps_recognized_definition_sentences(self):
        lesson = (
            "Cloud computing is the delivery of computing services such as servers, storage, "
            "databases, networking, and software over the internet. "
            "Virtualization is the technology that allows one physical server to run several "
            "virtual machines. "
            "Scalability is the ability of a system to handle a growing amount of work by "
            "adding more resources."
        )
        with patch(
            "app.questions.predict_distributions",
            side_effect=lambda texts, examples, **kwargs: [
                [{"topic": "Networking", "confidence": 0.2}]
                for _ in texts
            ],
        ):
            lesson_result = self.generate(
                sources=[{
                    "id": "lesson-1",
                    "name": "One lesson",
                    "text": lesson,
                    "kind": "lesson",
                }],
                lesson_mode=True,
                difficulty="Easy",
                requested_count=1,
                per_topic=1,
            )
            course_result = self.generate(
                sources=[{
                    "id": "course-1",
                    "name": "One course",
                    "text": lesson,
                    "kind": "course",
                }],
                lesson_mode=False,
                difficulty="Easy",
                requested_count=1,
                per_topic=1,
            )
        self.assertEqual(lesson_result["diagnostics"]["sentences_matching_filters"], 3)
        self.assertEqual(course_result["diagnostics"]["sentences_matching_filters"], 0)

    def test_exam_question_stem_patterns_are_topic_labeled_and_counted(self):
        exam_sources = [{
            "name": "Networks_Exam.pdf",
            "text": (
                "Which of the following controls packet routing?\n"
                "Which of the following forwards traffic between networks?\n"
                "Identify the concept that selects a path for packets."
            ),
        }]
        with patch(
            "app.questions.predict_distributions",
            side_effect=lambda texts, *args, **kwargs: [
                [{"topic": "Networking", "confidence": 0.9}] for _ in texts
            ],
        ) as classify:
            templates = extract_exam_stem_templates(exam_sources, TRAINING)
            cached_templates = extract_exam_stem_templates(exam_sources, TRAINING)
        self.assertEqual(cached_templates, templates)
        classify.assert_called_once()
        which_template = next(
            item for item in templates if item["pattern"] == "which-of-following"
        )
        self.assertEqual(which_template["question_type"], "Multiple Choice")
        self.assertEqual(which_template["topic"], "Networking")
        self.assertEqual(which_template["frequency"], 2)
        self.assertNotIn("packet routing", which_template["template"].casefold())

    def test_exam_stem_classifier_skips_empty_and_stopword_only_inputs(self):
        exam_sources = [{
            "name": "Networks_Exam.pdf",
            "text": (
                "\n"
                "   \n"
                "What should we do?\n"
                "Which network protocol forwards data packets?"
            ),
        }]
        with patch(
            "app.questions.predict_distributions",
            return_value=[[{"topic": "Networking", "confidence": 0.9}]],
        ) as classify:
            templates = extract_exam_stem_templates(exam_sources, TRAINING)
        classify.assert_called_once()
        self.assertEqual(
            classify.call_args.args[0],
            ["Which network protocol forwards data packets?"],
        )
        self.assertTrue(templates)

    def test_exam_stem_classifier_skips_one_failed_stem_and_continues(self):
        exam_sources = [{
            "name": "Networks_Exam.pdf",
            "text": (
                "Which network forwards data packets?\n"
                "What protocol routes packets between networks?"
            ),
        }]

        def classify(texts, *args, **kwargs):
            if len(texts) > 1:
                raise ValueError("one stem could not be classified")
            if texts[0].startswith("Which"):
                raise ValueError("bad stem")
            return [[{"topic": "Networking", "confidence": 0.9}]]

        with patch("app.questions.predict_distributions", side_effect=classify):
            templates = extract_exam_stem_templates(exam_sources, TRAINING)
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0]["frequency"], 1)

    def test_generation_varies_question_wording_with_exam_stem_templates(self):
        result = self.generate(
            style_templates=[{
                "question_type": "Multiple Choice",
                "topic": "Networking",
                "pattern": "what",
                "template": "What concept is described in this lesson? {content}",
                "frequency": 3,
            }],
            requested_count=1,
            per_topic=1,
        )
        self.assertTrue(result["questions"])
        self.assertTrue(
            result["questions"][0]["question_text"].startswith(
                "What concept is described in this lesson?"
            )
        )

    def test_lesson_generation_balances_candidates_across_uploaded_lessons(self):
        lessons = [
            {
                "id": "lesson-a",
                "name": "Routing lesson",
                "kind": "lesson",
                "text": (
                    "Router is a device that forwards data packets between networks using routing tables. "
                    "Routing is a process that selects paths for data packets across connected networks. "
                    "Network protocol is a formal set of rules that controls communication between devices. "
                    "Packet switching is a method that sends data in units across a computer network."
                ),
            },
            {
                "id": "lesson-b",
                "name": "Address lesson",
                "kind": "lesson",
                "text": (
                    "Network address is an identifier that locates a device on an interconnected network. "
                    "Packet route is a selected path that carries packets through connected network devices. "
                    "Network link is a connection that transfers data between neighboring network devices. "
                    "Router interface is a connection point that sends packets between separate networks."
                ),
            },
        ]
        result = self.generate(
            sources=lessons,
            lesson_mode=True,
            requested_count=2,
            per_topic=2,
        )
        self.assertGreaterEqual(len(result["questions"]), 2)
        self.assertEqual(
            {item["source_material_id"] for item in result["questions"]},
            {"lesson-a", "lesson-b"},
        )

    def test_document_headers_and_student_fields_are_excluded_from_generation(self):
        text = (
            "Name: Alex Student\n"
            "Section: 02\n"
            "Exam 1\n"
            "Page 1 of 2\n"
            "2 / 2\n"
            "Router is a device that forwards data packets between networks using routing tables.\n"
        )
        cleaned, removed = strip_document_headers(text)
        self.assertEqual(removed, 5)
        self.assertNotIn("Alex Student", cleaned)
        self.assertNotIn("Exam 1", cleaned)
        self.assertIn("Router is a device", cleaned)

        result = self.generate(
            "Identification",
            sources=[{
                "id": "upload-headers",
                "name": "Exam 1",
                "text": text,
                "kind": "course",
            }],
        )
        self.assertTrue(result["questions"])
        self.assertEqual(result["diagnostics"]["header_lines_skipped"], 5)
        self.assertNotIn("Alex Student", result["questions"][0]["source_content"])

    def test_low_model_relevance_returns_no_misattributed_questions(self):
        low_distribution = [
            {"topic": "Networking", "confidence": 0.40},
            {"topic": "Databases", "confidence": 0.60},
        ]
        with patch(
            "app.questions.predict_distributions",
            side_effect=lambda texts, *args, **kwargs: [
                low_distribution for _ in texts
            ],
        ):
            result = self.generate("True or False", requested_count=1)
        self.assertEqual(result["questions"], [])
        self.assertIn("45%", result["shortfall_reason"])


class PracticeQuestionApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        storage = Path(self.temporary.name)
        self.uploads = storage / "uploads"
        patches = (
            patch.object(database, "STORAGE", storage),
            patch.object(database, "UPLOADS", self.uploads),
            patch.object(database, "DB_PATH", storage / "assistant.sqlite3"),
            patch.object(main, "UPLOADS", self.uploads),
        )
        self.patchers = list(patches)
        for item in self.patchers:
            item.start()
        self.addCleanup(self.cleanup_patches)
        database.initialize()
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)
        main._question_batches.clear()

    def cleanup_patches(self):
        for item in reversed(self.patchers):
            item.stop()
        self.temporary.cleanup()

    def create_dataset_and_material(self):
        csv = (
            "topic,text\n"
            "Networking,router routing packets network protocol connection\n"
            "Networking,routing table forwards packets network layer router\n"
            "Networking,network protocol communication packet switching\n"
            "Databases,relational database tables schema keys database\n"
            "Databases,database relation primary key query transactions\n"
            "Databases,tables store records relational schema normalization\n"
        )
        dataset_response = self.client.post(
            "/api/training-datasets/csv",
            files={"file": ("study.csv", csv.encode(), "text/csv")},
        )
        self.assertEqual(dataset_response.status_code, 200, dataset_response.text)
        material_response = self.client.post(
            "/api/upload",
            files={"file": ("networks.txt", MATERIAL.encode(), "text/plain")},
        )
        self.assertEqual(material_response.status_code, 200, material_response.text)
        return dataset_response.json(), material_response.json()

    def test_generation_conceals_answer_until_requested_and_saves_to_database(self):
        dataset, material = self.create_dataset_and_material()
        response = self.client.post(
            "/api/questions/generate",
            json={
                "training_dataset_id": dataset["id"],
                "upload_ids": [material["id"]],
                "question_type": "Identification",
                "difficulty": "Moderate",
                "number_of_questions": 2,
                "questions_per_topic": 2,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        generated = response.json()
        self.assertTrue(generated["questions"])
        self.assertIn("diagnostics", generated)
        self.assertNotIn("correct_answer", generated["questions"][0])
        question_id = generated["questions"][0]["id"]
        answer = self.client.post(
            "/api/questions/answers",
            json={"generation_id": generated["generation_id"], "question_ids": [question_id]},
        )
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertTrue(answer.json()["answers"][question_id])

        saved = self.client.post(
            "/api/questions/save",
            json={"generation_id": generated["generation_id"], "question_ids": [question_id]},
        )
        self.assertEqual(saved.status_code, 200, saved.text)
        public_saved = self.client.get("/api/questions/saved")
        self.assertEqual(public_saved.status_code, 200)
        sessions = public_saved.json()["sessions"]
        self.assertEqual(len(sessions), 1)
        session = sessions[0]
        self.assertEqual(session["id"], generated["generation_id"])
        self.assertEqual(session["question_count"], 1)
        self.assertEqual(session["topic"], "All Topics")
        self.assertEqual(session["source_file"], "networks.txt")
        self.assertNotIn("correct_answer", session["questions"][0])
        answer_key = self.client.get(
            "/api/questions/saved?include_answers=true"
        ).json()["sessions"][0]["questions"][0]
        self.assertTrue(answer_key["correct_answer"])
        targeted_answer_key = self.client.get(
            f"/api/questions/saved?include_answers=true&session_id={generated['generation_id']}"
        ).json()["sessions"][0]["questions"][0]
        self.assertEqual(targeted_answer_key["correct_answer"], answer_key["correct_answer"])

        another_question_id = next(
            item["id"] for item in generated["questions"] if item["id"] != question_id
        )
        saved_again = self.client.post(
            "/api/questions/save",
            json={
                "generation_id": generated["generation_id"],
                "question_ids": [another_question_id],
            },
        )
        self.assertEqual(saved_again.status_code, 200, saved_again.text)
        sessions_after_second_save = self.client.get("/api/questions/saved").json()["sessions"]
        self.assertEqual(len(sessions_after_second_save), 1)
        self.assertEqual(sessions_after_second_save[0]["question_count"], 2)
        duplicate = self.client.post(
            "/api/questions/save",
            json={"generation_id": generated["generation_id"], "question_ids": [question_id]},
        )
        self.assertEqual(duplicate.status_code, 409)

        renamed = self.client.put(
            f"/api/questions/sessions/{generated['generation_id']}",
            json={"title": "Networking review"},
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(
            self.client.get("/api/questions/saved").json()["sessions"][0]["title"],
            "Networking review",
        )
        deleted = self.client.delete(
            f"/api/questions/sessions/{generated['generation_id']}"
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get("/api/questions/saved").json()["sessions"], [])

    def test_lesson_upload_is_deduplicated_and_generates_session_from_lesson(self):
        dataset, material = self.create_dataset_and_material()
        duplicate = self.client.post(
            "/api/lessons/upload",
            files={"file": ("networks.txt", MATERIAL.encode(), "text/plain")},
        )
        self.assertEqual(duplicate.status_code, 200, duplicate.text)
        lesson = duplicate.json()
        self.assertTrue(lesson["duplicate"])
        self.assertEqual(lesson["id"], material["id"])

        response = self.client.post(
            "/api/questions/generate",
            json={
                "training_dataset_id": dataset["id"],
                "upload_ids": [lesson["id"]],
                "lesson_upload_id": lesson["id"],
                "lesson_only": True,
                "question_type": "Multiple Choice",
                "difficulty": "Moderate",
                "number_of_questions": 2,
                "questions_per_topic": 2,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        generated = response.json()
        self.assertFalse(generated["exam_evidence_available"])
        self.assertEqual(generated["lesson_name"], "networks.txt")
        self.assertTrue(generated["lesson_analysis"]["passages"])
        self.assertTrue(generated["questions"])
        self.assertIn("networks.txt", generated["questions"][0]["source_reference"])
        source_reference = generated["questions"][0]["source_reference"]
        self.assertTrue(any(
            sentence.strip() in source_reference
            for sentence in MATERIAL.split(". ")
        ))

        saved = self.client.post(
            "/api/questions/save",
            json={
                "generation_id": generated["generation_id"],
                "question_ids": [generated["questions"][0]["id"]],
            },
        )
        self.assertEqual(saved.status_code, 200, saved.text)
        session = self.client.get("/api/questions/saved").json()["sessions"][0]
        self.assertEqual(session["source_file"], "networks.txt")

    def test_multiple_lessons_are_the_only_generation_sources_and_share_one_batch(self):
        dataset, _ = self.create_dataset_and_material()
        lesson_texts = [
            ("routing.txt", MATERIAL),
            (
                "switching.txt",
                "Frame forwarding is a process that moves network frames between connected devices. "
                "Ethernet switching is a method that delivers frames within a local network.",
            ),
        ]
        lessons = []
        for name, text in lesson_texts:
            uploaded = self.client.post(
                "/api/lessons/upload",
                files={"file": (name, text.encode(), "text/plain")},
            )
            self.assertEqual(uploaded.status_code, 200, uploaded.text)
            lessons.append(uploaded.json())

        captured = {}

        def capture_generation(**kwargs):
            captured.update(kwargs)
            return {"questions": []}

        with patch.object(main, "generate_questions", side_effect=capture_generation), patch.object(
            main, "rank_lesson_likelihood", return_value={}
        ):
            response = self.client.post(
                "/api/questions/generate",
                json={
                    "training_dataset_id": dataset["id"],
                    "upload_ids": [lesson["id"] for lesson in lessons],
                    "lesson_upload_ids": [lesson["id"] for lesson in lessons],
                    "lesson_only": True,
                    "question_type": "Multiple Choice",
                    "difficulty": "Moderate",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(captured["lesson_mode"])
        self.assertEqual(
            [source["id"] for source in captured["sources"]],
            [lesson["id"] for lesson in lessons],
        )
        self.assertTrue(all(source["kind"] == "lesson" for source in captured["sources"]))
        self.assertEqual(response.json()["lesson_names"], ["networks.txt", "switching.txt"])
        self.assertEqual(
            main._question_batches[response.json()["generation_id"]]["source_file"],
            "networks.txt, switching.txt",
        )

    def test_lesson_generation_rejects_nonlesson_content_sources(self):
        dataset, lesson = self.create_dataset_and_material()
        response = self.client.post(
            "/api/questions/generate",
            json={
                "training_dataset_id": dataset["id"],
                "upload_ids": [lesson["id"], "other-source"],
                "lesson_upload_ids": [lesson["id"]],
                "lesson_only": True,
                "question_type": "Multiple Choice",
                "difficulty": "Moderate",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("only selected lesson files", response.json()["detail"])

    def test_lesson_generation_discovers_uploaded_exam_pdfs_as_evidence_only(self):
        dataset, lesson = self.create_dataset_and_material()
        exam_id = "past-exam"
        exam_path = self.uploads / f"{exam_id}.pdf"
        exam_path.write_bytes(b"local fake PDF for mocked extraction")
        with database.connect() as db:
            db.execute(
                "INSERT INTO uploads(id, name, stored_path, bytes, text_chars) "
                "VALUES(?, ?, ?, ?, ?)",
                (
                    exam_id,
                    "Networks_Exam.pdf",
                    str(exam_path),
                    exam_path.stat().st_size,
                    len(MATERIAL),
                ),
            )
        extract_text = main.extract_text

        def extract_exam_text(name, content):
            if name == "Networks_Exam.pdf":
                return MATERIAL + "\nWhich protocol controls communication between devices?"
            return extract_text(name, content)

        with patch.object(main, "extract_text", side_effect=extract_exam_text):
            response = self.client.post(
                "/api/questions/generate",
                json={
                    "training_dataset_id": dataset["id"],
                    "upload_ids": [lesson["id"]],
                    "lesson_upload_id": lesson["id"],
                    "lesson_only": True,
                    "question_type": "Multiple Choice",
                    "difficulty": "Moderate",
                    "number_of_questions": 2,
                    "questions_per_topic": 2,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        generated = response.json()
        self.assertTrue(generated["exam_evidence_available"])
        self.assertIn("Networks_Exam.pdf", generated["exam_sources"])
        self.assertTrue(generated["style_reference_enabled"])
        self.assertEqual(generated["style_reference_files"], 1)
        self.assertGreater(generated["style_template_count"], 0)
        self.assertIn("fit_seconds", generated["classifier_training"])
        self.assertGreater(generated["classifier_training"]["exam_passages_added"], 0)
        self.assertTrue(generated["questions"])
        self.assertTrue(
            all(item["source_material_name"] == lesson["name"] for item in generated["questions"])
        )
        self.assertTrue(
            all("Networks_Exam.pdf" not in item["source_reference"] for item in generated["questions"])
        )

        with patch.object(main, "extract_exam_stem_templates", wraps=main.extract_exam_stem_templates) as extract:
            style_disabled = self.client.post(
                "/api/questions/generate",
                json={
                    "training_dataset_id": dataset["id"],
                    "upload_ids": [lesson["id"]],
                    "lesson_upload_id": lesson["id"],
                    "lesson_only": True,
                    "use_exam_style": False,
                    "question_type": "Multiple Choice",
                    "difficulty": "Moderate",
                    "number_of_questions": 2,
                    "questions_per_topic": 2,
                },
            )
        self.assertEqual(style_disabled.status_code, 200, style_disabled.text)
        self.assertFalse(style_disabled.json()["style_reference_enabled"])
        self.assertEqual(style_disabled.json()["style_reference_files"], 0)
        extract.assert_not_called()

    def test_generation_accepts_all_selected_materials_beyond_ten_files(self):
        upload_ids = []
        for index in range(11):
            response = self.client.post(
                "/api/upload",
                files={
                    "file": (
                        f"exam-{index}.txt",
                        MATERIAL.encode(),
                        "text/plain",
                    )
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            upload_ids.append(response.json()["id"])

        captured = {}

        def capture_sources(**kwargs):
            captured["sources"] = kwargs["sources"]
            return {"questions": []}

        with patch.object(main, "generate_questions", side_effect=capture_sources):
            response = self.client.post(
                "/api/questions/generate",
                json={
                    "training_dataset_id": "builtin",
                    "upload_ids": upload_ids,
                    "question_type": "Multiple Choice",
                    "difficulty": "Moderate",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            sum(source["kind"] == "course" for source in captured["sources"]),
            11,
        )

    def test_empty_training_data_unsupported_file_and_generation_failure_are_errors(self):
        invalid_upload = self.client.post(
            "/api/upload", files={"file": ("program.exe", b"not a document", "application/octet-stream")}
        )
        self.assertEqual(invalid_upload.status_code, 415)

        dataset, material = self.create_dataset_and_material()
        request = {
            "training_dataset_id": dataset["id"],
            "upload_ids": [material["id"]],
            "question_type": "Multiple Choice",
            "difficulty": "Difficult",
        }
        with patch.object(main, "generate_questions", side_effect=ValueError("Question generation failed.")):
            failed = self.client.post("/api/questions/generate", json=request)
        self.assertEqual(failed.status_code, 422)
        self.assertEqual(failed.json()["detail"], "Question generation failed.")

    def test_topic_lookup_and_invalid_priority_weights(self):
        dataset, material = self.create_dataset_and_material()
        topic_response = self.client.get(f"/api/training-datasets/{dataset['id']}/topics")
        self.assertEqual(topic_response.json()["topics"], ["Databases", "Networking"])
        invalid = self.client.post(
            "/api/questions/generate",
            json={
                "training_dataset_id": dataset["id"],
                "upload_ids": [material["id"]],
                "question_type": "True or False",
                "difficulty": "Easy",
                "priority_weights": {
                    "historical_frequency": 0.8,
                    "course_importance": 0.3,
                    "model_confidence": 0.2,
                    "recent_occurrence": 0.1,
                },
            },
        )
        self.assertEqual(invalid.status_code, 422)


if __name__ == "__main__":
    unittest.main()
