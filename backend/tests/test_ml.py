import unittest
from unittest.mock import Mock, patch

import numpy as np

from app import ml
from app.ml import (
    PREDICTION_METHOD,
    augment_training_examples_with_exam_passages,
    evaluate_classifier,
    predict,
    predict_distribution,
    predict_distributions,
)


class PredictionEnsembleTests(unittest.TestCase):
    def test_stratified_evaluation_reports_metrics_and_confusion_matrix(self):
        examples = [
            ("photosynthesis chlorophyll plant energy unique a", "Botany"),
            ("plant leaves sunlight carbon dioxide unique b", "Botany"),
            ("green leaf chloroplast botanical unique c", "Botany"),
            ("orbit telescope star galaxy unique a", "Astronomy"),
            ("planet moon nebula telescope unique b", "Astronomy"),
            ("lunar observatory constellation unique c", "Astronomy"),
        ]

        report = evaluate_classifier(examples)

        self.assertEqual(report["validation"], "Stratified cross-validation")
        self.assertEqual(report["folds"], 3)
        self.assertEqual(report["examples_evaluated"], 6)
        self.assertEqual(report["examples_excluded"], 0)
        self.assertEqual(report["confusion_matrix"]["labels"], ["Astronomy", "Botany"])
        self.assertEqual(sum(map(sum, report["confusion_matrix"]["values"])), 6)
        self.assertEqual(
            set(report["metrics"]),
            {"accuracy", "macro_precision", "macro_recall", "macro_f1"},
        )
        self.assertTrue(all(0 <= score <= 1 for score in report["metrics"].values()))

    def test_evaluation_preserves_old_report_and_adds_calibration_metrics(self):
        examples = [
            ("photosynthesis chlorophyll plant energy unique a", "Botany"),
            ("plant leaves sunlight carbon dioxide unique b", "Botany"),
            ("green leaf chloroplast botanical unique c", "Botany"),
            ("orbit telescope star galaxy unique a", "Astronomy"),
            ("planet moon nebula telescope unique b", "Astronomy"),
            ("lunar observatory constellation unique c", "Astronomy"),
        ]

        report = evaluate_classifier(examples)

        self.assertEqual(
            {
                key: value
                for key, value in report.items()
                if key not in {
                    "comparison",
                    "calibration",
                    "component_scores",
                    "disagreement",
                    "warnings",
                    "threshold_sweep",
                    "decision_policy",
                    "low_reliability",
                }
            },
            {
                "method": "TF-IDF + SVM + Naive Bayes ensemble",
                "validation": "Stratified cross-validation",
                "folds": 3,
                "examples_evaluated": 6,
                "examples_excluded": 0,
                "topics": ["Astronomy", "Botany"],
                "metrics": {
                    "accuracy": 0.8333333333333334,
                    "macro_precision": 0.875,
                    "macro_recall": 0.8333333333333333,
                    "macro_f1": 0.8285714285714285,
                },
                "confusion_matrix": {
                    "labels": ["Astronomy", "Botany"],
                    "values": [[3, 0], [1, 2]],
                },
                "disclaimer": (
                    "Cross-validation estimates performance on this labeled dataset. "
                    "It does not measure accuracy on future exams or guarantee exam topics."
                ),
            },
        )
        self.assertEqual(
            set(report) - {
                "method",
                "validation",
                "folds",
                "examples_evaluated",
                "examples_excluded",
                "topics",
                "metrics",
                "confusion_matrix",
                "disclaimer",
            },
            {
                "comparison",
                "calibration",
                "component_scores",
                "disagreement",
                "warnings",
                "threshold_sweep",
                "decision_policy",
                "low_reliability",
            },
        )
        self.assertEqual(
            set(report["comparison"]),
            {"svm_only", "nb_only", "ensemble", "ensemble_calibrated"},
        )
        self.assertEqual(
            set(report["comparison"]["ensemble"]),
            {
                "accuracy",
                "macro_f1",
                "log_loss",
                "brier_score",
                "expected_calibration_error",
            },
        )
        self.assertEqual(
            report["comparison"]["ensemble_calibrated"]["accuracy"],
            report["comparison"]["ensemble"]["accuracy"],
        )
        sweep = report["threshold_sweep"]
        self.assertTrue(
            all(
                left["coverage"] >= right["coverage"]
                for left, right in zip(sweep, sweep[1:])
            )
        )
        policy = report["decision_policy"]
        self.assertIsInstance(policy["low_reliability"], bool)
        if not policy["low_reliability"]:
            selected = next(
                item for item in sweep
                if item["threshold"] == policy["confidence_threshold"]
            )
            self.assertGreaterEqual(selected["precision"], policy["target_precision"])
            self.assertGreaterEqual(selected["coverage"], 0.20)
        self.assertEqual(
            set(report["calibration"]),
            {"svm_temperature", "ensemble_temperature", "reliability_bins"},
        )
        self.assertEqual(
            set(report["calibration"]["reliability_bins"]),
            {"ensemble", "ensemble_calibrated"},
        )
        self.assertEqual(
            set(report["component_scores"]),
            {"svm_only", "nb_only"},
        )
        self.assertEqual(
            set(report["disagreement"]),
            {
                "agreement_rate",
                "accuracy_when_agree",
                "accuracy_when_disagree",
                "agreement_count",
                "disagreement_count",
            },
        )
        self.assertIsInstance(report["warnings"], list)

    def test_comparison_metrics_match_hand_computed_ece_and_brier(self):
        probabilities = np.array([[0.8, 0.2], [0.4, 0.6]])
        labels = np.array([0, 1])

        metrics = ml._comparison_metrics(probabilities, labels)
        bins = ml._reliability_bins(probabilities, labels)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.2)
        self.assertAlmostEqual(metrics["expected_calibration_error"], 0.3)
        self.assertEqual(bins[6]["count"], 1)
        self.assertEqual(bins[8]["count"], 1)
        self.assertAlmostEqual(bins[6]["mean_confidence"], 0.6)
        self.assertAlmostEqual(bins[8]["accuracy"], 1.0)

    def test_cross_fitting_rows_are_disjoint_for_every_fold(self):
        fold_ids = np.array([0, 1, 2, 0, 1, 2])

        splits = list(ml._cross_fitting_indices(fold_ids))

        self.assertEqual(len(splits), 3)
        for fit_indices, scored_indices in splits:
            self.assertFalse(set(fit_indices) & set(scored_indices))
            self.assertEqual(set(fit_indices) | set(scored_indices), set(range(6)))

    def test_missing_fold_classes_are_zero_filled_and_probabilities_normalized(self):
        aligned, class_mask = ml._align_class_columns(
            np.array([[2.0], [-1.0]]),
            ["Astronomy"],
            ["Astronomy", "Botany"],
        )

        self.assertEqual(aligned.tolist(), [[2.0, 0.0], [-1.0, 0.0]])
        self.assertEqual(class_mask.tolist(), [True, False])
        probabilities = ml._probabilities_from_margins(aligned, class_mask)
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(2),
            rtol=0,
            atol=1e-12,
        )
        self.assertTrue(np.all(probabilities[:, 1] < 1e-10))

    def test_two_fold_evaluation_skips_calibrated_metrics(self):
        report = evaluate_classifier([
            ("photosynthesis chlorophyll plant energy", "Botany"),
            ("plant leaves sunlight carbon dioxide", "Botany"),
            ("orbit telescope star galaxy", "Astronomy"),
            ("planet moon nebula telescope", "Astronomy"),
        ])

        self.assertEqual(report["folds"], 2)
        self.assertIsNone(report["comparison"]["ensemble_calibrated"])
        self.assertIsNone(
            report["calibration"]["reliability_bins"]["ensemble_calibrated"]
        )
        self.assertTrue(any("Fewer than 3 folds" in warning for warning in report["warnings"]))

    def test_classify_without_stored_policy_matches_legacy_prediction(self):
        examples = [
            ("photosynthesis chlorophyll plant energy", "Botany"),
            ("plant leaves sunlight carbon dioxide", "Botany"),
            ("orbit telescope star galaxy", "Astronomy"),
            ("planet moon nebula telescope", "Astronomy"),
        ]
        text = "photosynthesis chlorophyll plant energy"

        legacy_prediction = predict_distribution(
            text,
            training_examples=examples,
        )[0]
        classification = ml.classify_with_policy(
            text,
            training_examples=examples,
            decision_policy=ml.LEGACY_DECISION_POLICY,
        )

        self.assertEqual(classification["topic"], legacy_prediction["topic"])
        self.assertAlmostEqual(
            classification["confidence"],
            legacy_prediction["confidence"],
            delta=1e-12,
        )
        self.assertIsNone(classification["abstain_reason"])
        self.assertFalse(ml.LEGACY_DECISION_POLICY["disagreement_abstain"])
        self.assertEqual(ml.LEGACY_DECISION_POLICY["min_margin"], 0.0)

    def test_disagreement_abstains_with_explicit_reason(self):
        vectorizer = Mock()
        vectorizer.transform.return_value = np.ones((1, 1))
        svm = Mock()
        svm.classes_ = np.array(["Astronomy", "Botany"])
        svm.decision_function.return_value = np.array([2.0])
        naive_bayes = Mock()
        naive_bayes.classes_ = svm.classes_
        naive_bayes.predict_proba.return_value = np.array([[0.9, 0.1]])
        policy = {
            **ml.LEGACY_DECISION_POLICY,
            "confidence_threshold": 0.3,
            "disagreement_abstain": True,
        }

        with patch.object(ml, "_get_models", return_value=(vectorizer, svm, naive_bayes)):
            result = ml.classify_with_policy(
                "test passage",
                training_examples=[("a", "Astronomy"), ("b", "Botany")],
                decision_policy=policy,
            )

        self.assertEqual(result["topic"], "Unclassified")
        self.assertEqual(result["abstain_reason"], "model_disagreement")
        self.assertFalse(result["models_agree"])

    def test_stratified_evaluation_rejects_topics_with_one_example(self):
        with self.assertRaisesRegex(ValueError, "at least two readable examples"):
            evaluate_classifier([
                ("photosynthesis plant", "Botany"),
                ("green leaves", "Botany"),
                ("orbit stars", "Astronomy"),
            ])

    def test_stratified_evaluation_reports_unreadable_labeled_rows(self):
        report = evaluate_classifier([
            ("photosynthesis chlorophyll plant", "Botany"),
            ("green leaves sunlight", "Botany"),
            ("orbit telescope galaxy", "Astronomy"),
            ("planet moon nebula", "Astronomy"),
            ("the and or", "Astronomy"),
        ])

        self.assertEqual(report["examples_evaluated"], 4)
        self.assertEqual(report["examples_excluded"], 1)

    def test_svm_and_naive_bayes_produce_a_ranked_ensemble(self):
        predictions = predict(
            "normalization relations functional dependencies sql transactions"
        )

        self.assertEqual(len(predictions), 5)
        self.assertEqual(predictions[0]["topic"], "Database Normalization")
        self.assertTrue(
            all(0 < item["confidence"] <= 1 for item in predictions)
        )
        self.assertEqual(
            PREDICTION_METHOD,
            "TF-IDF + SVM + Naive Bayes ensemble",
        )

    def test_custom_labeled_examples_define_prediction_topics(self):
        predictions = predict(
            "photosynthesis chlorophyll plant energy",
            training_examples=[
                ("photosynthesis chlorophyll plant energy", "Botany"),
                ("plant leaves sunlight carbon dioxide", "Botany"),
                ("orbit telescope star galaxy", "Astronomy"),
                ("planet moon nebula telescope", "Astronomy"),
            ],
        )

        self.assertEqual(predictions[0]["topic"], "Botany")
        self.assertEqual(
            {item["topic"] for item in predictions},
            {"Botany", "Astronomy"},
        )

    def test_binary_ensemble_confidences_are_normalized(self):
        predictions = predict_distribution(
            "photosynthesis chlorophyll plant energy",
            training_examples=[
                ("photosynthesis chlorophyll plant energy", "Botany"),
                ("plant leaves sunlight carbon dioxide", "Botany"),
                ("orbit telescope star galaxy", "Astronomy"),
                ("planet moon nebula telescope", "Astronomy"),
            ],
        )
        self.assertAlmostEqual(sum(item["confidence"] for item in predictions), 1.0)
        self.assertEqual(predictions[0]["topic"], "Botany")

    def test_score_components_preserve_predictions_and_normalize(self):
        examples = [
            ("photosynthesis chlorophyll plant energy", "Botany"),
            ("plant leaves sunlight carbon dioxide", "Botany"),
            ("orbit telescope star galaxy", "Astronomy"),
            ("planet moon nebula telescope", "Astronomy"),
        ]
        texts = ["photosynthesis chlorophyll", "orbit telescope"]
        expected = [
            [
                ("Botany", 0.7504643543191101),
                ("Astronomy", 0.2495356456808899),
            ],
            [
                ("Astronomy", 0.7807143832869377),
                ("Botany", 0.21928561671306238),
            ],
        ]

        predictions = predict_distributions(texts, training_examples=examples)
        vectorizer, svm, naive_bayes = ml._get_models(tuple(examples))
        labels, svm_scores, nb_scores, ensemble_scores = ml._score_components(
            [ml.clean_text(text) for text in texts],
            svm,
            naive_bayes,
            vectorizer,
        )

        self.assertEqual(labels.tolist(), ["Astronomy", "Botany"])
        for prediction, expected_row in zip(predictions, expected):
            self.assertEqual(
                [item["topic"] for item in prediction],
                [topic for topic, _ in expected_row],
            )
            for item, (_, expected_score) in zip(prediction, expected_row):
                self.assertLessEqual(
                    abs(item["confidence"] - expected_score),
                    1e-9,
                )
        np.testing.assert_array_equal(
            ensemble_scores,
            (svm_scores + nb_scores) / 2,
        )
        for score_rows in (svm_scores, nb_scores, ensemble_scores):
            np.testing.assert_allclose(
                score_rows.sum(axis=1),
                np.ones(len(texts)),
                rtol=0,
                atol=1e-9,
            )

    def test_temperature_one_preserves_session_one_scores(self):
        examples = [
            ("photosynthesis chlorophyll plant energy", "Botany"),
            ("plant leaves sunlight carbon dioxide", "Botany"),
            ("orbit telescope star galaxy", "Astronomy"),
            ("planet moon nebula telescope", "Astronomy"),
        ]
        texts = ["photosynthesis chlorophyll", "orbit telescope"]
        vectorizer, svm, naive_bayes = ml._get_models(tuple(examples))
        clean_texts = [ml.clean_text(text) for text in texts]

        baseline = ml._score_components(
            clean_texts, svm, naive_bayes, vectorizer
        )
        scaled = ml._score_components(
            clean_texts,
            svm,
            naive_bayes,
            vectorizer,
            svm_temperature=1.0,
            ensemble_temperature=1.0,
        )
        np.testing.assert_allclose(scaled[1], baseline[1], rtol=0, atol=1e-9)
        np.testing.assert_allclose(scaled[2], baseline[2], rtol=0, atol=1e-9)
        np.testing.assert_allclose(scaled[3], baseline[3], rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            ml.apply_ensemble_temperature(baseline[3], 1.0),
            baseline[3],
            rtol=0,
            atol=1e-9,
        )

    def test_temperatures_preserve_argmax_and_normalize_rows(self):
        random = np.random.default_rng(42)
        probabilities = random.random((200, 6))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        original_labels = np.argmax(probabilities, axis=1)
        margins = random.normal(size=(200, 6))
        bayes_probabilities = random.random((200, 6))
        bayes_probabilities /= bayes_probabilities.sum(axis=1, keepdims=True)
        svm = Mock()
        svm.classes_ = np.arange(6)
        svm.decision_function.return_value = margins
        naive_bayes = Mock()
        naive_bayes.predict_proba.return_value = bayes_probabilities

        for temperature in (0.05, 0.2, 0.7, 1.0, 3.0, 20.0):
            scaled = ml.apply_ensemble_temperature(probabilities, temperature)
            np.testing.assert_array_equal(
                np.argmax(scaled, axis=1),
                original_labels,
            )
            np.testing.assert_allclose(
                scaled.sum(axis=1),
                np.ones(len(probabilities)),
                rtol=0,
                atol=1e-12,
            )
            _, svm_scaled, _, _ = ml._score_components(
                np.empty((len(margins), 0)),
                svm,
                naive_bayes,
                svm_temperature=temperature,
            )
            np.testing.assert_array_equal(
                np.argmax(svm_scaled, axis=1),
                np.argmax(margins, axis=1),
            )
            np.testing.assert_allclose(
                svm_scaled.sum(axis=1),
                np.ones(len(margins)),
                rtol=0,
                atol=1e-12,
            )

    def test_fit_temperature_calibrates_overconfident_and_underconfident_scores(self):
        random = np.random.default_rng(123)

        overconfident_labels = random.integers(0, 2, size=1000)
        overconfident_predictions = overconfident_labels.copy()
        overconfident_predictions[:400] = 1 - overconfident_predictions[:400]
        overconfident_scores = np.full((1000, 2), -4.0)
        overconfident_scores[
            np.arange(1000), overconfident_predictions
        ] = 4.0
        overconfident_temperature = ml.fit_temperature(
            overconfident_scores,
            overconfident_labels,
        )
        self.assertGreater(overconfident_temperature, 1.0)

        underconfident_labels = random.integers(0, 2, size=1000)
        underconfident_predictions = underconfident_labels.copy()
        underconfident_predictions[:50] = 1 - underconfident_predictions[:50]
        underconfident_scores = np.full((1000, 2), -0.1)
        underconfident_scores[
            np.arange(1000), underconfident_predictions
        ] = 0.1
        underconfident_temperature = ml.fit_temperature(
            underconfident_scores,
            underconfident_labels,
        )
        self.assertLess(underconfident_temperature, 1.0)

    def test_fit_temperature_rejects_empty_or_mismatched_data(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            ml.fit_temperature(np.empty((0, 2)), [])
        with self.assertRaisesRegex(ValueError, "matching sample counts"):
            ml.fit_temperature(np.ones((2, 2)), [0])

    def test_batch_prediction_matches_individual_predictions(self):
        examples = [
            ("photosynthesis chlorophyll plant energy", "Botany"),
            ("plant leaves sunlight carbon dioxide", "Botany"),
            ("orbit telescope star galaxy", "Astronomy"),
            ("planet moon nebula telescope", "Astronomy"),
        ]
        texts = ["photosynthesis chlorophyll", "orbit telescope"]

        batched = predict_distributions(texts, training_examples=examples)
        individual = [
            predict_distribution(text, training_examples=examples)
            for text in texts
        ]

        for batch_result, individual_result in zip(batched, individual):
            self.assertEqual(
                [item["topic"] for item in batch_result],
                [item["topic"] for item in individual_result],
            )
            for batch_item, individual_item in zip(batch_result, individual_result):
                self.assertAlmostEqual(
                    batch_item["confidence"],
                    individual_item["confidence"],
                )

    def test_batch_prediction_skips_inputs_without_usable_words(self):
        examples = [
            ("photosynthesis chlorophyll plant energy", "Botany"),
            ("plant leaves sunlight carbon dioxide", "Botany"),
            ("orbit telescope star galaxy", "Astronomy"),
            ("planet moon nebula telescope", "Astronomy"),
        ]
        predictions = predict_distributions(
            [
                "",
                "   \t ",
                "What should we do?",
                "Which network protocol forwards packets?",
            ],
            training_examples=examples,
        )

        self.assertEqual(len(predictions), 4)
        self.assertEqual(predictions[:3], [[], [], []])
        self.assertTrue(predictions[3])
        self.assertEqual(predict_distributions([], training_examples=examples), [])

    def test_batch_fits_large_training_data_once(self):
        examples = tuple(
            (f"astronomy planet orbit star galaxy sample {index}", "Astronomy")
            if index % 2
            else (f"botany plant leaf photosynthesis sample {index}", "Botany")
            for index in range(2001)
        )
        with patch.object(ml, "_fit_uncached_models", wraps=ml._fit_uncached_models) as fit:
            predictions = predict_distributions(
                ["planet orbit", "plant photosynthesis"],
                training_examples=examples,
            )

        self.assertEqual(fit.call_count, 1)
        self.assertEqual(len(predictions), 2)

    def test_classifier_fit_is_on_demand_and_timing_is_reported(self):
        examples = [
            ("volcanic magma tectonic plates geology unique fit case", "Geology"),
            ("igneous rocks lava crust minerals unique fit sample", "Geology"),
            ("orbit telescope star galaxy unique fit case", "Astronomy"),
            ("planet nebula lunar observatory unique fit sample", "Astronomy"),
        ]
        ml._fit_models.cache_clear()
        self.addCleanup(ml._fit_models.cache_clear)
        first_metrics = {}
        predict_distributions(
            ["volcanic magma"], examples, training_metrics=first_metrics
        )
        self.assertEqual(first_metrics["fit_count"], 1)
        self.assertGreaterEqual(first_metrics["fit_seconds"], 0)

        cached_metrics = {}
        predict_distributions(
            ["orbit telescope"], examples, training_metrics=cached_metrics
        )
        self.assertEqual(cached_metrics["fit_count"], 0)
        self.assertEqual(cached_metrics["cache_hits"], 1)

    def test_confident_exam_passages_are_added_with_predicted_topic_labels(self):
        examples = [
            ("photosynthesis chlorophyll plant energy", "Botany"),
            ("plant leaves sunlight carbon dioxide", "Botany"),
            ("orbit telescope star galaxy", "Astronomy"),
            ("planet moon nebula telescope", "Astronomy"),
        ]
        with patch.object(
            ml,
            "predict_distributions",
            return_value=[[{"topic": "Botany", "confidence": 0.8}]],
        ):
            augmented, added = augment_training_examples_with_exam_passages(
                examples,
                [{"text": "Photosynthesis uses chlorophyll to convert sunlight into plant energy."}],
            )
        self.assertEqual(added, 1)
        self.assertEqual(augmented[-1][1], "Botany")
        self.assertIn("chlorophyll", augmented[-1][0])


if __name__ == "__main__":
    unittest.main()
