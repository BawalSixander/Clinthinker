import unittest
from unittest.mock import patch

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
