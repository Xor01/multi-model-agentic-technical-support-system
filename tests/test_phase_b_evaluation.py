import math
import unittest

from support_agent.phase_b_evaluation import (
    GOLDEN_SET,
    agent_metrics,
    classification_metrics,
    evaluate_golden_set,
    perplexity_from_loss,
    qa_exact_match,
    qa_token_f1,
    retrieval_metrics,
    router_metrics,
    trainer_log_frames,
)


class PhaseBEvaluationTests(unittest.TestCase):
    def test_perplexity_uses_exponential_and_guards_extreme_loss(self):
        self.assertAlmostEqual(perplexity_from_loss(2.0), math.exp(2.0))
        self.assertEqual(perplexity_from_loss(20.0), float("inf"))

    def test_trainer_logs_are_split_without_inventing_points(self):
        train_logs, eval_logs = trainer_log_frames(
            [
                {"epoch": 1, "loss": 2.0},
                {"epoch": 1, "eval_loss": 1.5},
                {"epoch": 2, "learning_rate": 0.001},
            ]
        )

        self.assertEqual(train_logs[["epoch", "loss"]].to_dict("records"), [{"epoch": 1, "loss": 2.0}])
        self.assertEqual(eval_logs[["epoch", "eval_loss"]].to_dict("records"), [{"epoch": 1, "eval_loss": 1.5}])

    def test_classification_metrics_include_confusion_and_per_class_recall(self):
        metrics = classification_metrics(
            y_true=["api", "api", "database", "database"],
            y_pred=["api", "database", "database", "database"],
            labels=["api", "database"],
        )

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision_macro"], (1.0 + 2 / 3) / 2)
        self.assertEqual(metrics["recall_macro"], 0.75)
        self.assertAlmostEqual(metrics["f1_macro"], (2 / 3 + 0.8) / 2)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metrics["per_class_recall"], {"api": 0.5, "database": 1.0})

    def test_qa_metrics_normalize_articles_punctuation_and_case(self):
        self.assertEqual(qa_exact_match("The Authorization header.", "authorization header"), 1.0)
        self.assertAlmostEqual(qa_token_f1("Bearer access token", "access token"), 0.8)
        self.assertEqual(qa_token_f1("", ""), 1.0)

    def test_retrieval_metrics_report_failed_queries(self):
        metrics = retrieval_metrics(
            relevant_by_query={"q1": {"d2"}, "q2": {"d3", "d4"}},
            ranked_by_query={"q1": ["d1", "d2"], "q2": ["d1", "d2"]},
            k=2,
        )

        self.assertEqual(metrics["mrr"], 0.25)
        self.assertEqual(metrics["recall_at_k"], 0.5)
        self.assertEqual(metrics["failed_queries"], ["q2"])

    def test_router_metrics_include_wrong_route_diagnostics(self):
        metrics = router_metrics(
            [
                {"id": "r1", "expected_route": "qa", "actual_route": "qa", "fallback": False, "latency_ms": 10},
                {"id": "r2", "expected_route": "tools", "actual_route": "fallback", "fallback": True, "latency_ms": 30},
            ]
        )

        self.assertEqual(metrics["routing_accuracy"], 0.5)
        self.assertEqual(metrics["fallback_rate"], 0.5)
        self.assertEqual(metrics["average_latency_ms"], 20.0)
        self.assertEqual(metrics["wrong_routes"], [{"id": "r2", "expected": "tools", "actual": "fallback"}])

    def test_agent_metrics_cover_success_latency_and_escalation(self):
        metrics = agent_metrics(
            [
                {"task_success": True, "latency_ms": 20, "escalated": False},
                {"task_success": False, "latency_ms": 40, "escalated": True},
            ]
        )

        self.assertEqual(metrics, {"task_success_rate": 0.5, "average_latency_ms": 30.0, "escalation_rate": 0.5})

    def test_golden_set_has_ten_project_specific_cases(self):
        self.assertEqual(len(GOLDEN_SET), 10)
        self.assertTrue(all(case["required"] for case in GOLDEN_SET))
        self.assertEqual(len({case["id"] for case in GOLDEN_SET}), 10)

    def test_required_case_regression_fails_gate(self):
        result = evaluate_golden_set(
            baseline_results={"G01": True, "G02": True},
            fine_tuned_results={"G01": False, "G02": True},
            golden_set=GOLDEN_SET[:2],
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["regressions"], ["G01"])
        self.assertEqual(result["failed_required"], ["G01"])

    def test_missing_golden_results_are_not_treated_as_passes(self):
        result = evaluate_golden_set({}, {}, GOLDEN_SET[:1])

        self.assertFalse(result["passed"])
        self.assertEqual(result["not_evaluated"], ["G01"])


if __name__ == "__main__":
    unittest.main()
