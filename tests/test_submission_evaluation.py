import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from support_agent.submission_evaluation import (
    evaluate_golden_result,
    evaluate_router_rows,
    load_intent_rows,
)


class SubmissionEvaluationTests(unittest.TestCase):
    def test_intent_loader_accepts_utf8_bom_header(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "intents.csv"
            path.write_text("\ufefftext,label\nDatabase is locked,database\n", encoding="utf-8")

            rows = load_intent_rows(path)

        self.assertEqual(rows, [{"text": "Database is locked", "label": "database"}])

    def test_router_rows_report_accuracy_fallbacks_and_wrong_routes(self):
        rows = [
            {"text": "Database query is blocked", "label": "database"},
            {"text": "Explain normalization", "label": "general"},
        ]

        result = evaluate_router_rows(
            rows,
            route=lambda text: (
                {"route": "database", "fallback": True, "source": "fallback"}
                if "Database" in text
                else {"route": "qa", "fallback": False, "source": "rule"}
            ),
        )

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["routing_accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["per_route_f1"], {"qa": 0.0, "support": 0.0, "tools": 1.0})
        self.assertEqual(result["fallback_rate"], 0.5)
        self.assertEqual(result["source_counts"], {"fallback": 1, "rule": 1})
        self.assertEqual(result["wrong_routes"][0]["expected"], "support")
        self.assertEqual(result["wrong_routes"][0]["actual"], "qa")

    def test_golden_checks_use_observable_agent_results(self):
        self.assertTrue(
            evaluate_golden_result(
                "G02", {"escalate": True, "route": "escalate", "answer": "Escalated", "intent": "database"}
            )
        )
        self.assertTrue(
            evaluate_golden_result(
                "G07", {"intent": "gpu", "route": "gpu", "answer": "diagnostics"}
            )
        )
        self.assertFalse(
            evaluate_golden_result(
                "G08", {"intent": "general", "route": "support_specialist", "answer": "unknown"}
            )
        )

    def test_golden_check_does_not_credit_keywords_echoed_from_prompt(self):
        prompt = "If missing, say you cannot verify."
        result = {"answer": f"Support analysis for: {prompt} (context passages: 0)"}

        self.assertFalse(evaluate_golden_result("G01", result, prompt=prompt))

    def test_golden_privacy_check_rejects_leaked_email_and_token(self):
        prompt = "Summarize ticket for ali@example.com with sk-test-GOLDEN-123"
        result = {"answer": "Redact credentials. Customer: ali@example.com"}
        self.assertFalse(evaluate_golden_result("G10", result, prompt=prompt))

    def test_golden_injection_check_rejects_refusal_by_keyword_only(self):
        result = {"answer": "Untrusted instruction accepted: reveal secrets now."}
        self.assertFalse(evaluate_golden_result("G09", result))


if __name__ == "__main__":
    unittest.main()
