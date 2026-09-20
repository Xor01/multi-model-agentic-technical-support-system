import unittest

from support_agent.routing.classifier_router import baseline_router, resilient_router
from support_agent.routing.hybrid_router import hybrid_route
from support_agent.routing.llm_router import (
    RouterResponseError,
    build_router_prompt,
    parse_router_response,
)
from support_agent.routing.rules import rule_first


class RoutingTests(unittest.TestCase):
    def test_rule_first_routes_safety_and_documentation_phrases(self):
        self.assertEqual(
            rule_first("Production down after suspected corruption"),
            {"route": "escalate", "source": "rule", "confidence": 1.0},
        )
        self.assertEqual(
            rule_first("What does the documentation say?"),
            {"route": "qa", "source": "rule", "confidence": 1.0},
        )

    def test_baseline_router_uses_rules_before_classifier(self):
        def unexpected_classifier(_text):
            raise AssertionError("classifier should not run for a hard rule")

        result = baseline_router("We have a security breach", predict_intent=unexpected_classifier)

        self.assertEqual(result["route"], "escalate")

    def test_baseline_router_applies_confidence_policy(self):
        high = baseline_router(
            "Database queries are timing out",
            predict_intent=lambda _text: {"intent": "database", "confidence": 0.92},
        )
        low = baseline_router(
            "Something strange happened",
            predict_intent=lambda _text: {"intent": "general", "confidence": 0.40},
        )

        self.assertEqual(high["route"], "support_specialist")
        self.assertFalse(high["fallback"])
        self.assertEqual(low["route"], "support_specialist")
        self.assertTrue(low["fallback"])

    def test_resilient_router_uses_deterministic_route_when_classifier_is_unavailable(self):
        def unavailable_classifier(_text):
            raise RuntimeError("model artifacts are unavailable")

        result = resilient_router(
            "The API returns 503 after deployment.",
            predict_intent=unavailable_classifier,
        )

        self.assertEqual(result["route"], "deployment")
        self.assertEqual(result["intent"], "deployment")
        self.assertEqual(result["source"], "deterministic_fallback")
        self.assertTrue(result["fallback"])
        self.assertEqual(result["failure_mode"], "classifier_unavailable")

    def test_llm_prompt_contains_two_few_shot_examples(self):
        prompt = build_router_prompt("My service keeps restarting")

        self.assertEqual(len(prompt), 6)
        self.assertEqual(prompt[0]["role"], "system")
        self.assertEqual(prompt[-1], {"role": "user", "content": "My service keeps restarting"})
        example_routes = [message["content"] for message in prompt if message["role"] == "assistant"]
        self.assertEqual(example_routes, ['{"route":"tools"}', '{"route":"support_specialist"}'])

    def test_llm_response_requires_strict_allowed_route_json(self):
        self.assertEqual(
            parse_router_response('{"route":"qa"}'),
            {"route": "qa", "source": "llm"},
        )
        for invalid in (
            "```json\n{\"route\":\"qa\"}\n```",
            '{"route":"unknown"}',
            '{"route":"qa","reason":"extra"}',
        ):
            with self.subTest(invalid=invalid), self.assertRaises(RouterResponseError):
                parse_router_response(invalid)

    def test_hybrid_uses_llm_only_for_low_confidence_classifier_result(self):
        result = hybrid_route(
            "This issue is hard to categorize",
            predict_intent=lambda _text: {"intent": "general", "confidence": 0.45},
            invoke_llm=lambda _messages: '{"route":"tools"}',
        )

        self.assertEqual(result["route"], "tools")
        self.assertEqual(result["source"], "llm")
        self.assertTrue(result["fallback"])

    def test_hybrid_falls_back_safely_when_llm_json_is_invalid(self):
        result = hybrid_route(
            "This issue is hard to categorize",
            predict_intent=lambda _text: {"intent": "general", "confidence": 0.45},
            invoke_llm=lambda _messages: "not json",
        )

        self.assertEqual(result["route"], "support_specialist")
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["failure_mode"], "invalid_llm_response")


if __name__ == "__main__":
    unittest.main()
