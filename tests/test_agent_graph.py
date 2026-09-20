import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support_agent.agent.conditions import choose_after_router
from support_agent.agent.graph import build_support_graph
from support_agent.agent.nodes import qa_node, route_node


class SupportAgentGraphTests(unittest.TestCase):
    def test_route_node_and_condition_use_router_decision(self):
        state = route_node(
            {"user_message": "Check the docs"},
            router=lambda _text: {"route": "qa", "intent": "api", "confidence": 0.91},
        )

        self.assertEqual(state, {"route": "qa", "intent": "api", "confidence": 0.91})
        self.assertEqual(choose_after_router(state), "qa")

    def test_qa_path_retrieves_context_before_answering(self):
        result = qa_node(
            {"user_message": "Which header carries a bearer token?"},
            qa_runner=lambda _question, contexts: contexts[0]["answer"],
        )

        self.assertTrue(result["context"])
        self.assertEqual(result["answer"], "Authorization")

    def test_compiled_graph_runs_qa_path(self):
        graph = build_support_graph(
            router=lambda _text: {"route": "qa", "intent": "authentication", "confidence": 1.0},
            qa_runner=lambda _question, contexts: contexts[0]["answer"],
        )

        result = graph.invoke({"user_message": "Which header carries a bearer token?"})

        self.assertEqual(result["route"], "qa")
        self.assertEqual(result["answer"], "Authorization")

    def test_compiled_graph_runs_tools_then_support(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("support_agent.tools.tickets.DB_PATH", Path(directory) / "support.db"):
                graph = build_support_graph(
                    router=lambda _text: {"route": "tools", "intent": "database", "confidence": 0.82},
                    support_runner=lambda _message, _context, tool_results: (
                        f"Used {len(tool_results)} diagnostic result"
                    ),
                )
                result = graph.invoke({"user_message": "Database queries are timing out"})

        self.assertEqual(result["route"], "tools")
        self.assertEqual(len(result["tool_results"]), 1)
        self.assertEqual(result["answer"], "Used 1 diagnostic result")

    def test_default_graph_survives_unavailable_classifier(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("support_agent.tools.tickets.DB_PATH", Path(directory) / "support.db"):
                graph = build_support_graph()
                result = graph.invoke(
                    {"user_message": "The API returns 503 after deployment."}
                )

        self.assertEqual(result["route"], "deployment")
        self.assertEqual(result["intent"], "deployment")
        self.assertEqual(len(result["tool_results"]), 1)
        self.assertIn("Support analysis", result["answer"])

    def test_compiled_graph_runs_escalation_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("support_agent.tools.tickets.DB_PATH", Path(directory) / "support.db"):
                graph = build_support_graph(
                    router=lambda _text: {"route": "escalate", "confidence": 1.0}
                )
                result = graph.invoke({"user_message": "Production down with possible data loss"})

        self.assertTrue(result["escalate"])
        self.assertIn("Escalated to human support", result["answer"])


if __name__ == "__main__":
    unittest.main()
