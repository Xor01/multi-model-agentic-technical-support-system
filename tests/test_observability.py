import unittest

from support_agent.observability.tracing import invoke_with_langfuse


class FakeGraph:
    def __init__(self, result=None, error=None):
        self.result = result or {"answer": "done"}
        self.error = error
        self.state = None
        self.config = None

    def invoke(self, state, config):
        self.state = state
        self.config = config
        if self.error:
            raise self.error
        return self.result


class LangfuseTracingTests(unittest.TestCase):
    def test_invocation_passes_callback_metadata_and_trace_id(self):
        fake_graph = FakeGraph({"answer": "traced"})
        fake_callback = object()

        result = invoke_with_langfuse(
            "The API returns 503 after deployment.",
            compiled_graph=fake_graph,
            callback=fake_callback,
            student="Student Name",
            trace_id="trace-123",
        )

        self.assertEqual(result, {"answer": "traced"})
        self.assertEqual(
            fake_graph.state,
            {"user_message": "The API returns 503 after deployment.", "trace_id": "trace-123"},
        )
        self.assertEqual(fake_graph.config["callbacks"], [fake_callback])
        self.assertEqual(
            fake_graph.config["metadata"],
            {
                "project": "tuwaiq-weekend-support-agent",
                "student": "Student Name",
                "trace_id": "trace-123",
            },
        )

    def test_custom_metadata_is_added_without_removing_required_fields(self):
        fake_graph = FakeGraph()

        invoke_with_langfuse(
            "Check service health",
            compiled_graph=fake_graph,
            callback=object(),
            student="Student Name",
            metadata={"environment": "lab"},
        )

        self.assertEqual(fake_graph.config["metadata"]["project"], "tuwaiq-weekend-support-agent")
        self.assertEqual(fake_graph.config["metadata"]["student"], "Student Name")
        self.assertEqual(fake_graph.config["metadata"]["environment"], "lab")
        self.assertEqual(fake_graph.state["trace_id"], fake_graph.config["metadata"]["trace_id"])

    def test_graph_errors_are_not_hidden(self):
        fake_graph = FakeGraph(error=RuntimeError("node failed"))

        with self.assertRaisesRegex(RuntimeError, "node failed"):
            invoke_with_langfuse(
                "Trigger failure",
                compiled_graph=fake_graph,
                callback=object(),
                student="Student Name",
            )


if __name__ == "__main__":
    unittest.main()
