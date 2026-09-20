import json
import unittest

from fastapi.testclient import TestClient

from support_agent.api.main import MODEL_ID, create_app


class FakeAgentInvoker:
    def __init__(self):
        self.calls = []

    def __call__(self, user_message, *, trace_id):
        self.calls.append({"user_message": user_message, "trace_id": trace_id})
        return {
            "answer": "Check the deployment logs.",
            "route": "support_specialist",
            "intent": "deployment",
        }


class OpenAICompatibleApiTests(unittest.TestCase):
    def setUp(self):
        self.invoker = FakeAgentInvoker()
        self.client = TestClient(create_app(agent_invoker=self.invoker))

    def test_health_and_model_discovery(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], MODEL_ID)

    def test_readiness_reports_runtime_mode_without_loading_the_model(self):
        readiness = {
            "status": "ready",
            "mode": "deterministic_fallback",
            "components": {
                "api": {"ready": True},
                "intent_classifier": {
                    "ready": False,
                    "reason": "missing_config",
                },
            },
        }
        client = TestClient(
            create_app(
                agent_invoker=self.invoker,
                readiness_provider=lambda: readiness,
            )
        )

        response = client.get("/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), readiness)

    def test_chat_completion_uses_latest_user_message_and_openai_shape(self):
        response = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer local-test"},
            json={
                "model": MODEL_ID,
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Old question"},
                    {"role": "assistant", "content": "Old answer"},
                    {"role": "user", "content": "The API returns 503 after deployment."},
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["id"].startswith("chatcmpl-"))
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], MODEL_ID)
        self.assertEqual(payload["choices"][0]["message"], {"role": "assistant", "content": "Check the deployment logs."})
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payload["usage"], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        self.assertEqual(payload["system_metadata"]["route"], "support_specialist")
        self.assertEqual(payload["system_metadata"]["intent"], "deployment")
        self.assertGreaterEqual(payload["system_metadata"]["latency_seconds"], 0)
        self.assertEqual(self.invoker.calls[0]["user_message"], "The API returns 503 after deployment.")
        self.assertEqual(self.invoker.calls[0]["trace_id"], payload["id"])

    def test_streaming_returns_openai_compatible_sse_events(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = [line.removeprefix("data: ") for line in response.text.splitlines() if line]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(event) for event in events[:-1]]
        self.assertEqual(chunks[0]["object"], "chat.completion.chunk")
        self.assertEqual(
            chunks[0]["choices"][0]["delta"],
            {"role": "assistant", "content": "Check the deployment logs."},
        )
        self.assertIsNone(chunks[0]["choices"][0]["finish_reason"])
        self.assertEqual(chunks[1]["choices"][0]["delta"], {})
        self.assertEqual(chunks[1]["choices"][0]["finish_reason"], "stop")

    def test_request_requires_a_user_message(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "system", "content": "hello"}]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "No user message provided")


if __name__ == "__main__":
    unittest.main()
