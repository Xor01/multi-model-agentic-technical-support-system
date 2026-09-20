from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any, Optional
import uuid

from fastapi import FastAPI, Header, HTTPException

from support_agent.observability.tracing import invoke_with_langfuse
from support_agent.schemas.chat import ChatCompletionRequest


MODEL_ID = "tuwaiq-tech-support-agent"
AgentInvoker = Callable[..., dict[str, Any]]


def create_app(agent_invoker: AgentInvoker = invoke_with_langfuse) -> FastAPI:
    app = FastAPI(title="Tuwaiq Technical Support Agent")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "student"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(
        request: ChatCompletionRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict[str, object]:
        del authorization  # Accepted for OpenAI-compatible clients; not enforced in this lab.
        if request.stream:
            raise HTTPException(400, "Streaming is not required for this challenge")
        user_messages = [message.content for message in request.messages if message.role == "user"]
        if not user_messages:
            raise HTTPException(400, "No user message provided")

        started = time.perf_counter()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        result = agent_invoker(user_messages[-1], trace_id=request_id)
        answer = result.get("answer", "Unable to produce an answer.")
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "system_metadata": {
                "route": result.get("route"),
                "intent": result.get("intent"),
                "latency_seconds": round(time.perf_counter() - started, 4),
            },
        }

    return app


app = create_app()
