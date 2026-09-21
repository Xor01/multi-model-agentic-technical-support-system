from __future__ import annotations

from collections.abc import Callable
import json
import time
from typing import Any, Optional
import uuid

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from support_agent.observability.tracing import invoke_with_langfuse
from support_agent.routing.classifier_router import classifier_runtime_status
from support_agent.routing.hybrid_router import openai_router_runtime_status
from support_agent.schemas.chat import ChatCompletionRequest


MODEL_ID = "tuwaiq-tech-support-agent"
AgentInvoker = Callable[..., dict[str, Any]]
ReadinessProvider = Callable[[], dict[str, Any]]


def runtime_readiness() -> dict[str, Any]:
    classifier = classifier_runtime_status()
    llm_router = openai_router_runtime_status()
    return {
        "status": "ready",
        "mode": (
            "classifier"
            if classifier["ready"]
            else "gpt_router"
            if llm_router["configured"]
            else "deterministic_fallback"
        ),
        "components": {
            "api": {"ready": True},
            "intent_classifier": classifier,
            "llm_router": llm_router,
        },
    }


def create_app(
    agent_invoker: AgentInvoker = invoke_with_langfuse,
    readiness_provider: ReadinessProvider = runtime_readiness,
) -> FastAPI:
    app = FastAPI(title="Tuwaiq Technical Support Agent")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def readiness() -> dict[str, Any]:
        return readiness_provider()

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
    ) -> Any:
        del authorization  # Accepted for OpenAI-compatible clients; not enforced in this lab.
        user_messages = [message.content for message in request.messages if message.role == "user"]
        if not user_messages:
            raise HTTPException(400, "No user message provided")

        started = time.perf_counter()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        result = agent_invoker(user_messages[-1], trace_id=request_id)
        answer = str(result.get("answer", "Unable to produce an answer."))
        created = int(time.time())

        if request.stream:
            def stream_events():
                chunks = (
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": answer},
                            "finish_reason": None,
                        }],
                    },
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    },
                )
                for chunk in chunks:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
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
