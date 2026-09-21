from __future__ import annotations

from collections.abc import Callable
import json
import os
from typing import Any


ALLOWED_ROUTES = {"qa", "support_specialist", "tools", "escalate"}
DEFAULT_OPENAI_ROUTER_MODEL = "gpt-5-mini"

ROUTER_SYSTEM = """
You are a routing controller for a technical support system.
Return JSON only with exactly this shape: {"route":"<allowed route>"}.
Allowed routes: qa, support_specialist, tools, escalate.
Use qa for questions answerable from trusted documentation.
Use tools for live/system/ticket/log/package information.
Use support_specialist for troubleshooting synthesis.
Use escalate for high-risk or unresolved production incidents.
""".strip()

FEW_SHOT_MESSAGES = [
    {"role": "user", "content": "Check the currently installed package version."},
    {"role": "assistant", "content": '{"route":"tools"}'},
    {"role": "user", "content": "My service keeps restarting after deployment."},
    {"role": "assistant", "content": '{"route":"support_specialist"}'},
]


class RouterResponseError(ValueError):
    pass


def build_router_prompt(user_message: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        *FEW_SHOT_MESSAGES,
        {"role": "user", "content": user_message},
    ]


def parse_router_response(raw_response: str) -> dict[str, str]:
    try:
        payload = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RouterResponseError("router response must be valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"route"}:
        raise RouterResponseError("router response must contain only the route field")
    if payload["route"] not in ALLOWED_ROUTES:
        raise RouterResponseError("router returned an unsupported route")
    return {"route": payload["route"], "source": "llm"}


def llm_route(
    user_message: str,
    invoke_llm: Callable[[list[dict[str, str]]], str],
) -> dict[str, str]:
    """Route with the existing small instruction model via its generation callback."""
    return parse_router_response(invoke_llm(build_router_prompt(user_message)))


def invoke_openai_router(
    messages: list[dict[str, str]],
    *,
    client: Any = None,
    model: str | None = None,
) -> str:
    """Invoke GPT-5 Mini with a strict one-field routing schema."""
    if client is None:
        from openai import OpenAI

        client = OpenAI()

    response = client.chat.completions.create(
        model=model or os.getenv("OPENAI_ROUTER_MODEL", DEFAULT_OPENAI_ROUTER_MODEL),
        messages=messages,
        max_completion_tokens=256,
        reasoning_effort="low",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "support_route",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "route": {
                            "type": "string",
                            "enum": sorted(ALLOWED_ROUTES),
                        }
                    },
                    "required": ["route"],
                    "additionalProperties": False,
                },
            },
        },
    )
    content = response.choices[0].message.content
    if not content:
        raise RouterResponseError("OpenAI router returned an empty response")
    return content
