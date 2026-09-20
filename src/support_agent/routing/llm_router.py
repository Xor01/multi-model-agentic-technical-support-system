from __future__ import annotations

from collections.abc import Callable
import json


ALLOWED_ROUTES = {"qa", "support_specialist", "tools", "escalate"}

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
