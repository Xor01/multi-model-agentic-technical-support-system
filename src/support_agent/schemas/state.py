from typing import Any, TypedDict


class SupportState(TypedDict, total=False):
    user_message: str
    route: str
    intent: str
    confidence: float
    context: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    answer: str
    escalate: bool
    trace_id: str
