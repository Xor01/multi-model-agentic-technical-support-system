from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from support_agent.routing.classifier_router import (
    CLASSIFIER_CONFIDENCE_THRESHOLD,
    INTENT_TO_ROUTE,
    classifier_route,
)
from support_agent.routing.llm_router import RouterResponseError, llm_route
from support_agent.routing.rules import rule_first


def hybrid_route(
    text: str,
    predict_intent: Callable[[str], dict[str, object]] = classifier_route,
    invoke_llm: Callable[[list[dict[str, str]]], str] | None = None,
    confidence_threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD,
) -> dict[str, object]:
    """Apply safety rules, then classifier, then the LLM only when ambiguous."""
    started = perf_counter()
    hard_rule = rule_first(text)
    if hard_rule:
        result: dict[str, object] = {**hard_rule, "fallback": False}
    else:
        prediction = predict_intent(text)
        confidence = float(prediction["confidence"])
        if confidence >= confidence_threshold:
            result = {
                **prediction,
                "route": INTENT_TO_ROUTE.get(
                    str(prediction["intent"]), "support_specialist"
                ),
                "fallback": False,
            }
        elif invoke_llm is None:
            result = {
                "route": "support_specialist",
                "source": "fallback",
                "confidence": confidence,
                "fallback": True,
                "failure_mode": "llm_not_configured",
            }
        else:
            try:
                result = {
                    **llm_route(text, invoke_llm),
                    "confidence": confidence,
                    "fallback": True,
                }
            except RouterResponseError:
                result = {
                    "route": "support_specialist",
                    "source": "fallback",
                    "confidence": confidence,
                    "fallback": True,
                    "failure_mode": "invalid_llm_response",
                }
    result["latency_ms"] = (perf_counter() - started) * 1000
    return result
