from __future__ import annotations

from collections.abc import Callable
import os
from time import perf_counter

from dotenv import load_dotenv

from support_agent.routing.classifier_router import (
    CLASSIFIER_CONFIDENCE_THRESHOLD,
    INTENT_TO_ROUTE,
    classifier_route,
    deterministic_route,
)
from support_agent.routing.llm_router import (
    RouterResponseError,
    invoke_openai_router,
    llm_route,
)
from support_agent.routing.rules import rule_first


load_dotenv()


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
            except Exception as exc:
                result = {
                    "route": "support_specialist",
                    "source": "fallback",
                    "confidence": confidence,
                    "fallback": True,
                    "failure_mode": (
                        "invalid_llm_response"
                        if isinstance(exc, RouterResponseError)
                        else "llm_request_failed"
                    ),
                    "error_type": type(exc).__name__,
                }
    result["latency_ms"] = (perf_counter() - started) * 1000
    return result


def _openai_is_configured() -> bool:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key and key not in {"local-demo-key", "replace_me"})


def configured_hybrid_route(
    text: str,
    predict_intent: Callable[[str], dict[str, object]] = classifier_route,
    invoke_llm: Callable[[list[dict[str, str]]], str] | None = None,
    confidence_threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD,
) -> dict[str, object]:
    """Use GPT-5 Mini for ambiguity and preserve deterministic availability."""
    started = perf_counter()
    hard_rule = rule_first(text)
    if hard_rule:
        return {
            **hard_rule,
            "fallback": False,
            "latency_ms": (perf_counter() - started) * 1000,
        }

    resolved_llm = invoke_llm
    if resolved_llm is None and _openai_is_configured():
        resolved_llm = invoke_openai_router

    try:
        prediction = predict_intent(text)
    except Exception as classifier_error:
        deterministic = deterministic_route(text)
        if resolved_llm is None:
            return {
                **deterministic,
                "classifier_failure": type(classifier_error).__name__,
                "failure_mode": "llm_not_configured",
                "latency_ms": (perf_counter() - started) * 1000,
            }
        try:
            return {
                **llm_route(text, resolved_llm),
                "confidence": 0.0,
                "fallback": True,
                "classifier_failure": type(classifier_error).__name__,
                "latency_ms": (perf_counter() - started) * 1000,
            }
        except Exception as llm_error:
            return {
                **deterministic,
                "classifier_failure": type(classifier_error).__name__,
                "failure_mode": "llm_request_failed",
                "error_type": type(llm_error).__name__,
                "latency_ms": (perf_counter() - started) * 1000,
            }

    return hybrid_route(
        text,
        predict_intent=lambda _text: prediction,
        invoke_llm=resolved_llm,
        confidence_threshold=confidence_threshold,
    )


def openai_router_runtime_status() -> dict[str, object]:
    return {
        "configured": _openai_is_configured(),
        "model": os.getenv("OPENAI_ROUTER_MODEL", "gpt-5-mini"),
    }
