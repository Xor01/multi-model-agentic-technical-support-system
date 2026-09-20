from __future__ import annotations

from collections.abc import Callable
from typing import Any

from support_agent.routing.classifier_router import baseline_router
from support_agent.schemas.state import SupportState
from support_agent.tools.diagnostic_rubook import diagnostic_runbook
from support_agent.tools.esclation import escalate_to_human
from support_agent.tools.knowledge_base import knowledge_base_search


Router = Callable[[str], dict[str, object]]
QARunner = Callable[[str, list[dict[str, Any]]], str]
SupportRunner = Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], str]


def default_qa_runner(question: str, contexts: list[dict[str, Any]]) -> str:
    """Deterministic fallback until the saved Model B runner is connected."""
    if not contexts:
        return "I cannot verify the answer from the supplied knowledge base."
    return str(contexts[0].get("answer") or contexts[0].get("passage", ""))


def default_support_runner(
    user_message: str,
    context: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> str:
    """Deterministic fallback until the saved Model C runner is connected."""
    return (
        f"Support analysis for: {user_message} "
        f"(context passages: {len(context)}, tool results: {len(tool_results)})"
    )


def route_node(state: SupportState, router: Router = baseline_router) -> SupportState:
    decision = router(state["user_message"])
    return {
        "route": str(decision.get("route") or decision.get("intent") or "support_specialist"),
        "intent": str(decision.get("intent", "")),
        "confidence": float(decision.get("confidence", 0.0)),
    }


def qa_node(state: SupportState, qa_runner: QARunner = default_qa_runner) -> SupportState:
    kb = knowledge_base_search.invoke({"query": state["user_message"]})
    contexts = kb.get("results", [])
    answer = qa_runner(state["user_message"], contexts)
    return {"context": contexts, "answer": answer}


def infer_service(message: str) -> str:
    lowered = message.lower()
    if any(word in lowered for word in ("database", "sql", "query")):
        return "database"
    if any(word in lowered for word in ("gpu", "cuda", "memory")):
        return "gpu-worker"
    if any(word in lowered for word in ("api", "http", "endpoint", "service")):
        return "api"
    return "unknown"


def tool_node(state: SupportState) -> SupportState:
    result = diagnostic_runbook.invoke(
        {
            "issue_type": state.get("intent", "unknown"),
            "service": infer_service(state["user_message"]),
            "symptom": state["user_message"],
        }
    )
    return {"tool_results": [result]}


def support_node(
    state: SupportState, support_runner: SupportRunner = default_support_runner
) -> SupportState:
    answer = support_runner(
        state["user_message"],
        state.get("context", []),
        state.get("tool_results", []),
    )
    return {"answer": answer}


def escalation_node(state: SupportState) -> SupportState:
    result = escalate_to_human.invoke(
        {
            "reason": "Policy or unresolved technical incident",
            "evidence": str(
                {
                    "user_message": state["user_message"],
                    "intent": state.get("intent", ""),
                    "tool_results": state.get("tool_results", []),
                }
            ),
        }
    )
    return {
        "answer": f"Escalated to human support: {result}",
        "escalate": True,
        "tool_results": [result],
    }
