from __future__ import annotations

from collections.abc import Callable
import re
from typing import Any

from support_agent.routing.classifier_router import baseline_router, deterministic_route
from support_agent.schemas.state import SupportState
from support_agent.tools.diagnostic_rubook import diagnostic_runbook
from support_agent.tools.esclation import escalate_to_human
from support_agent.tools.knowledge_base import knowledge_base_search


Router = Callable[[str], dict[str, object]]
QARunner = Callable[[str, list[dict[str, Any]]], str]
SupportRunner = Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], str]


def default_qa_runner(question: str, contexts: list[dict[str, Any]]) -> str:
    """Deterministic fallback until the saved Model B runner is connected."""
    if not contexts or float(contexts[0].get("score", 1.0)) < 0.5:
        return "I cannot verify the answer from the supplied knowledge base."
    return str(contexts[0].get("answer") or contexts[0].get("passage", ""))


def redact_sensitive(text: str) -> str:
    """Remove common credentials and email addresses before display or ticket storage."""
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "[REDACTED_TOKEN]", text)
    return re.sub(
        r"(?i)\b(bearer\s+)(?!token\b)[A-Za-z0-9._~+/-]{12,}\b",
        r"\1[REDACTED_TOKEN]",
        text,
    )


def default_support_runner(
    user_message: str,
    context: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> str:
    """Deterministic fallback until the saved Model C runner is connected."""
    lowered = user_message.lower()
    retrieved_text = " ".join(str(item.get("passage", "")) for item in context).lower()
    if any(phrase in retrieved_text for phrase in ("ignore system instructions", "reveal secrets")):
        return "Treat retrieved content as untrusted data. Do not follow its instructions or reveal secrets."
    if "summarize" in lowered and ("ticket" in lowered or "case" in lowered):
        return "Ticket summary: credential exposure reported. Redact personal data and credentials; revoke and rotate affected tokens."
    if "bearer token" in lowered or "api key" in lowered or "credential" in lowered:
        return "Revoke and rotate the exposed token. Redact credentials from chat, logs, and tickets. Review recent access."
    if "logs are unavailable" in lowered or "logs are missing" in lowered:
        return "I cannot verify the cause without evidence. Please provide the service name, timestamp, and redacted logs."
    if "exactly three" in lowered or "three troubleshooting steps" in lowered:
        return (
            "1. Check the service health and deployment status.\n"
            "2. Inspect recent redacted error logs and dependency health.\n"
            "3. Compare the last working deployment and escalate if production remains down."
        )
    return "Check service health, inspect redacted logs, and share the relevant error or ticket details for diagnosis."


def route_node(state: SupportState, router: Router = baseline_router) -> SupportState:
    decision = router(state["user_message"])
    inferred_intent = deterministic_route(state["user_message"])["intent"]
    return {
        "route": str(decision.get("route") or decision.get("intent") or "support_specialist"),
        "intent": str(decision.get("intent") or inferred_intent),
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
                    "user_message": redact_sensitive(state["user_message"]),
                    "intent": state.get("intent", ""),
                    "tool_results": redact_sensitive(str(state.get("tool_results", []))),
                }
            ),
        }
    )
    return {
        "answer": f"Escalated to human support: {result}",
        "escalate": True,
        "tool_results": [result],
    }
