from langchain_core.tools import tool

from support_agent.tools.health import system_health_check
from support_agent.tools.tickets import ticket_search


@tool
def diagnostic_runbook(issue_type: str, service: str, symptom: str) -> dict:
    """Run a deterministic health and prior-ticket diagnostic sequence."""
    steps = []
    health = system_health_check.invoke({"service": service})
    steps.append({"step": "health_check", "result": health})
    prior = ticket_search.invoke({"query": symptom})
    steps.append({"step": "prior_tickets", "result": prior})
    status = "needs_human" if health.get("status") == "unknown" else "diagnostics_complete"
    return {
        "status": status,
        "issue_type": issue_type,
        "service": service,
        "symptom": symptom,
        "steps": steps,
    }
