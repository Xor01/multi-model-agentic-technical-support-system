from langchain_core.tools import tool

from support_agent.tools.tickets import ticket_create


@tool
def escalate_to_human(reason: str, evidence: str) -> dict:
    """Escalate a support case with a high-priority ticket and evidence."""
    ticket = ticket_create.invoke(
        {
            "title": f"ESCALATION: {reason}",
            "description": evidence,
            "priority": "high",
        }
    )
    return {"escalated": bool(ticket.get("ok")), "reason": reason, "evidence": evidence, "ticket": ticket}
