RULES = {
    "escalate": ["data loss", "security breach", "production down", "corruption"],
    "qa": ["according to the docs", "documentation", "what does the manual say"],
}


def rule_first(text: str) -> dict[str, object] | None:
    lowered = text.lower()
    if ("production database" in lowered and ("corrupt" in lowered or "delete" in lowered)):
        return {"route": "escalate", "source": "rule", "confidence": 1.0}
    if "cuda" in lowered and "out of memory" in lowered:
        return {"route": "tools", "source": "rule", "confidence": 1.0}
    if any(phrase in lowered for phrase in ("pasted bearer token", "summarize this support ticket", "retrieved support article", "logs are unavailable")):
        return {"route": "support_specialist", "source": "rule", "confidence": 1.0}
    for route, phrases in RULES.items():
        if any(phrase in lowered for phrase in phrases):
            return {"route": route, "source": "rule", "confidence": 1.0}
    return None
