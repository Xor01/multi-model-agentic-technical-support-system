RULES = {
    "escalate": ["data loss", "security breach", "production down", "corruption"],
    "qa": ["according to the docs", "documentation", "what does the manual say"],
}


def rule_first(text: str) -> dict[str, object] | None:
    lowered = text.lower()
    for route, phrases in RULES.items():
        if any(phrase in lowered for phrase in phrases):
            return {"route": route, "source": "rule", "confidence": 1.0}
    return None
