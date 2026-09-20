from langchain_core.tools import tool


@tool
def web_search(query: str) -> dict:
    """Return deterministic mock external-search results for the lab environment."""
    return {
        "ok": True,
        "query": query,
        "backend": "deterministic_mock",
        "results": [
            {
                "title": "External technical search is mocked in the lab",
                "url": "https://example.invalid/technical-support",
                "snippet": f"Mock result for: {query}",
            }
        ],
    }
