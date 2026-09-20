from langchain_core.tools import tool

from support_agent.tools.knowledge_base import _load_records, _rank


@tool
def documentation_search(query: str) -> dict:
    """Search the project's local trusted product and API documentation records."""
    results = [
        {
            "source_id": record["id"],
            "score": round(score, 4),
            "title": record["source_title"],
            "section": record["source_section"],
            "source_url": record["source_url"],
            "passage": record["context"],
        }
        for score, record in _rank(query, _load_records(), ("question", "context", "source_title"), 5)
    ]
    return {"ok": True, "query": query, "backend": "local_document_index", "results": results}
