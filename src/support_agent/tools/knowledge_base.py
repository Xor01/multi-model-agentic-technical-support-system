import json
from pathlib import Path
import re

from langchain_core.tools import tool


DATA_PATH = Path(__file__).parents[3] / "data" / "technical_support_qa_100_official_docs.json"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _load_records() -> list[dict]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _rank(query: str, records: list[dict], fields: tuple[str, ...], limit: int) -> list[tuple[float, dict]]:
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    scored = []
    for record in records:
        document_tokens = _tokens(" ".join(str(record.get(field, "")) for field in fields))
        score = len(query_tokens & document_tokens) / len(query_tokens)
        if score:
            scored.append((score, record))
    return sorted(scored, key=lambda item: (-item[0], item[1]["id"]))[:limit]


@tool
def knowledge_base_search(query: str) -> dict:
    """Return top trusted local knowledge-base passages with scores and source IDs."""
    results = [
        {
            "source_id": record["id"],
            "score": round(score, 4),
            "passage": record["context"],
            "answer": record["answer_text"],
            "source_title": record["source_title"],
        }
        for score, record in _rank(query, _load_records(), ("question", "context", "answer_text"), 5)
    ]
    return {"ok": True, "query": query, "backend": "local_token_overlap", "results": results}
