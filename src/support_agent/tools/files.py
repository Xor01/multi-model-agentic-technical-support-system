import os
from pathlib import Path

from langchain_core.tools import tool


UPLOAD_ROOT = Path(os.getenv("SUPPORT_UPLOAD_ROOT", "data/uploads"))
ALLOWED_SUFFIXES = {".conf", ".ini", ".json", ".log", ".txt", ".yaml", ".yml"}


@tool
def file_search(query: str, path: str = "data/uploads") -> dict:
    """Search approved uploaded logs and configuration files inside the upload root."""
    root = UPLOAD_ROOT.resolve()
    requested = root if path == "data/uploads" else (root / path).resolve()
    if requested != root and root not in requested.parents:
        return {"ok": False, "error": "Path is outside the approved upload directory", "matches": []}
    if not requested.exists():
        return {"ok": True, "query": query, "path": str(requested), "matches": []}
    matches = []
    files = [requested] if requested.is_file() else requested.rglob("*")
    for candidate in files:
        if not candidate.is_file() or candidate.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        if candidate.stat().st_size > 1_000_000:
            continue
        for line_number, line in enumerate(candidate.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if query.lower() in line.lower():
                matches.append({"file": str(candidate.relative_to(root)), "line": line_number, "text": line[:500]})
                if len(matches) == 20:
                    return {"ok": True, "query": query, "path": str(requested), "matches": matches}
    return {"ok": True, "query": query, "path": str(requested), "matches": matches}
