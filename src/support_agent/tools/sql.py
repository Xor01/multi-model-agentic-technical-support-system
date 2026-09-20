import re

from langchain_core.tools import tool

from support_agent.tools.tickets import connect_database


ALLOWED_TABLES = {"tickets"}


@tool
def sql_query(query: str) -> dict:
    """Execute one allowlisted read-only SELECT query against support data."""
    normalized = query.strip()
    lowered = normalized.lower()
    if not lowered.startswith("select ") or ";" in normalized:
        return {"ok": False, "error": "Only one read-only SELECT statement is allowed"}
    referenced_tables = set(re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", lowered))
    if not referenced_tables or not referenced_tables <= ALLOWED_TABLES:
        return {"ok": False, "error": "Query references a non-allowlisted table"}
    connection = connect_database()
    try:
        cursor = connection.execute(normalized)
        rows = [dict(row) for row in cursor.fetchmany(100)]
        return {"ok": True, "columns": [item[0] for item in cursor.description], "rows": rows, "count": len(rows)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        connection.close()
