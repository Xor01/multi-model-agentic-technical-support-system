import os
from pathlib import Path
import sqlite3

from langchain_core.tools import tool


DB_PATH = Path(os.getenv("SUPPORT_DB_PATH", "data/mock_support.db"))


def connect_database() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL,
            resolution TEXT
        )
        """
    )
    connection.commit()
    return connection


@tool
def ticket_search(query: str) -> dict:
    """Search previous support tickets in the local support database."""
    connection = connect_database()
    try:
        rows = connection.execute(
            """
            SELECT id, title, status, priority, resolution
            FROM tickets
            WHERE title LIKE ? OR description LIKE ?
            ORDER BY id DESC
            LIMIT 5
            """,
            (f"%{query}%", f"%{query}%"),
        ).fetchall()
        tickets = [dict(row) for row in rows]
        return {"ok": True, "query": query, "tickets": tickets, "count": len(tickets)}
    finally:
        connection.close()


@tool
def ticket_create(title: str, description: str, priority: str = "medium") -> dict:
    """Create a structured support ticket in the local support database."""
    if priority not in {"low", "medium", "high", "critical"}:
        return {"ok": False, "error": "Unsupported priority"}
    connection = connect_database()
    try:
        cursor = connection.execute(
            "INSERT INTO tickets(title, description, priority, status) VALUES (?, ?, ?, 'open')",
            (title, description, priority),
        )
        connection.commit()
        return {
            "ok": True,
            "ticket_id": cursor.lastrowid,
            "status": "open",
            "priority": priority,
        }
    finally:
        connection.close()
