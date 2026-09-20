import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support_agent.tools import ALL_TOOLS
from support_agent.tools.calculator import calculator
from support_agent.tools.diagnostic_rubook import diagnostic_runbook
from support_agent.tools.documentation import documentation_search
from support_agent.tools.esclation import escalate_to_human
from support_agent.tools.files import file_search
from support_agent.tools.health import system_health_check
from support_agent.tools.knowledge_base import knowledge_base_search
from support_agent.tools.logs import log_analyzer
from support_agent.tools.packages import package_lookup
from support_agent.tools.sql import sql_query
from support_agent.tools.tickets import ticket_create, ticket_search
from support_agent.tools.web import web_search


EXPECTED_TOOLS = {
    "knowledge_base_search",
    "ticket_search",
    "ticket_create",
    "system_health_check",
    "log_analyzer",
    "documentation_search",
    "package_lookup",
    "sql_query",
    "calculator",
    "file_search",
    "web_search",
    "escalate_to_human",
    "diagnostic_runbook",
}


class ToolContractTests(unittest.TestCase):
    def test_all_thirteen_tool_contracts_are_exposed(self):
        self.assertEqual(set(ALL_TOOLS), EXPECTED_TOOLS)
        self.assertTrue(all(tool.args_schema is not None for tool in ALL_TOOLS.values()))

    def test_calculator_accepts_arithmetic_and_rejects_code(self):
        self.assertEqual(calculator.invoke({"expression": "(2 + 3) * 4"}), {"ok": True, "result": 20})
        self.assertFalse(calculator.invoke({"expression": "__import__('os')"})["ok"])

    def test_log_analyzer_extracts_errors_and_warnings(self):
        result = log_analyzer.invoke(
            {"log_text": "INFO ready\nWARN disk nearly full\nERROR connection failed\nFatal shutdown"}
        )

        self.assertEqual(result["error_count"], 2)
        self.assertEqual(result["warning_count"], 1)
        self.assertEqual(result["errors"][0], "ERROR connection failed")

    def test_ticket_create_search_escalation_and_runbook_use_local_database(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "support.db"
            with patch("support_agent.tools.tickets.DB_PATH", database):
                created = ticket_create.invoke(
                    {"title": "Database timeout", "description": "Queries exceed 30s", "priority": "high"}
                )
                found = ticket_search.invoke({"query": "Database"})
                escalation = escalate_to_human.invoke(
                    {"reason": "possible corruption", "evidence": "checksum mismatch"}
                )
                runbook = diagnostic_runbook.invoke(
                    {"issue_type": "database", "service": "database", "symptom": "timeout"}
                )

        self.assertEqual(created["status"], "open")
        self.assertEqual(found["tickets"][0]["title"], "Database timeout")
        self.assertTrue(escalation["escalated"])
        self.assertEqual(escalation["ticket"]["priority"], "high")
        self.assertEqual(runbook["status"], "diagnostics_complete")
        self.assertEqual([step["step"] for step in runbook["steps"]], ["health_check", "prior_tickets"])

    def test_health_registry_is_deterministic(self):
        self.assertEqual(system_health_check.invoke({"service": "api"})["status"], "healthy")
        self.assertEqual(system_health_check.invoke({"service": "missing"}), {"service": "missing", "status": "unknown"})

    def test_local_knowledge_and_documentation_search_return_structured_hits(self):
        kb = knowledge_base_search.invoke({"query": "bearer token authorization header"})
        docs = documentation_search.invoke({"query": "bearer token authorization header"})

        self.assertTrue(kb["results"])
        self.assertIn("source_id", kb["results"][0])
        self.assertTrue(docs["results"])
        self.assertIn("source_url", docs["results"][0])

    def test_package_and_web_lookups_are_deterministic_mocks(self):
        package = package_lookup.invoke({"package_name": "transformers", "version": "4.57"})
        web = web_search.invoke({"query": "python ssl troubleshooting"})

        self.assertTrue(package["found"])
        self.assertEqual(package["backend"], "local_registry")
        self.assertEqual(web["backend"], "deterministic_mock")

    def test_sql_query_is_read_only_and_allowlisted(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "support.db"
            with patch("support_agent.tools.tickets.DB_PATH", database):
                ticket_create.invoke({"title": "API issue", "description": "500 error"})
                selected = sql_query.invoke({"query": "SELECT id, title, status FROM tickets"})
                rejected = sql_query.invoke({"query": "DELETE FROM tickets"})

        self.assertTrue(selected["ok"])
        self.assertEqual(selected["rows"][0]["title"], "API issue")
        self.assertFalse(rejected["ok"])

    def test_file_search_stays_inside_approved_upload_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_root = Path(directory)
            (upload_root / "app.log").write_text("ERROR database unavailable", encoding="utf-8")
            with patch("support_agent.tools.files.UPLOAD_ROOT", upload_root):
                found = file_search.invoke({"query": "database"})
                blocked = file_search.invoke({"query": "secret", "path": ".."})

        self.assertEqual(found["matches"][0]["file"], "app.log")
        self.assertFalse(blocked["ok"])


if __name__ == "__main__":
    unittest.main()
