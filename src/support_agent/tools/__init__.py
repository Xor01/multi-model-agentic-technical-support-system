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


ALL_TOOLS = {
    tool.name: tool
    for tool in (
        knowledge_base_search,
        ticket_search,
        ticket_create,
        system_health_check,
        log_analyzer,
        documentation_search,
        package_lookup,
        sql_query,
        calculator,
        file_search,
        web_search,
        escalate_to_human,
        diagnostic_runbook,
    )
}

__all__ = ["ALL_TOOLS", *ALL_TOOLS]
