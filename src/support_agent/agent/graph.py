from langgraph.graph import END, START, StateGraph

from support_agent.agent.conditions import choose_after_router
from support_agent.agent.nodes import (
    QARunner,
    Router,
    SupportRunner,
    default_qa_runner,
    default_support_runner,
    escalation_node,
    qa_node,
    route_node,
    support_node,
    tool_node,
)
from support_agent.routing.hybrid_router import configured_hybrid_route
from support_agent.schemas.state import SupportState


def build_support_graph(
    router: Router = configured_hybrid_route,
    qa_runner: QARunner = default_qa_runner,
    support_runner: SupportRunner = default_support_runner,
):
    builder = StateGraph(SupportState)
    builder.add_node("route", lambda state: route_node(state, router=router))
    builder.add_node("qa", lambda state: qa_node(state, qa_runner=qa_runner))
    builder.add_node("tools", tool_node)
    builder.add_node(
        "support", lambda state: support_node(state, support_runner=support_runner)
    )
    builder.add_node("escalate", escalation_node)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        choose_after_router,
        {"qa": "qa", "tools": "tools", "support": "support", "escalate": "escalate"},
    )
    builder.add_edge("qa", END)
    builder.add_edge("tools", "support")
    builder.add_edge("support", END)
    builder.add_edge("escalate", END)
    return builder.compile()


graph = build_support_graph()
