from __future__ import annotations

import os
from typing import Any
from uuid import uuid4


PROJECT_NAME = "tuwaiq-weekend-support-agent"


def invoke_with_langfuse(
    user_message: str,
    *,
    compiled_graph: Any = None,
    callback: Any = None,
    student: str | None = None,
    metadata: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Invoke the agent with Langfuse tracing across all LangGraph nodes.

    LangGraph forwards the callback through router, specialist, and tool nodes.
    Langfuse records their inputs, outputs, timings, and raised errors.
    """
    if compiled_graph is None:
        from support_agent.agent.graph import graph

        compiled_graph = graph
    if callback is None:
        from support_agent.observability.langfuse import langfuse_handler

        callback = langfuse_handler

    resolved_trace_id = trace_id or str(uuid4())
    resolved_student = (
        student
        or os.getenv("TUWAIQ_STUDENT_NAME", "N/A")
    )
    trace_metadata = {
        **(metadata or {}),
        "project": PROJECT_NAME,
        "student": resolved_student,
        "trace_id": resolved_trace_id,
    }
    return compiled_graph.invoke(
        {"user_message": user_message, "trace_id": resolved_trace_id},
        config={"callbacks": [callback], "metadata": trace_metadata},
    )
