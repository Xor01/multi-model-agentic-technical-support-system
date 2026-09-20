from typing import Literal

from support_agent.schemas.state import SupportState


def choose_after_router(
    state: SupportState,
) -> Literal["qa", "tools", "support", "escalate"]:
    route = state.get("route", "")
    if route == "escalate":
        return "escalate"
    if route == "qa":
        return "qa"
    if route == "tools" or route in {"database", "gpu", "deployment", "network"}:
        return "tools"
    return "support"
