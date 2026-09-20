from langchain_core.tools import tool


HEALTH_REGISTRY = {
    "api": {"status": "healthy", "latency_ms": 42},
    "database": {"status": "degraded", "connections_pct": 91},
    "gpu-worker": {"status": "healthy", "gpu_utilization": 74},
}


@tool
def system_health_check(service: str) -> dict:
    """Return deterministic service health for the lab environment."""
    result = HEALTH_REGISTRY.get(service)
    return {"service": service, **result} if result else {"service": service, "status": "unknown"}
