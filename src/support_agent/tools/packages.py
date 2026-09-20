from langchain_core.tools import tool


PACKAGE_REGISTRY = {
    "transformers": {"tested_version": "4.57", "python": ">=3.10", "notes": "Used by Models A, B, and C."},
    "torch": {"tested_version": "2.8", "python": ">=3.10", "notes": "GPU support depends on the installed CUDA build."},
    "langchain": {"tested_version": "1.4", "python": ">=3.10", "notes": "Provides the tool contracts."},
}


@tool
def package_lookup(package_name: str, version: str = "") -> dict:
    """Look up package and version facts in the deterministic lab registry."""
    normalized = package_name.strip().lower()
    details = PACKAGE_REGISTRY.get(normalized)
    return {
        "ok": True,
        "found": details is not None,
        "package": normalized,
        "requested_version": version or None,
        "details": details,
        "backend": "local_registry",
    }
