import re

from langchain_core.tools import tool


@tool
def log_analyzer(log_text: str) -> dict:
    """Extract common ERROR/WARN patterns from support logs."""
    lines = log_text.splitlines()
    errors = [line for line in lines if re.search(r"\b(ERROR|FATAL|EXCEPTION)\b", line, re.I)]
    warnings = [line for line in lines if re.search(r"\bWARN(ING)?\b", line, re.I)]
    return {
        "errors": errors[:20],
        "warnings": warnings[:20],
        "error_count": len(errors),
        "warning_count": len(warnings),
    }
