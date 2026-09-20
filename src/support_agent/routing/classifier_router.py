from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from support_agent.routing.rules import rule_first


CLASSIFIER_CONFIDENCE_THRESHOLD = 0.75
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "intent_classifier"


INTENT_TO_ROUTE = {
    "authentication": "support_specialist",
    "network": "support_specialist",
    "deployment": "support_specialist",
    "database": "support_specialist",
    "gpu": "support_specialist",
    "api": "support_specialist",
    "package": "support_specialist",
    "general": "support_specialist",
}

_router_tokenizer = None
_router_model = None


DETERMINISTIC_INTENTS = {
    "database": ("database", "sql", "query", "postgres"),
    "gpu": ("gpu", "cuda", "vram"),
    "deployment": ("deploy", "deployment", "503", "container", "docker"),
    "network": ("network", "dns", "timeout", "connection"),
    "package": ("package", "dependency", "version", "pip"),
    "api": ("api", "http", "endpoint"),
}


def classifier_runtime_status() -> dict[str, object]:
    """Report whether the local classifier can be loaded without loading it."""
    missing_dependencies = [
        name for name in ("torch", "transformers") if find_spec(name) is None
    ]
    if missing_dependencies:
        return {
            "ready": False,
            "reason": f"missing_dependencies: {', '.join(missing_dependencies)}",
        }

    config_path = MODEL_PATH / "config.json"
    weights_path = MODEL_PATH / "model.safetensors"
    if not config_path.is_file():
        return {"ready": False, "reason": "missing_config"}
    if not weights_path.is_file():
        return {"ready": False, "reason": "missing_weights"}
    if weights_path.read_bytes()[:40].startswith(b"version https://git-lfs.github.com"):
        return {"ready": False, "reason": "weights_are_git_lfs_pointer"}
    return {"ready": True, "reason": "artifacts_available"}


def _load_classifier() -> tuple[Any, Any]:
    global _router_model, _router_tokenizer
    if _router_model is None or _router_tokenizer is None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required to run the classifier router") from exc
        _router_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        _router_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
        _router_model.eval()
    return _router_tokenizer, _router_model


def classifier_route(text: str) -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to run the classifier router") from exc

    tokenizer, model = _load_classifier()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        index = int(torch.argmax(probabilities))
    return {
        "intent": model.config.id2label[index],
        "confidence": float(probabilities[index]),
        "source": "classifier",
    }


def _apply_classifier_policy(
    prediction: dict[str, object], threshold: float
) -> dict[str, object]:
    intent = str(prediction["intent"])
    confidence = float(prediction["confidence"])
    return {
        **prediction,
        "route": INTENT_TO_ROUTE.get(intent, "support_specialist"),
        "fallback": confidence < threshold,
    }


def baseline_router(
    text: str,
    predict_intent: Callable[[str], dict[str, object]] = classifier_route,
    confidence_threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD,
) -> dict[str, object]:
    hard_rule = rule_first(text)
    if hard_rule:
        return hard_rule
    return _apply_classifier_policy(predict_intent(text), confidence_threshold)


def deterministic_route(text: str) -> dict[str, object]:
    """Provide an offline route when trained classifier artifacts cannot load."""
    lowered = text.lower()
    for intent, keywords in DETERMINISTIC_INTENTS.items():
        if any(keyword in lowered for keyword in keywords):
            return {
                "route": intent,
                "intent": intent,
                "confidence": 0.5,
                "source": "deterministic_fallback",
                "fallback": True,
            }
    return {
        "route": "support_specialist",
        "intent": "general",
        "confidence": 0.0,
        "source": "deterministic_fallback",
        "fallback": True,
    }


def resilient_router(
    text: str,
    predict_intent: Callable[[str], dict[str, object]] = classifier_route,
    confidence_threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD,
) -> dict[str, object]:
    """Use normal routing when possible and remain operational when it is not."""
    try:
        return baseline_router(text, predict_intent, confidence_threshold)
    except Exception as exc:
        return {
            **deterministic_route(text),
            "failure_mode": "classifier_unavailable",
            "error_type": type(exc).__name__,
        }
