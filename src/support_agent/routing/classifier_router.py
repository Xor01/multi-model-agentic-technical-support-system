from __future__ import annotations

from collections.abc import Callable
from typing import Any

from support_agent.routing.rules import rule_first


CLASSIFIER_CONFIDENCE_THRESHOLD = 0.75
MODEL_PATH = "models/intent_classifier"


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


def _load_classifier() -> tuple[Any, Any]:
    global _router_model, _router_tokenizer
    if _router_model is None or _router_tokenizer is None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required to run the classifier router") from exc
        _router_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _router_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
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
