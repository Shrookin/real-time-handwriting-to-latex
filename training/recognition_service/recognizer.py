"""Dependency-free deterministic recognizer used by the local service.

This is deliberately a service seam, not the final ML model. A trained
checkpoint can replace `recognize_payload` while preserving the HTTP contract.
"""

from __future__ import annotations

from typing import Any


MODEL_VERSION = "deterministic-service-1"


def _strokes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [stroke for stroke in payload.get("strokes", []) if isinstance(stroke, dict)]


def recognize_payload(payload: dict[str, Any], model: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a contract-compatible response for one recognition request."""

    strokes = _strokes(payload)
    if model is not None and not isinstance(model, dict):
        from .expression_inference import recognize

        result = recognize(model, strokes)
        return {
            "schemaVersion": 1,
            "regionId": payload.get("regionId", ""),
            "revision": payload.get("revision", 0),
            **result,
        }
    if model is not None:
        from .expression import recognize_expression

        result = recognize_expression(model, strokes)
        return {
            "schemaVersion": 1,
            "regionId": payload.get("regionId", ""),
            "revision": payload.get("revision", 0),
            **result,
        }
    points = [point for stroke in strokes for point in stroke.get("points", []) if isinstance(point, dict)]
    xs = [float(point["x"]) for point in points if "x" in point]
    width = max(xs) - min(xs) if xs else 0.0
    point_count = len(points)

    if len(strokes) >= 4 or point_count > 90:
        result = {"latex": r"\frac{a}{b} + c", "display": "a⁄b + c", "confidence": 0.91}
    elif width > 180:
        result = {"latex": "x^2 + y^2 = r^2", "display": "x² + y² = r²", "confidence": 0.87}
    else:
        result = {"latex": "x^2", "display": "x²", "confidence": 0.78}

    return {
        "schemaVersion": 1,
        "regionId": payload.get("regionId", ""),
        "revision": payload.get("revision", 0),
        **result,
        "alternatives": [],
        "modelVersion": MODEL_VERSION,
    }
