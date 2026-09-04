"""Run Real-Time Handwriting to LaTeX v1 from a JSON online-ink request."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "model" / "handwriting-to-latex-v1.onnx"
DEFAULT_VOCAB = ROOT / "model" / "handwriting-to-latex-v1-vocab.json"
MAX_POINTS = 384


def online_ink_features(strokes: list[dict[str, Any]]) -> np.ndarray:
    """Implement v1's six-feature online-ink preprocessing exactly."""
    cleaned: list[list[tuple[float, float]]] = []
    for stroke in strokes:
        points = [(float(point["x"]), float(point["y"])) for point in stroke.get("points", []) if "x" in point and "y" in point]
        if points:
            cleaned.append(points)
    point_count = sum(len(stroke) for stroke in cleaned)
    if point_count == 0:
        return np.zeros((1, 1, 6), dtype=np.float32)
    stride = max(1, math.ceil(point_count / MAX_POINTS))
    selected: list[tuple[int, float, float]] = []
    offset = 0
    for stroke_index, stroke in enumerate(cleaned):
        for point_index, (x, y) in enumerate(stroke):
            if point_index == 0 or (offset + point_index) % stride == 0:
                selected.append((stroke_index, x, y))
        offset += len(stroke)
    selected = selected[:MAX_POINTS]
    min_x, max_x = min(row[1] for row in selected), max(row[1] for row in selected)
    min_y, max_y = min(row[2] for row in selected), max(row[2] for row in selected)
    scale = max(max_x - min_x, max_y - min_y, 1.0)
    previous_stroke, previous_x, previous_y = selected[0]
    rows = []
    for stroke_index, x, y in selected:
        new_stroke = 1.0 if stroke_index != previous_stroke else 0.0
        rows.append([(x - min_x) / scale * 2.0 - 1.0, (y - min_y) / scale * 2.0 - 1.0, (x - previous_x) / scale, (y - previous_y) / scale, new_stroke, 1.0 - new_stroke])
        previous_stroke, previous_x, previous_y = stroke_index, x, y
    return np.asarray([rows], dtype=np.float32)


def decode(token_ids: np.ndarray, token_scores: np.ndarray, vocab: dict[str, Any]) -> dict[str, Any]:
    decoded, scores = [], []
    for token_id, score in zip(token_ids[0], token_scores[0]):
        token = vocab["tokens"][int(token_id)] if 0 <= int(token_id) < len(vocab["tokens"]) else "<unk>"
        if token == vocab["eosToken"]:
            break
        if token not in {"<pad>", "<bos>"}:
            decoded.append(token)
            scores.append(float(score))
    return {"latex": "".join(decoded), "tokens": decoded, "confidence": sum(scores) / len(scores) if scores else 0.0, "modelVersion": vocab["modelVersion"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSON file containing a strokes array")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    args = parser.parse_args()
    request = json.loads(args.input.read_text(encoding="utf-8"))
    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    token_ids, token_scores = session.run(None, {"features": online_ink_features(request.get("strokes", []))})
    print(json.dumps(decode(token_ids, token_scores, vocab), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
