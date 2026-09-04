"""Pure-Python nearest-neighbor baseline for isolated MathWriting symbols."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .mathwriting import MathWritingSample, InkPoint, read_inkml


MODEL_VERSION = "mathwriting-symbol-knn-1"
FEATURE_VERSION = 1
RESAMPLED_POINTS = 8
MAX_STROKES = 8


def _xy(point: Any) -> tuple[float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point.x), float(point.y)


def _stroke_points(stroke: Any) -> Sequence[Any]:
    return stroke.get("points", []) if isinstance(stroke, dict) else stroke.points


def _resample(points: Sequence[Any], count: int) -> list[tuple[float, float]]:
    xy = [_xy(point) for point in points]
    if not xy:
        return [(0.0, 0.0)] * count
    if len(xy) == 1:
        return xy * count
    lengths = [0.0]
    for first, second in zip(xy, xy[1:]):
        lengths.append(lengths[-1] + math.hypot(second[0] - first[0], second[1] - first[1]))
    total = lengths[-1]
    if total == 0:
        return [xy[0]] * count
    result = []
    for index in range(count):
        target = total * index / (count - 1)
        right = next((position for position, length in enumerate(lengths) if length >= target), len(lengths) - 1)
        if right == 0:
            result.append(xy[0])
            continue
        left = right - 1
        span = lengths[right] - lengths[left]
        ratio = 0.0 if span == 0 else (target - lengths[left]) / span
        result.append((xy[left][0] + (xy[right][0] - xy[left][0]) * ratio, xy[left][1] + (xy[right][1] - xy[left][1]) * ratio))
    return result


def extract_features(strokes: Sequence[Any]) -> list[float]:
    """Normalize stroke geometry into a fixed-size, stroke-aware vector."""

    all_points = [_xy(point) for stroke in strokes for point in _stroke_points(stroke)]
    if not all_points:
        return [0.0] * (3 + MAX_STROKES * (1 + RESAMPLED_POINTS * 2))
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    scale = max(max_x - min_x, max_y - min_y, 1.0)
    features = [min(1.0, len(strokes) / MAX_STROKES), min(1.0, len(all_points) / 200.0), (max_x - min_x) / scale]
    for stroke_index in range(MAX_STROKES):
        if stroke_index >= len(strokes):
            features.extend([0.0] * (1 + RESAMPLED_POINTS * 2))
            continue
        points = _stroke_points(strokes[stroke_index])
        features.append(1.0)
        for x, y in _resample(points, RESAMPLED_POINTS):
            features.extend([(x - min_x) / scale, (y - min_y) / scale])
    return features


def _sample_features(sample: MathWritingSample) -> list[float]:
    return extract_features(sample.strokes)


def build_model(dataset_root: str | Path, *, limit: int = 0, evaluation_limit: int = 200) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a nearest-neighbor model and evaluate it on a deterministic holdout."""

    paths = sorted((Path(dataset_root) / "symbols").glob("*.inkml"))
    if limit:
        paths = paths[:limit]
    samples = [read_inkml(path) for path in paths]
    samples = [sample for sample in samples if sample.label]
    split = max(1, int(len(samples) * 0.8))
    training = samples[:split]
    evaluation = samples[split:split + evaluation_limit] if evaluation_limit else samples[split:]
    examples = [{"id": sample.sample_id, "label": sample.label, "features": _sample_features(sample)} for sample in training]
    model = {"schemaVersion": 1, "modelVersion": MODEL_VERSION, "featureVersion": FEATURE_VERSION, "examples": examples}
    correct = 0
    for sample in evaluation:
        prediction = predict(model, [{"points": [{"x": point.x, "y": point.y} for point in stroke.points]} for stroke in sample.strokes])["latex"]
        correct += prediction == sample.label
    report = {"modelVersion": MODEL_VERSION, "samples": len(samples), "trainingSamples": len(training), "evaluationSamples": len(evaluation), "top1Accuracy": correct / len(evaluation) if evaluation else 0.0, "labelCount": len({sample.label for sample in samples})}
    return model, report


def save_model(model: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def load_model(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)) / max(1, len(first)))


def _coarse_distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Cheap shape prefilter before comparing the full stroke descriptor."""

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first[:3], second[:3])))


def predict(model: dict[str, Any], strokes: Sequence[Any], alternatives: int = 3) -> dict[str, Any]:
    features = extract_features(strokes)
    examples = model.get("examples", [])
    # Full-expression samples create many symbol queries. Keep the same
    # nearest-neighbor model, but avoid calculating the 131-dimensional
    # descriptor against every example for every query.
    coarse = sorted(((_coarse_distance(features, example["features"]), example) for example in examples), key=lambda item: item[0])
    candidates = [example for _, example in coarse[: min(384, len(coarse))]]
    ranked = sorted(((_distance(features, example["features"]), example) for example in candidates), key=lambda item: item[0])
    if not ranked:
        return {"latex": "", "display": "", "confidence": 0.0, "alternatives": [], "modelVersion": model.get("modelVersion", "unknown")}
    selected_distance, selected = ranked[0]
    labels = []
    for distance, example in ranked:
        if example["label"] not in labels:
            labels.append(example["label"])
        if len(labels) >= alternatives:
            break
    confidence = max(0.05, min(0.99, 1.0 - selected_distance * 2.5))
    return {"latex": selected["label"], "display": selected["label"], "confidence": confidence, "alternatives": [{"latex": label, "display": label, "confidence": max(0.05, confidence - index * 0.08)} for index, label in enumerate(labels[1:], start=1)], "modelVersion": model.get("modelVersion", "unknown")}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train a pure-Python MathWriting isolated-symbol baseline.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--evaluation-limit", type=int, default=200, help="Held-out samples to score; 0 means all.")
    args = parser.parse_args()
    model, report = build_model(args.dataset_root, limit=args.limit, evaluation_limit=args.evaluation_limit)
    save_model(model, args.output)
    print(json.dumps(report, indent=2))
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
