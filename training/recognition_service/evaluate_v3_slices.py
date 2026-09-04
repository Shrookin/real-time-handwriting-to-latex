"""Evaluate a checkpoint on the v3 targeted slices and structural data."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .evaluate import edit_distance
from .expression_inference import _recognize_sequence, load_checkpoint
from .mathwriting import MathWritingSample, iter_inkml, read_inkml, to_request, tokenize_expression


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "exactMatchRate": sum(bool(row["exactMatch"]) for row in rows) / max(1, len(rows)),
        "meanTokenErrorRate": sum(float(row["tokenErrorRate"]) for row in rows) / max(1, len(rows)),
    }


def _rows(checkpoint: Any, samples: Iterable[MathWritingSample], source: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if limit and index >= limit:
            break
        target = sample.label
        if not target:
            continue
        response = _recognize_sequence(checkpoint, to_request(sample)["strokes"])
        prediction = str(response.get("latex", ""))
        reference_tokens = tokenize_expression(target)
        hypothesis_tokens = tokenize_expression(prediction) if prediction else []
        distance = edit_distance(reference_tokens, hypothesis_tokens)
        rows.append({
            "source": source,
            "sampleId": sample.sample_id,
            "augmentation": sample.annotations.get("augmentation", ""),
            "target": target,
            "prediction": prediction,
            "confidence": float(response.get("confidence", 0.0)),
            "exactMatch": target == prediction,
            "tokenErrorRate": distance / max(1, len(reference_tokens)),
        })
    return rows


def _symbol_samples(root: Path) -> Iterable[MathWritingSample]:
    for path in sorted(root.glob("*.inkml")):
        try:
            yield read_inkml(path)
        except (OSError, ValueError):
            continue


def build_report(checkpoint_path: Path, targeted_root: Path, piecewise_root: Path | None, mathwriting_root: Path | None, device: str, limit: int) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint = load_checkpoint(checkpoint_path, device)
    rows: list[dict[str, Any]] = []
    if targeted_root.exists():
        rows.extend(_rows(checkpoint, iter_inkml(targeted_root, "valid"), "targeted-v3", limit))
    if piecewise_root and piecewise_root.exists():
        rows.extend(_rows(checkpoint, iter_inkml(piecewise_root, "valid"), "piecewise-synthetic-v1", limit))
    if mathwriting_root and mathwriting_root.exists():
        rows.extend(_rows(checkpoint, _symbol_samples(mathwriting_root / "symbols"), "mathwriting-symbols", limit))

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_augmentation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
        if row["augmentation"]:
            by_augmentation[row["augmentation"]].append(row)
    return {
        "benchmark": "mathwriting-v3-targeted-slices",
        "checkpoint": str(checkpoint_path),
        "model": checkpoint.model_version,
        "device": str(checkpoint.device),
        "overall": _metrics(rows),
        "bySource": {key: _metrics(value) for key, value in sorted(by_source.items())},
        "byAugmentation": {key: _metrics(value) for key, value in sorted(by_augmentation.items())},
        "elapsedSeconds": round(time.perf_counter() - started, 2),
        "predictions": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate v3 checkpoints on targeted recognition slices.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targeted-root", type=Path, required=True)
    parser.add_argument("--piecewise-root", type=Path)
    parser.add_argument("--mathwriting-root", type=Path)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_report(args.checkpoint, args.targeted_root, args.piecewise_root, args.mathwriting_root, args.device, args.limit)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
