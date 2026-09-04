"""Evaluate the current geometry-plus-symbol model on full MathWriting splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate import evaluate_records
from .expression import recognize_expression
from .mathwriting import iter_inkml
from .symbol_model import load_model


def sample_strokes(sample: Any) -> list[dict[str, Any]]:
    return [{"points": [{"x": point.x, "y": point.y} for point in stroke.points]} for stroke in sample.strokes]


def predict_split(model: dict[str, Any], dataset_root: str | Path, split: str, limit: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in iter_inkml(dataset_root, split):
        result = recognize_expression(model, sample_strokes(sample))
        records.append({
            "sampleId": sample.sample_id,
            "target": sample.label,
            "prediction": result["latex"],
            "confidence": result["confidence"],
            "symbolCount": len(result.get("symbols", [])),
            "modelVersion": result["modelVersion"],
        })
        if limit and len(records) >= limit:
            break
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the current recognizer on a full MathWriting expression split.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--limit", type=int, default=0, help="Maximum expressions; 0 means the whole split.")
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = predict_split(load_model(args.model), args.dataset_root, args.split, args.limit)
    report = evaluate_records(records)
    report.update({
        "split": args.split,
        "modelVersion": records[0]["modelVersion"] if records else "unknown",
        "meanConfidence": sum(record["confidence"] for record in records) / len(records) if records else 0.0,
        "meanPredictedSymbols": sum(record["symbolCount"] for record in records) / len(records) if records else 0.0,
    })
    if args.predictions_out:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_out.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
