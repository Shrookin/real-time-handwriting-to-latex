"""Evaluate a checkpoint against the versioned structural challenge set."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluate import edit_distance
from .expression_inference import _recognize_sequence, load_checkpoint
from .failure_analysis import classify_failure
from .mathwriting import iter_inkml, read_inkml, to_request, tokenize_expression


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0, "exactMatchRate": 0.0, "meanTokenErrorRate": 0.0}
    return {
        "samples": len(rows),
        "exactMatchRate": sum(bool(row["exactMatch"]) for row in rows) / len(rows),
        "meanTokenErrorRate": sum(float(row["tokenErrorRate"]) for row in rows) / len(rows),
    }


def _load_cases(manifest: Path, roots: dict[str, Path]) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    benchmark = payload.get("benchmark")
    if benchmark not in {"structural-challenge-v1", "piecewise-challenge-v1"}:
        raise ValueError(f"Unsupported benchmark: {benchmark!r}")
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for case in payload.get("cases", []):
        grouped[(str(case["source"]), str(case["split"]))].add(str(case["sampleId"]))
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for (source, split), wanted in grouped.items():
        root = roots.get(source)
        if root is None:
            raise ValueError(f"No dataset root was supplied for benchmark source {source!r}")
        if source == "piecewise-challenge" and split == ".":
            for sample_id in wanted:
                path = root / f"{sample_id}.inkml"
                if path.exists():
                    found[(source, sample_id)] = {"sample": read_inkml(path), "source": source}
            continue
        for sample in iter_inkml(root, split):
            if sample.sample_id in wanted:
                found[(source, sample.sample_id)] = {"sample": sample, "source": source}
    cases: list[dict[str, Any]] = []
    for case in payload["cases"]:
        key = (str(case["source"]), str(case["sampleId"]))
        match = found.get(key)
        if match is None:
            raise FileNotFoundError(f"Benchmark sample is missing from its dataset: {key}")
        if match["sample"].label != case["target"]:
            raise ValueError(f"Target changed for benchmark sample {key}")
        cases.append({**case, **match})
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the structural challenge benchmark.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--piecewise-root", type=Path)
    parser.add_argument("--piecewise-challenge-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, default=Path("artifacts/structural-challenge-v1-report.json"))
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    roots = {"mathwriting": args.dataset_root}
    if args.piecewise_root:
        roots["synthetic-piecewise"] = args.piecewise_root
    if args.piecewise_challenge_root:
        roots["piecewise-challenge"] = args.piecewise_challenge_root
    cases = _load_cases(args.manifest, roots)
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    rows: list[dict[str, Any]] = []
    for case in cases:
        sample = case["sample"]
        response = _recognize_sequence(checkpoint, to_request(sample)["strokes"])
        prediction = str(response.get("latex", ""))
        reference_tokens = tokenize_expression(case["target"])
        hypothesis_tokens = tokenize_expression(prediction) if prediction else []
        distance = edit_distance(reference_tokens, hypothesis_tokens)
        rows.append({
            "source": case["source"],
            "split": case["split"],
            "sampleId": case["sampleId"],
            "categories": case["categories"],
            "target": case["target"],
            "prediction": prediction,
            "confidence": float(response.get("confidence", 0.0)),
            "exactMatch": case["target"] == prediction,
            "tokenErrorRate": distance / max(1, len(reference_tokens)),
            "failureCategory": classify_failure(case["target"], prediction),
        })
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
        for category in row["categories"]:
            by_category[category].append(row)
    report = {
        "benchmark": json.loads(args.manifest.read_text(encoding="utf-8")).get("benchmark"),
        "checkpoint": str(args.checkpoint),
        "model": checkpoint.model_version,
        "device": str(checkpoint.device),
        "gpu": checkpoint.torch.cuda.get_device_name(0) if checkpoint.device.type == "cuda" else None,
        "overall": _metrics(rows),
        "bySource": {source: _metrics(source_rows) for source, source_rows in sorted(by_source.items())},
        "byCategory": {category: _metrics(category_rows) for category, category_rows in sorted(by_category.items())},
        "failureCategories": {category: sum(row["failureCategory"] == category for row in rows) for category in sorted({row["failureCategory"] for row in rows})},
        "elapsedSeconds": round(time.perf_counter() - started, 2),
        "predictions": rows,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.predictions_out:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
