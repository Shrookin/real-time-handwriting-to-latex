"""Evaluate a checkpoint on CROHME InkML expression data."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

from .crohme import iter_crohme_inkml
from .evaluate import edit_distance
from .expression_inference import load_checkpoint, normalize_operator_scripts
from .expression_torch import _decode_ids, _sample_features
from .failure_analysis import analyze_rows, classify_failure
from .mathwriting import tokenize_expression


def _categories(target: str) -> set[str]:
    tokens = tokenize_expression(target)
    categories = {"all"}
    if len(tokens) <= 8:
        categories.add("short")
    if len(tokens) >= 16:
        categories.add("long")
    patterns = {
        "fractions": (r"\frac", r"\dfrac"),
        "roots": (r"\sqrt",),
        "superscripts": ("^",),
        "subscripts": ("_",),
        "matrices": (r"\begin{matrix}",),
        "integrals": (r"\int",),
        "sums": (r"\sum",),
        "piecewise": (r"\begin{cases}", r"\begin{array}", r"\begin{aligned}"),
    }
    for category, needles in patterns.items():
        if any(needle in tokens for needle in needles):
            categories.add(category)
    return categories


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0, "exactMatchRate": 0.0, "meanTokenErrorRate": 0.0}
    return {
        "samples": len(rows),
        "exactMatchRate": sum(row["exactMatch"] for row in rows) / len(rows),
        "meanTokenErrorRate": sum(row["tokenErrorRate"] for row in rows) / len(rows),
    }


def _batches(items: list[tuple[str, list[list[float]], str]], batch_size: int) -> Iterable[list[tuple[str, list[list[float]], str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on CROHME without training.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="CROHME release directory containing InkML files")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, default=Path("artifacts/crohme-evaluation.json"))
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    samples = []
    skipped_invalid_labels = []
    for sample in iter_crohme_inkml(args.dataset_root):
        try:
            tokenize_expression(sample.label)
        except ValueError:
            skipped_invalid_labels.append(sample.sample_id)
            continue
        samples.append((sample.sample_id, _sample_features(sample, checkpoint.max_points), sample.label))
        if args.limit and len(samples) >= args.limit:
            break
    if not samples:
        raise SystemExit("No truth-bearing CROHME InkML files were found below --dataset-root.")

    torch = checkpoint.torch
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    token_to_id = {token: index for index, token in enumerate(checkpoint.vocab)}
    rows: list[dict[str, Any]] = []
    checkpoint.model.eval()
    with torch.no_grad():
        for batch in _batches(samples, args.batch_size):
            max_features = max(len(item[1]) for item in batch)
            features = torch.zeros((len(batch), max_features, 6), dtype=torch.float32, device=checkpoint.device)
            lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long, device=checkpoint.device)
            for index, (_sample_id, feature_row, _target) in enumerate(batch):
                features[index, : len(feature_row)] = torch.tensor(feature_row, dtype=torch.float32, device=checkpoint.device)
            predicted, scores = checkpoint.model.greedy_decode(
                features,
                lengths,
                token_to_id["<bos>"],
                token_to_id["<eos>"],
                checkpoint.max_tokens,
                pack_padded_sequence,
                pad_packed_sequence,
                return_scores=True,
            )
            for index, (sample_id, _feature_row, target) in enumerate(batch):
                prediction = normalize_operator_scripts(_decode_ids(predicted[index].cpu().tolist(), checkpoint.vocab))
                reference_tokens = tokenize_expression(target)
                hypothesis_tokens = tokenize_expression(prediction) if prediction else []
                distance = edit_distance(reference_tokens, hypothesis_tokens)
                token_scores = scores[index].cpu().tolist()
                rows.append({
                    "sampleId": sample_id,
                    "target": target,
                    "prediction": prediction,
                    "confidence": sum(token_scores) / max(1, len(token_scores)),
                    "tokenErrorRate": distance / max(1, len(reference_tokens)),
                    "exactMatch": target == prediction,
                    "failureCategory": classify_failure(target, prediction),
                })

    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for category in _categories(row["target"]):
            by_category.setdefault(category, []).append(row)
    report = {
        "dataset": "CROHME",
        "evaluationOnly": False,
        "licenseNote": "This report uses CROHME data released through the Figshare CC BY 4.0 bundle; preserve attribution when sharing derived artifacts.",
        "checkpoint": str(args.checkpoint),
        "modelVersion": checkpoint.model_version,
        "device": str(checkpoint.device),
        "gpu": torch.cuda.get_device_name(0) if checkpoint.device.type == "cuda" else None,
        "samples": len(rows),
        "skippedInvalidLabels": skipped_invalid_labels,
        "overall": _metrics(rows),
        "categories": {category: _metrics(category_rows) for category, category_rows in sorted(by_category.items())},
        "failureAnalysis": analyze_rows(rows),
        "elapsedSeconds": round(time.perf_counter() - started, 2),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.predictions_out:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
