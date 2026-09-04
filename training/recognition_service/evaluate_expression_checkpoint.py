"""Evaluate a trained expression checkpoint with category-level metrics."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from .expression_full_train import _build_cache, _cached_batches
from .expression_inference import load_checkpoint, normalize_operator_scripts
from .expression_torch import _decode_ids
from .evaluate import edit_distance
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
        "greek": tuple(
            f"\\{name}" for name in (
                "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "pi", "sigma", "phi", "varphi", "omega",
            )
        ),
    }
    for category, needles in patterns.items():
        if any(needle in tokens for needle in needles):
            categories.add(category)
    return categories


def _empty_metric() -> dict[str, Any]:
    return {"samples": 0, "exactMatchRate": 0.0, "meanTokenErrorRate": 0.0}


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return _empty_metric()
    return {
        "samples": len(rows),
        "exactMatchRate": sum(row["exactMatch"] for row in rows) / len(rows),
        "meanTokenErrorRate": sum(row["tokenErrorRate"] for row in rows) / len(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a full-expression checkpoint on expression validation data.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-kind", choices=("mathwriting", "synthetic-piecewise"), default="mathwriting")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/mathwriting-cache-full-valid"))
    parser.add_argument("--report-out", type=Path, default=Path("artifacts/expression-v2-full-valid.json"))
    parser.add_argument("--predictions-out", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    vocab = checkpoint.vocab
    valid_manifest = _build_cache(
        args.dataset_root,
        args.cache_dir,
        "valid",
        args.limit,
        checkpoint.max_points,
        checkpoint.max_tokens,
        vocab,
        args.shard_size,
        args.dataset_kind,
        checkpoint.feature_width,
    )
    import numpy as np
    torch = checkpoint.torch
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    token_to_id = {token: index for index, token in enumerate(checkpoint.vocab)}
    rows: list[dict[str, Any]] = []
    checkpoint.model.eval()
    with torch.no_grad():
        for features_np, lengths_np, _targets_np, labels_np in _cached_batches(valid_manifest, args.cache_dir, args.batch_size, False, random.Random(0), np):
            features = torch.from_numpy(features_np).float().to(checkpoint.device)
            lengths = torch.from_numpy(lengths_np).long().to(checkpoint.device)
            predicted, _scores = checkpoint.model.greedy_decode(
                features,
                lengths,
                token_to_id["<bos>"],
                token_to_id["<eos>"],
                checkpoint.max_tokens,
                pack_padded_sequence,
                pad_packed_sequence,
                return_scores=True,
            )
            for target, prediction in zip(labels_np.tolist(), predicted.cpu().tolist()):
                target = str(target)
                hypothesis = normalize_operator_scripts(_decode_ids(prediction, checkpoint.vocab))
                reference_tokens = tokenize_expression(target)
                hypothesis_tokens = tokenize_expression(hypothesis) if hypothesis else []
                distance = edit_distance(reference_tokens, hypothesis_tokens)
                rows.append({
                    "target": target,
                    "prediction": hypothesis,
                    "tokenErrorRate": distance / max(1, len(reference_tokens)),
                    "exactMatch": target == hypothesis,
                    "failureCategory": classify_failure(target, hypothesis),
                })

    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for category in _categories(row["target"]):
            by_category.setdefault(category, []).append(row)
    report = {
        "model": checkpoint.model_version,
        "datasetKind": args.dataset_kind,
        "checkpoint": str(args.checkpoint),
        "device": str(checkpoint.device),
        "gpu": torch.cuda.get_device_name(0) if checkpoint.device.type == "cuda" else None,
        "samples": len(rows),
        "overall": _metrics(rows),
        "categories": {category: _metrics(category_rows) for category, category_rows in sorted(by_category.items())},
        "failureAnalysis": analyze_rows(rows),
        "elapsedSeconds": round(time.perf_counter() - started, 2),
        "cacheDir": str(args.cache_dir),
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
