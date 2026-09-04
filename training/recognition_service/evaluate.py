"""MathWriting-style token error evaluation for JSONL predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .mathwriting import tokenize_expression


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Return Levenshtein distance between token sequences."""

    previous = list(range(len(hypothesis) + 1))
    for reference_token in reference:
        current = [previous[0] + 1]
        for index, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[index] + 1,
                previous[index - 1] + (reference_token != hypothesis_token),
            ))
        previous = current
    return previous[-1]


def token_error_rate(reference: str, hypothesis: str) -> float:
    """Compute token edit distance divided by reference token count."""

    reference_tokens = tokenize_expression(reference)
    if not reference_tokens:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference_tokens, tokenize_expression(hypothesis)) / len(reference_tokens)


def _prediction_value(record: dict[str, Any]) -> str:
    return str(record.get("prediction", record.get("latex", "")))


def evaluate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate records containing `target` and `prediction` or `latex`."""

    rows = []
    for record in records:
        target = str(record.get("target", ""))
        prediction = _prediction_value(record)
        reference_tokens = tokenize_expression(target)
        distance = edit_distance(reference_tokens, tokenize_expression(prediction))
        rows.append({
            "sampleId": record.get("sampleId", record.get("regionId", "")),
            "target": target,
            "prediction": prediction,
            "tokenErrorRate": 0.0 if not reference_tokens and not prediction else distance / max(1, len(reference_tokens)),
            "exactMatch": target == prediction,
        })

    if not rows:
        return {"samples": 0, "exactMatchRate": 0.0, "meanTokenErrorRate": 0.0, "results": []}
    return {
        "samples": len(rows),
        "exactMatchRate": sum(row["exactMatch"] for row in rows) / len(rows),
        "meanTokenErrorRate": sum(row["tokenErrorRate"] for row in rows) / len(rows),
        "results": rows,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LaTeX predictions using MathWriting token error rate.")
    parser.add_argument("--predictions", required=True, type=Path, help="JSONL with target and prediction/latex fields.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = evaluate_records(load_jsonl(args.predictions))
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
