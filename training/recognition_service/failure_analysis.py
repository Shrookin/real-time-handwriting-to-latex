"""Heuristic failure classification for expression-recognition predictions.

The classifier is intentionally diagnostic rather than a ground-truth parser. It
assigns one primary failure class to each non-exact prediction so model runs can
be compared consistently before we have symbol-level alignment annotations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .mathwriting import tokenize_expression


STRUCTURAL_COMMANDS = {
    r"\frac",
    r"\dfrac",
    r"\sqrt",
    r"\begin{matrix}",
    r"\end{matrix}",
    r"\int",
    r"\sum",
    r"\left",
    r"\right",
}
SCRIPT_TOKENS = {"^", "_"}
OPERATORS = {"+", "-", "=", r"\cdot", r"\times", r"\div"}


def _tokens(value: str) -> list[str]:
    try:
        return tokenize_expression(value)
    except ValueError:
        return list(value)


def _has_operator_script(tokens: list[str]) -> bool:
    """Detect an operator immediately emitted as a script.

    This catches the common geometry error where a slightly displaced baseline
    plus/minus is decoded as a superscript/subscript, while not flagging a
    normal exponent such as ``x^{-1}``.
    """

    return any(token in OPERATORS and index > 0 and tokens[index - 1] in SCRIPT_TOKENS for index, token in enumerate(tokens))


def classify_failure(target: str, prediction: str) -> str:
    reference = _tokens(target)
    hypothesis = _tokens(prediction)
    if target == prediction:
        return "correct"

    if len(reference) >= 16 and abs(len(reference) - len(hypothesis)) >= max(4, len(reference) // 4):
        return "long-expression-composition"
    if _has_operator_script(hypothesis) and not _has_operator_script(reference):
        return "operator-baseline"

    reference_scripts = sum(token in SCRIPT_TOKENS for token in reference)
    hypothesis_scripts = sum(token in SCRIPT_TOKENS for token in hypothesis)
    if reference_scripts != hypothesis_scripts or (reference_scripts and hypothesis_scripts):
        return "superscript-subscript"

    reference_structural = {token for token in reference if token in STRUCTURAL_COMMANDS or token in {"{", "}"}}
    hypothesis_structural = {token for token in hypothesis if token in STRUCTURAL_COMMANDS or token in {"{", "}"}}
    if reference_structural != hypothesis_structural:
        return "latex-decoding"

    if abs(len(reference) - len(hypothesis)) >= 3:
        return "stroke-grouping-segmentation"
    return "symbol-recognition"


def analyze_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = {}
    for row in materialized:
        category = classify_failure(str(row.get("target", "")), str(row.get("prediction", row.get("latex", ""))))
        counts[category] += 1
        if category != "correct" and len(examples.setdefault(category, [])) < 5:
            examples[category].append({
                "target": str(row.get("target", "")),
                "prediction": str(row.get("prediction", row.get("latex", ""))),
            })
    failures = len(materialized) - counts.get("correct", 0)
    return {
        "samples": len(materialized),
        "correct": counts.get("correct", 0),
        "failures": failures,
        "categories": {
            category: {
                "count": count,
                "rateOfFailures": count / max(1, failures),
                "examples": examples.get(category, []),
            }
            for category, count in sorted(counts.items())
            if category != "correct"
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify expression-recognition prediction failures.")
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL rows with target and prediction fields")
    parser.add_argument("--output", type=Path, default=Path("artifacts/failure-analysis.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = {"predictions": str(args.predictions), **analyze_rows(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
