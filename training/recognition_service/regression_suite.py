"""Run a small fixed recognition regression suite against an expression checkpoint.

The suite intentionally uses stable held-out files from the local evaluation
corpora. It is small enough to run after every model change, but covers the
failure modes that matter most to NewNotes: operators, integrals, sine,
sequences, fractions, and piecewise expressions.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluate import edit_distance
from .expression_inference import _recognize_sequence, load_checkpoint
from .mathwriting import read_inkml, tokenize_expression, to_request


REGRESSION_CASES: tuple[dict[str, str], ...] = (
    {
        "id": "operators-plus",
        "category": "operators",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000000.inkml",
        "target": "x+y=z",
    },
    {
        "id": "operators-minus",
        "category": "operators",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000002.inkml",
        "target": "a-b=c",
    },
    {
        "id": "integral-long",
        "category": "integrals",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000003.inkml",
        "target": r"P=\int I(x,y)dxdy+CU_{t}=1-D_{t}/C_{t}",
    },
    {
        "id": "integral-with-fraction",
        "category": "integrals",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000061.inkml",
        "target": r"\int f(x)d_{q}x+a=-\frac{GM}{r^{2}}",
    },
    {
        "id": "sin-basic",
        "category": "sin",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000010.inkml",
        "target": "sin(x)",
    },
    {
        "id": "sin-with-integral",
        "category": "sin",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000025.inkml",
        "target": r"I_{n}=\int sin^{n}x+\sqrt{ax+b}",
    },
    {
        "id": "sequence-short",
        "category": "sequences",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000021.inkml",
        "target": "0,1,2,3,4",
    },
    {
        "id": "sequence-long",
        "category": "sequences",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000005.inkml",
        "target": "0,1,2,3,4,5,6",
    },
    {
        "id": "fraction-composite",
        "category": "fractions",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000004.inkml",
        "target": r"P=K_{1}\rho^{\frac{5}{3}}+\frac{\phi(t)-\phi(0)}{t}",
    },
    {
        "id": "fraction-derivative",
        "category": "fractions",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000017.inkml",
        "target": r"\frac{dP}{dt}+\hat{w}_{i}^{\prime}",
    },
    {
        "id": "piecewise-two-row",
        "category": "piecewise",
        "relativePath": "datasets/piecewise-synthetic-v1/valid/piecewise-valid-000001.inkml",
        "target": r"\begin{cases}-x & 0\le x\\x & x=1\end{cases}",
    },
    {
        "id": "piecewise-four-row",
        "category": "piecewise",
        "relativePath": "datasets/piecewise-synthetic-v1/valid/piecewise-valid-000011.inkml",
        "target": r"\begin{cases}x+1 & x\le0\\x+1 & x=1\\a+b & x=0\\a+b & x>1\end{cases}",
    },
)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0, "exactMatchRate": 0.0, "meanTokenErrorRate": 0.0}
    return {
        "samples": len(rows),
        "exactMatchRate": sum(bool(row["exactMatch"]) for row in rows) / len(rows),
        "meanTokenErrorRate": sum(float(row["tokenErrorRate"]) for row in rows) / len(rows),
    }


def run_suite(repo_root: Path, checkpoint_path: Path, device: str) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint = load_checkpoint(checkpoint_path, device)
    rows: list[dict[str, Any]] = []

    for case in REGRESSION_CASES:
        source_path = repo_root / Path(case["relativePath"])
        if not source_path.exists():
            raise FileNotFoundError(
                f"Regression fixture is missing: {source_path}. "
                "The local targeted-v3 and piecewise evaluation datasets are required."
            )
        sample = read_inkml(source_path)
        if sample.label != case["target"]:
            raise ValueError(
                f"Fixture target changed for {case['id']}: "
                f"expected {case['target']!r}, found {sample.label!r}"
            )

        response = _recognize_sequence(checkpoint, to_request(sample)["strokes"])
        prediction = str(response.get("latex", ""))
        reference_tokens = tokenize_expression(case["target"])
        hypothesis_tokens = tokenize_expression(prediction) if prediction else []
        distance = edit_distance(reference_tokens, hypothesis_tokens)
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "source": case["relativePath"],
            "target": case["target"],
            "prediction": prediction,
            "confidence": float(response.get("confidence", 0.0)),
            "exactMatch": case["target"] == prediction,
            "tokenErrorRate": distance / max(1, len(reference_tokens)),
        })

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    return {
        "benchmark": "newnotes-fixed-regression-v1",
        "checkpoint": str(checkpoint_path),
        "model": checkpoint.model_version,
        "device": str(checkpoint.device),
        "overall": _metrics(rows),
        "byCategory": {category: _metrics(category_rows) for category, category_rows in sorted(by_category.items())},
        "rows": rows,
        "elapsedSeconds": round(time.perf_counter() - started, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NewNotes' fixed expression regression suite.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_suite(args.repo_root.resolve(), args.checkpoint.resolve(), args.device)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
