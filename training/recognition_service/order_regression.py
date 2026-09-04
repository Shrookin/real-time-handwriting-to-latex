"""Probe recognition sensitivity to the order in which expressions are written.

NewNotes users do not always write left-to-right. In particular, an integral or
large anchor may be written first and a prefix such as ``1+`` may be added on
the left afterward. The model checkpoint is unchanged by this probe; it only
reorders held-out MathWriting traces and records the resulting degradation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .evaluate import edit_distance
from .expression import group_strokes
from .expression_inference import _recognize_sequence, load_checkpoint, recognize
from .mathwriting import read_inkml, tokenize_expression, to_request


ORDER_CASES: tuple[dict[str, str], ...] = (
    {
        "id": "integral-prefix-1-plus",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000244.inkml",
        "target": r"\int_{X}p(x)dx=1+S-I=EX-IM",
    },
    {
        "id": "integral-continuation-plus",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000327.inkml",
        "target": r"F(x)=\int_{a}^{x}f(t)dt+u=x+\frac{1}{x}",
    },
    {
        "id": "integral-long-continuation",
        "relativePath": "datasets/targeted-v3/valid/targeted-valid-000003.inkml",
        "target": r"P=\int I(x,y)dxdy+CU_{t}=1-D_{t}/C_{t}",
    },
)


def _stroke_left(stroke: dict[str, Any]) -> float:
    return min(float(point["x"]) for point in stroke["points"])


def _flatten_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [stroke for group in groups for stroke in group]


def _reorder(strokes: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    if variant == "original":
        return strokes
    if variant == "reverse-trace-order":
        return list(reversed(strokes))
    if variant == "geometry-left-to-right":
        return sorted(strokes, key=_stroke_left)
    if variant == "geometry-right-to-left":
        return sorted(strokes, key=_stroke_left, reverse=True)
    groups = group_strokes(strokes)
    if len(groups) < 3:
        return strokes
    if variant == "component-geometry-left-to-right":
        return _flatten_groups(groups)
    if variant == "component-geometry-right-to-left":
        return _flatten_groups(list(reversed(groups)))
    if variant == "anchor-first-left-continuation":
        return _flatten_groups([*groups[1:], groups[0]])
    if variant == "anchor-first-right-continuation":
        return _flatten_groups([groups[-1], *groups[:-1]])
    raise ValueError(f"Unknown stroke-order variant: {variant}")


def _score(target: str, prediction: str) -> tuple[bool, float]:
    reference = tokenize_expression(target)
    hypothesis = tokenize_expression(prediction) if prediction else []
    return target == prediction, edit_distance(reference, hypothesis) / max(1, len(reference))


def run_probe(repo_root: Path, checkpoint_path: Path, device: str) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint = load_checkpoint(checkpoint_path, device)
    rows: list[dict[str, Any]] = []
    variants = (
        "original",
        "reverse-trace-order",
        "geometry-left-to-right",
        "geometry-right-to-left",
        "component-geometry-left-to-right",
        "component-geometry-right-to-left",
        "anchor-first-left-continuation",
        "anchor-first-right-continuation",
    )

    for case in ORDER_CASES:
        source_path = repo_root / Path(case["relativePath"])
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        sample = read_inkml(source_path)
        if sample.label != case["target"]:
            raise ValueError(f"Fixture target changed for {case['id']}: {sample.label!r}")
        base_strokes = to_request(sample)["strokes"]
        for variant in variants:
            request_strokes = _reorder(base_strokes, variant)
            response = (
                recognize(checkpoint, request_strokes)
                if variant.startswith("anchor-first-")
                else _recognize_sequence(checkpoint, request_strokes)
            )
            prediction = str(response.get("latex", ""))
            exact, token_error = _score(case["target"], prediction)
            rows.append({
                "id": case["id"],
                "variant": variant,
                "source": case["relativePath"],
                "target": case["target"],
                "prediction": prediction,
                "confidence": float(response.get("confidence", 0.0)),
                "exactMatch": exact,
                "tokenErrorRate": token_error,
            })

    return {
        "benchmark": "newnotes-order-regression-v2",
        "checkpoint": str(checkpoint_path),
        "model": checkpoint.model_version,
        "device": str(checkpoint.device),
        "cases": len(ORDER_CASES),
        "variants": list(variants),
        "rows": rows,
        "elapsedSeconds": round(time.perf_counter() - started, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe NewNotes recognition under non-left-to-right stroke order.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_probe(args.repo_root.resolve(), args.checkpoint.resolve(), args.device)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
