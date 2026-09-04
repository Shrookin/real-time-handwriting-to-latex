"""Build a deterministic, versioned benchmark for expression structure.

The benchmark stores sample IDs and targets rather than copying InkML. This
keeps the benchmark small while making it reproducible against a pinned local
dataset archive. The selected samples come from an untouched test split when
possible; synthetic piecewise samples are recorded separately.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .mathwriting import iter_inkml, tokenize_expression


BENCHMARK_VERSION = "structural-challenge-v1"
DEFAULT_CATEGORIES = (
    "operator-expressions",
    "scripts",
    "fractions",
    "roots",
    "sums-integrals",
    "long-expressions",
)


def target_categories(target: str) -> set[str]:
    """Return measurable structural categories for a LaTeX target."""

    tokens = tokenize_expression(target)
    categories: set[str] = set()
    if any(token in {"+", "-", "=", r"\cdot", r"\times", r"\div"} for token in tokens):
        categories.add("operator-expressions")
    if "^" in tokens or "_" in tokens:
        categories.add("scripts")
    if r"\frac" in tokens or r"\dfrac" in tokens:
        categories.add("fractions")
    if r"\sqrt" in tokens:
        categories.add("roots")
    if r"\sum" in tokens or r"\int" in tokens:
        categories.add("sums-integrals")
    if len(tokens) >= 16:
        categories.add("long-expressions")
    if r"\begin{cases}" in tokens or r"\begin{array}" in tokens or r"\begin{aligned}" in tokens:
        categories.add("piecewise")
    return categories


def _select_cases(root: Path, source: str, split: str, per_category: int, categories: Iterable[str]) -> list[dict]:
    wanted = set(categories)
    counts: Counter[str] = Counter()
    selected: list[dict] = []
    for sample in iter_inkml(root, split):
        target = sample.label
        if not target:
            continue
        sample_categories = target_categories(target) & wanted
        if not sample_categories:
            continue
        eligible = {category for category in sample_categories if counts[category] < per_category}
        if not eligible:
            continue
        selected.append({
            "source": source,
            "split": split,
            "sampleId": sample.sample_id,
            "target": target,
            "categories": sorted(sample_categories),
        })
        for category in eligible:
            counts[category] += 1
        if all(counts[category] >= per_category for category in wanted):
            break
    return selected


def build_benchmark(
    dataset_root: Path,
    output: Path,
    *,
    split: str = "test",
    per_category: int = 40,
    piecewise_root: Path | None = None,
    piecewise_split: str = "valid",
    piecewise_count: int = 100,
) -> dict:
    categories = list(DEFAULT_CATEGORIES)
    cases = _select_cases(dataset_root, "mathwriting", split, per_category, categories)
    if piecewise_root and piecewise_root.exists():
        piecewise_cases = _select_cases(piecewise_root, "synthetic-piecewise", piecewise_split, piecewise_count, ("piecewise",))
        cases.extend(piecewise_cases)
    if not cases:
        raise RuntimeError("No benchmark cases were selected; check dataset roots and splits")
    report = {
        "benchmark": BENCHMARK_VERSION,
        "selection": {
            "mathwritingSplit": split,
            "mathwritingPerCategory": per_category,
            "piecewiseSplit": piecewise_split,
            "piecewiseCount": piecewise_count,
        },
        "samples": len(cases),
        "categoryCounts": dict(sorted(Counter(category for case in cases for category in case["categories"]).items())),
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the versioned structural recognition benchmark.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/structural-challenge-v1.json"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-category", type=int, default=40)
    parser.add_argument("--piecewise-root", type=Path)
    parser.add_argument("--piecewise-split", default="valid")
    parser.add_argument("--piecewise-count", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_benchmark(
        args.dataset_root,
        args.output,
        split=args.split,
        per_category=args.per_category,
        piecewise_root=args.piecewise_root,
        piecewise_split=args.piecewise_split,
        piecewise_count=args.piecewise_count,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
