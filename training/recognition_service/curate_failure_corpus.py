"""Curate reproducible recognition failures from benchmark reports.

The corpus is diagnostic: it preserves the source InkML strokes, target,
prediction, ordering variant, and a stable failure category.  It is intended
to guide a later targeted experiment, not to silently become training data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .failure_analysis import classify_failure
from .mathwriting import read_inkml, to_request


def _read_reports(paths: Iterable[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
            raise ValueError(f"Expected a report with a rows array: {path}")
        reports.append(payload)
    return reports


def _category(row: dict[str, Any], original_rows: dict[str, dict[str, Any]]) -> str:
    variant = str(row.get("variant", ""))
    original = original_rows.get(str(row.get("id", "")))
    if variant != "original" and original and bool(original.get("exactMatch")):
        return "stroke-order"
    return classify_failure(str(row.get("target", "")), str(row.get("prediction", "")))


def curate_failure_corpus(repo_root: Path, reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        report_rows = [row for row in report["rows"] if isinstance(row, dict)]
        originals = {
            str(row.get("id", "")): row
            for row in report_rows
            if row.get("variant") == "original"
        }
        for row in report_rows:
            if bool(row.get("exactMatch")):
                continue
            source = repo_root / Path(str(row["source"]))
            if not source.exists():
                raise FileNotFoundError(source)
            sample = read_inkml(source)
            target = str(row.get("target", ""))
            if sample.label != target:
                raise ValueError(f"Fixture target changed for {source}: {sample.label!r}")
            rows.append({
                "id": str(row.get("id", sample.sample_id)),
                "variant": str(row.get("variant", "original")),
                "source": str(row["source"]),
                "target": target,
                "prediction": str(row.get("prediction", "")),
                "confidence": float(row.get("confidence", 0.0)),
                "tokenErrorRate": float(row.get("tokenErrorRate", 0.0)),
                "category": _category(row, originals),
                "strokes": to_request(sample)["strokes"],
            })

    counts = Counter(str(row["category"]) for row in rows)
    return {
        "corpus": "newnotes-failure-corpus-v1",
        "purpose": "diagnostic-only; not training data until explicitly reviewed",
        "samples": len(rows),
        "categories": dict(sorted(counts.items())),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate failed benchmark rows into a reproducible diagnostic corpus.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = curate_failure_corpus(args.repo_root.resolve(), _read_reports(args.report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
