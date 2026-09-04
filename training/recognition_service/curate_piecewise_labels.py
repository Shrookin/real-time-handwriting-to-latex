"""Curate a deduplicated, label-only piecewise bank from InkML sources.

This intentionally copies no strokes and makes no licensing claim about the
input. It is a provenance-preserving way to inspect MathWriting/HME100K-like
archives before deciding whether their labels can drive synthetic online-ink
training. Raw source archives remain outside the repository.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

from .audit_piecewise_dataset import _label


_CATEGORY_PATTERNS = {
    "cases": re.compile(r"\\begin\{cases\}|\\end\{cases\}"),
    "array": re.compile(r"\\begin\{array\}|\\end\{array\}"),
    "aligned": re.compile(r"\\begin\{aligned\}|\\end\{aligned\}"),
    "leftBrace": re.compile(r"\\left\s*\\\{"),
    "rowBreak": re.compile(r"\\\\"),
}


def _normalize(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip())


def curate_labels(paths: Iterable[tuple[str, bytes]], limit: int = 0) -> tuple[list[dict], dict]:
    records: list[dict] = []
    seen: set[str] = set()
    scanned = 0
    for source, content in paths:
        if limit and scanned >= limit:
            break
        scanned += 1
        label = _normalize(_label(content))
        if not label or not any(pattern.search(label) for pattern in _CATEGORY_PATTERNS.values()):
            continue
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]
        if digest in seen:
            continue
        seen.add(digest)
        records.append({
            "label": label,
            "labelId": digest,
            "source": source,
            "categories": [name for name, pattern in _CATEGORY_PATTERNS.items() if pattern.search(label)],
        })
    counts = Counter(category for record in records for category in record["categories"])
    return records, {"samplesScanned": scanned, "uniqueLabels": len(records), "categoryCounts": dict(counts)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate a provenance-preserving piecewise label bank from InkML.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", type=Path)
    source.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dataset_root:
        from .audit_piecewise_dataset import _directory_paths
        paths = _directory_paths(args.dataset_root)
    else:
        from .audit_piecewise_dataset import _zip_paths
        paths = _zip_paths(args.archive)
    records, report = curate_labels(paths, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    report["output"] = str(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
