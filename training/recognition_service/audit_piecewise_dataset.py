"""Audit an InkML directory or ZIP for piecewise/cases expressions.

This is intentionally a label-only audit. It can inspect a large CROHME
archive without extracting or modifying it, which lets us measure actual
piecewise coverage before adding external data to training.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import unescape


_ANNOTATION_RE = re.compile(rb'<annotation[^>]*type=["\']([^"\']+)["\'][^>]*>(.*?)</annotation>', re.DOTALL | re.IGNORECASE)
_PIECEWISE_PATTERNS = {
    "cases": re.compile(r"\\begin\{cases\}|\\end\{cases\}"),
    "array": re.compile(r"\\begin\{array\}|\\end\{array\}"),
    "aligned": re.compile(r"\\begin\{aligned\}|\\end\{aligned\}"),
    "leftBrace": re.compile(r"\\left\s*\\\{"),
    "rowBreak": re.compile(r"\\\\"),
}


def _label(content: bytes) -> str:
    annotations: dict[str, str] = {}
    for match in _ANNOTATION_RE.finditer(content):
        annotations[match.group(1).decode("utf-8", errors="replace")] = unescape(match.group(2).decode("utf-8", errors="replace").strip())
    return annotations.get("normalizedLabel") or annotations.get("truth") or annotations.get("label") or ""


def _record(label: str, source: str, counts: Counter[str], examples: dict[str, list[dict[str, str]]]) -> None:
    counts["labeled"] += bool(label)
    if not label:
        return
    counts["labels"] += 1
    for category, pattern in _PIECEWISE_PATTERNS.items():
        if pattern.search(label):
            counts[category] += 1
            if len(examples[category]) < 10:
                examples[category].append({"source": source, "label": label})


def audit_files(paths: Iterable[tuple[str, bytes]], limit: int = 0) -> dict:
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = {category: [] for category in _PIECEWISE_PATTERNS}
    for index, (source, content) in enumerate(paths):
        if limit and index >= limit:
            break
        counts["inkml"] += 1
        _record(_label(content), source, counts, examples)
    return {"samplesScanned": counts["inkml"], "labeledSamples": counts["labels"], "piecewiseCounts": {category: counts[category] for category in _PIECEWISE_PATTERNS}, "examples": examples}


def _directory_paths(root: Path) -> Iterable[tuple[str, bytes]]:
    for path in sorted(root.rglob("*.inkml")):
        try:
            yield str(path), path.read_bytes()
        except OSError:
            continue


def _zip_paths(archive: Path) -> Iterable[tuple[str, bytes]]:
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".inkml"):
                continue
            try:
                yield info.filename, source.read(info)
            except (OSError, RuntimeError, zipfile.BadZipFile):
                continue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count piecewise/cases labels in InkML data.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", type=Path)
    source.add_argument("--archive", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report-out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dataset_root:
        result = audit_files(_directory_paths(args.dataset_root), args.limit)
    else:
        result = audit_files(_zip_paths(args.archive), args.limit)
    result["source"] = str(args.dataset_root or args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
