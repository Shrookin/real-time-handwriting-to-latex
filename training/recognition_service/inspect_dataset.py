"""Inspect MathWriting coverage and optionally export JSONL training records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .mathwriting import iter_inkml, to_training_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a MathWriting dataset directory.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="Maximum samples to inspect; 0 means all.")
    parser.add_argument("--jsonl-out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    samples = []
    for sample in iter_inkml(args.dataset_root, args.split):
        samples.append(sample)
        if args.limit and len(samples) >= args.limit:
            break

    if not samples:
        raise SystemExit(f"No .inkml samples found in {args.dataset_root / args.split}")

    labels = Counter(sample.label for sample in samples)
    stroke_count = sum(len(sample.strokes) for sample in samples)
    point_count = sum(len(stroke.points) for sample in samples for stroke in sample.strokes)
    print(f"split: {args.split}")
    print(f"samples: {len(samples)}")
    print(f"strokes: {stroke_count}")
    print(f"points: {point_count}")
    print(f"unique labels: {len(labels)}")
    print("top labels:")
    for label, count in labels.most_common(10):
        print(f"  {count:>5}  {label}")

    if args.jsonl_out:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl_out.open("w", encoding="utf-8") as output:
            for sample in samples:
                output.write(json.dumps(to_training_record(sample), ensure_ascii=False) + "\n")
        print(f"wrote: {args.jsonl_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
