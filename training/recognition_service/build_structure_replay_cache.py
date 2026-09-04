"""Build a deterministic, structure-focused replay cache from MathWriting.

This is deliberately a filter over the existing training cache. It does not
invent labels or synthetic geometry; it selects real MathWriting examples
where the held-out evaluation shows the model still struggles.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_QUOTAS = {
    "scripts": 12_000,
    "matrices": 3_000,
    "sums": 2_000,
    "integrals": 4_000,
    "fractions": 6_000,
    "long": 3_000,
}


def label_categories(label: str) -> set[str]:
    categories: set[str] = set()
    if "^" in label or "_" in label:
        categories.add("scripts")
    if r"\begin{matrix}" in label:
        categories.add("matrices")
    if r"\sum" in label:
        categories.add("sums")
    if r"\int" in label:
        categories.add("integrals")
    if r"\frac" in label:
        categories.add("fractions")
    if len(label) >= 16:
        categories.add("long")
    return categories


def _reservoir_add(
    reservoir: list[tuple[str, int, str]],
    item: tuple[str, int, str],
    quota: int,
    seen: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < quota:
        reservoir.append(item)
        return
    replacement = rng.randrange(seen)
    if replacement < quota:
        reservoir[replacement] = item


def select_rows(source_dir: Path, quotas: dict[str, int]) -> tuple[list[tuple[str, int, str]], dict[str, int]]:
    import numpy as np

    manifest = json.loads((source_dir / "train.json").read_text(encoding="utf-8"))
    reservoirs: dict[str, list[tuple[str, int, str]]] = {category: [] for category in quotas}
    seen: dict[str, int] = defaultdict(int)
    rngs = {category: random.Random(1701 + index) for index, category in enumerate(quotas)}

    for shard in manifest["shards"]:
        with np.load(source_dir / shard["file"], allow_pickle=True) as data:
            for index, raw_label in enumerate(data["labels"].tolist()):
                label = str(raw_label)
                for category in label_categories(label):
                    seen[category] += 1
                    _reservoir_add(reservoirs[category], (shard["file"], index, label), quotas[category], seen[category], rngs[category])

    selected: dict[tuple[str, int], str] = {}
    for rows in reservoirs.values():
        for shard_file, index, label in rows:
            selected[(shard_file, index)] = label
    return [(shard_file, index, label) for (shard_file, index), label in sorted(selected.items())], dict(seen)


def build_cache(
    source_dir: Path,
    output_dir: Path,
    shard_size: int = 4096,
    quotas: dict[str, int] | None = None,
    dataset_kind: str = "mathwriting-structure-replay-v1",
) -> dict[str, Any]:
    import numpy as np

    source_manifest = json.loads((source_dir / "train.json").read_text(encoding="utf-8"))
    selected_quotas = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    selected, candidate_counts = select_rows(source_dir, selected_quotas)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_shard: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for shard_file, index, label in selected:
        by_shard[shard_file].append((index, label))

    arrays: list[dict[str, Any]] = []
    for shard_file in sorted(by_shard):
        with np.load(source_dir / shard_file, allow_pickle=True) as data:
            features = data["features"]
            feature_lengths = data["feature_lengths"]
            targets = data["targets"]
            for index, label in by_shard[shard_file]:
                arrays.append({
                    "features": features[index].copy(),
                    "feature_lengths": feature_lengths[index].copy(),
                    "targets": targets[index].copy(),
                    "labels": label,
                })

    shards: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(arrays), shard_size)):
        batch = arrays[start : start + shard_size]
        filename = f"train-{shard_index:05d}.npz"
        np.savez_compressed(
            output_dir / filename,
            features=np.stack([row["features"] for row in batch]),
            feature_lengths=np.asarray([row["feature_lengths"] for row in batch], dtype=np.int32),
            targets=np.stack([row["targets"] for row in batch]),
            labels=np.asarray([row["labels"] for row in batch], dtype=object),
        )
        shards.append({"file": filename, "samples": len(batch)})

    manifest = {
        "version": 1,
        "split": "train",
        "samples": len(arrays),
        "maxPoints": source_manifest["maxPoints"],
        "maxTokens": source_manifest["maxTokens"],
        "featureWidth": source_manifest["featureWidth"],
        "datasetKind": dataset_kind,
        "datasetSourceVersion": source_manifest.get("datasetSourceVersion", "unknown"),
        "vocab": source_manifest["vocab"],
        "quotas": selected_quotas,
        "candidateCounts": candidate_counts,
        "shards": shards,
    }
    (output_dir / "train.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the structure-focused MathWriting replay cache.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument(
        "--quota",
        action="append",
        default=[],
        metavar="CATEGORY=COUNT",
        help="Override a category quota; repeat for multiple categories.",
    )
    parser.add_argument("--dataset-kind", default="mathwriting-structure-replay-v1")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    quotas = dict(DEFAULT_QUOTAS)
    for override in args.quota:
        category, separator, count_text = override.partition("=")
        if not separator or category not in quotas:
            raise ValueError(f"Invalid quota override: {override!r}")
        count = int(count_text)
        if count < 0:
            raise ValueError(f"Quota must be non-negative: {override!r}")
        quotas[category] = count
    manifest = build_cache(args.source_dir, args.output_dir, args.shard_size, quotas, args.dataset_kind)
    print(json.dumps({key: value for key, value in manifest.items() if key != "vocab"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
