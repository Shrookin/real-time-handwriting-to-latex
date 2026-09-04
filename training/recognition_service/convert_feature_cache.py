"""Convert legacy six-feature caches to the ten-feature geometry contract.

The original cache already contains normalized sampled point positions and
stroke-boundary flags. The geometry features can therefore be reconstructed
without reparsing hundreds of thousands of InkML files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


LEGACY_FEATURE_WIDTH = 6
GEOMETRY_FEATURE_WIDTH = 10


def _add_bounds(features, lengths):
    import numpy as np

    converted = np.zeros((*features.shape[:2], GEOMETRY_FEATURE_WIDTH), dtype=np.float16)
    converted[:, :, :LEGACY_FEATURE_WIDTH] = features
    for row_index, length_value in enumerate(lengths.tolist()):
        length = int(length_value)
        if length <= 0:
            continue
        points = features[row_index, :length, :LEGACY_FEATURE_WIDTH].astype(np.float32)
        starts = [0] + [index for index in range(1, length) if points[index, 4] > 0.5]
        starts.append(length)
        for start, end in zip(starts, starts[1:]):
            segment = points[start:end, :2]
            bounds = np.array([segment[:, 0].min(), segment[:, 1].min(), segment[:, 0].max(), segment[:, 1].max()], dtype=np.float16)
            converted[row_index, start:end, LEGACY_FEATURE_WIDTH:] = bounds
    return converted


def convert_cache(source_dir: Path, target_dir: Path, overwrite: bool = False) -> dict:
    import numpy as np

    source_train = json.loads((source_dir / "train.json").read_text(encoding="utf-8"))
    if int(source_train.get("featureWidth", LEGACY_FEATURE_WIDTH)) != LEGACY_FEATURE_WIDTH:
        raise ValueError(f"Source cache is not a six-feature cache: {source_dir}")
    if target_dir.exists() and any(target_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Target cache is not empty: {target_dir}")
        for child in target_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)
    converted_files = 0
    sample_count = 0
    for split in ("train", "valid"):
        source_manifest = json.loads((source_dir / f"{split}.json").read_text(encoding="utf-8"))
        target_manifest = {**source_manifest, "featureWidth": GEOMETRY_FEATURE_WIDTH, "version": int(source_manifest.get("version", 1)) + 1}
        target_manifest["shards"] = []
        for shard in source_manifest.get("shards", []):
            source_path = source_dir / shard["file"]
            target_path = target_dir / shard["file"]
            with np.load(source_path, allow_pickle=True) as data:
                features = _add_bounds(data["features"], data["feature_lengths"])
                np.savez(
                    target_path,
                    features=features,
                    feature_lengths=data["feature_lengths"],
                    targets=data["targets"],
                    labels=data["labels"],
                )
            target_manifest["shards"].append(dict(shard))
            converted_files += 1
            sample_count += int(shard["samples"])
        (target_dir / f"{split}.json").write_text(json.dumps(target_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"source": str(source_dir), "target": str(target_dir), "featureWidth": GEOMETRY_FEATURE_WIDTH, "shards": converted_files, "samples": sample_count}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert six-feature expression caches to geometry-aware caches.")
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = convert_cache(args.source_cache_dir, args.target_cache_dir, args.overwrite)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
