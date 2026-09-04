"""Merge compatible disk-backed expression caches into a balanced v3 cache."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

from .expression_full_train import SPECIAL_TOKENS
from .expression_torch import LEGACY_FEATURE_WIDTH, _token_ids


def _manifest(cache_dir: Path, split: str) -> dict[str, Any]:
    path = cache_dir / f"{split}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("shards") or int(data.get("samples", 0)) <= 0:
        raise ValueError(f"Cache manifest has no samples: {path}")
    missing = [cache_dir / shard["file"] for shard in data["shards"] if not (cache_dir / shard["file"]).exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    return data


def _parse_source(value: str) -> tuple[Path, int]:
    path_text, _, limit_text = value.rpartition("=")
    if not path_text:
        return Path(value), 0
    try:
        limit = int(limit_text)
    except ValueError:
        return Path(value), 0
    return Path(path_text), limit


def _write_split(
    output: Path,
    split: str,
    sources: list[tuple[Path, int]],
    vocab: list[str],
    max_points: int,
    max_tokens: int,
    feature_width: int,
    shard_size: int,
    rng: random.Random,
) -> dict[str, Any]:
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    for old_shard in output.glob(f"{split}-*.npz"):
        old_shard.unlink()
    token_to_id = {token: index for index, token in enumerate(vocab)}
    feature_rows: list[Any] = []
    length_rows: list[int] = []
    label_rows: list[str] = []
    shards: list[dict[str, Any]] = []
    sample_count = 0

    def write_batch(features: Any, lengths: Any, labels: Any) -> None:
        nonlocal sample_count
        batch_size = len(labels)
        targets = np.full((batch_size, max_tokens), token_to_id["<pad>"], dtype=np.int32)
        for index, label in enumerate(labels):
            row = _token_ids(str(label), token_to_id, max_tokens)
            targets[index, : len(row)] = np.asarray(row, dtype=np.int32)
        name = f"{split}-{len(shards):05d}.npz"
        np.savez(
            output / name,
            features=np.asarray(features, dtype=np.float16),
            feature_lengths=np.asarray(lengths, dtype=np.int32),
            targets=targets,
            labels=np.asarray(labels, dtype=object),
        )
        shards.append({"file": name, "samples": batch_size})
        sample_count += batch_size

    def flush() -> None:
        nonlocal feature_rows, length_rows, label_rows
        if not feature_rows:
            return
        write_batch(feature_rows, length_rows, label_rows)
        feature_rows, length_rows, label_rows = [], [], []

    for source_dir, limit in sources:
        manifest = _manifest(source_dir, split)
        seen = 0
        for shard in manifest["shards"]:
            with np.load(source_dir / shard["file"], allow_pickle=True) as data:
                order = list(range(int(shard["samples"])))
                rng.shuffle(order)
                take = len(order)
                if limit:
                    take = min(take, max(0, limit - seen))
                selected = np.asarray(order[:take], dtype=np.intp)
                if not len(selected):
                    continue
                selected_features = data["features"][selected]
                if selected_features.shape[1:] != (max_points, feature_width):
                    raise ValueError(f"Incompatible feature shape in {source_dir / shard['file']}: {selected_features.shape}")
                selected_lengths = np.asarray(data["feature_lengths"], dtype=np.int32)[selected]
                selected_labels = np.asarray(data["labels"], dtype=object)[selected]
                if not feature_rows and len(selected) == shard_size:
                    write_batch(selected_features, selected_lengths, selected_labels)
                else:
                    for index in range(len(selected)):
                        feature_rows.append(np.array(selected_features[index], dtype=np.float16, copy=True))
                        length_rows.append(int(selected_lengths[index]))
                        label_rows.append(str(selected_labels[index]))
                        if len(feature_rows) >= shard_size:
                            flush()
                seen += len(selected)
            del data
            gc.collect()
            if limit and seen >= limit:
                break
    flush()
    manifest = {
        "version": 1,
        "split": split,
        "samples": sample_count,
        "maxPoints": max_points,
        "maxTokens": max_tokens,
        "featureWidth": feature_width,
        "datasetKind": "mathwriting-v3-balanced",
        "datasetSourceVersion": "mathwriting-v3-balanced-cache-1",
        "vocab": vocab,
        "shards": shards,
    }
    (output / f"{split}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_cache(
    output: Path,
    train_sources: list[tuple[Path, int]],
    valid_source: Path,
    *,
    checkpoint_vocab: Path | None = None,
    max_tokens: int = 64,
    shard_size: int = 4096,
    seed: int = 43,
) -> dict[str, Any]:
    import torch

    train_manifests = [_manifest(path, "train") for path, _limit in train_sources]
    valid_manifest = _manifest(valid_source, "valid")
    widths = {int(manifest.get("featureWidth", LEGACY_FEATURE_WIDTH)) for manifest in [*train_manifests, valid_manifest]}
    if len(widths) != 1:
        raise ValueError(f"Incompatible feature widths: {sorted(widths)}")
    feature_width = next(iter(widths))
    max_points = int(train_manifests[0]["maxPoints"])
    if any(int(manifest["maxPoints"]) != max_points or int(manifest["maxTokens"]) != max_tokens for manifest in [*train_manifests, valid_manifest]):
        raise ValueError("All caches must use the same maxPoints and maxTokens")

    vocab: list[str] = []
    if checkpoint_vocab:
        checkpoint = torch.load(checkpoint_vocab, map_location="cpu", weights_only=False)
        vocab.extend(str(token) for token in checkpoint.get("vocab", []))
    for manifest in [*train_manifests, valid_manifest]:
        for token in manifest.get("vocab", []):
            if token not in vocab:
                vocab.append(str(token))
    for token in SPECIAL_TOKENS:
        if token in vocab:
            vocab.remove(token)
    vocab = [*SPECIAL_TOKENS, *vocab]
    rng = random.Random(seed)
    train = _write_split(output, "train", train_sources, vocab, max_points, max_tokens, feature_width, shard_size, rng)
    valid = _write_split(output, "valid", [(valid_source, int(valid_manifest["samples"]))], vocab, max_points, max_tokens, feature_width, shard_size, random.Random(seed))
    report = {"dataset": "mathwriting-v3-balanced", "featureWidth": feature_width, "maxPoints": max_points, "maxTokens": max_tokens, "vocabSize": len(vocab), "trainSamples": train["samples"], "validSamples": valid["samples"], "sources": [{"path": str(path), "limit": limit} for path, limit in train_sources]}
    (output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge compatible expression caches for v3 balanced training.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="Training cache path optionally followed by =limit; repeat for each source.")
    parser.add_argument("--valid-source", type=Path, required=True)
    parser.add_argument("--checkpoint-vocab", type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=43)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_cache(args.output, [_parse_source(value) for value in args.source], args.valid_source, checkpoint_vocab=args.checkpoint_vocab, max_tokens=args.max_tokens, shard_size=args.shard_size, seed=args.seed)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
