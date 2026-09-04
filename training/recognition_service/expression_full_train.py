"""Disk-backed full MathWriting training runner.

The earlier experiment kept every expression as nested Python lists. That is
fine for a few thousand examples, but it does not scale to the 229k train
split. This runner first writes fixed-shape NumPy shards, then trains one shard
at a time on CUDA so RAM usage stays bounded.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from html import unescape
import json
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .evaluate import edit_distance
from .crohme import iter_crohme_labels, iter_crohme_split
from .crohme23 import iter_crohme23_labels, iter_crohme23_split
from .expression_torch import (
    GEOMETRY_FEATURE_WIDTH,
    LEGACY_FEATURE_WIDTH,
    _build_model,
    _decode_ids,
    _require_torch,
    _sample_features,
    _token_ids,
)
from .expression_inference import normalize_operator_scripts
from .mathwriting import MathWritingSample, iter_inkml, read_inkml, tokenize_expression


LABEL_RE = re.compile(rb'<annotation type="normalizedLabel">(.*?)</annotation>', re.DOTALL)
FAST_LABEL_RE = re.compile(rb'<annotation[^>]*type=["\']normalizedLabel["\'][^>]*>(.*?)</annotation>', re.DOTALL | re.IGNORECASE)
DATASET_SOURCE_VERSION = "crohme-expression-splits-v1"
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
PIECEWISE_VOCAB_LABELS = (
    r"\begin{cases}x & x<0\\0 & x\ge0\end{cases}",
    r"\begin{cases}x^2+1 & -1<x\\-x & x\le0\\1 & x>1\end{cases}",
)


def _paths(root: Path, split: str):
    return sorted((root / split).glob("*.inkml"))


def _samples(root: Path, split: str, dataset_kind: str):
    if dataset_kind == "crohme":
        return iter_crohme_split(root, split)
    if dataset_kind == "crohme23":
        return iter_crohme23_split(root, split)
    if dataset_kind == "mathwriting-synthetic":
        return iter_inkml(root, "synthetic")
    if dataset_kind == "mathwriting-symbols":
        return iter_inkml(root, "symbols")
    if dataset_kind == "mathwriting":
        return _iter_mathwriting_parallel(root, split)
    return iter_inkml(root, split)


def _read_mathwriting_sample(path: Path) -> MathWritingSample | None:
    try:
        return read_inkml(path)
    except (OSError, ValueError):
        return None


def _iter_mathwriting_parallel(root: Path, split: str):
    """Parse MathWriting InkML concurrently during cache construction.

    The files are small and the workload is dominated by file I/O and XML
    parsing. A bounded thread pool keeps memory bounded while avoiding the
    prohibitively slow one-file-at-a-time path for the expanded v3 dataset.
    """

    paths = _paths(root, split)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for sample in pool.map(_read_mathwriting_sample, paths):
            if sample is not None:
                yield sample


def _labels(root: Path, split: str, limit: int, dataset_kind: str) -> list[str]:
    labels: list[str] = []
    if dataset_kind == "synthetic-piecewise":
        # The generator writes a fixed canonical grammar.  Avoid reparsing
        # thousands of generated XML files just to rediscover these tokens.
        return list(PIECEWISE_VOCAB_LABELS)
    if dataset_kind == "crohme":
        for label in iter_crohme_labels(root, split):
            labels.append(label)
            if limit and len(labels) >= limit:
                break
        return labels
    if dataset_kind == "mathwriting":
        for path in sorted((root / split).glob("*.inkml")):
            try:
                content = path.read_bytes()
            except OSError:
                continue
            match = FAST_LABEL_RE.search(content)
            if match:
                label = unescape(match.group(1).decode("utf-8", errors="replace")).strip()
                if label:
                    labels.append(label)
            if limit and len(labels) >= limit:
                break
        return labels
    if dataset_kind == "crohme23":
        for label in iter_crohme23_labels(root, split):
            labels.append(label)
            if limit and len(labels) >= limit:
                break
        return labels
    for sample in _samples(root, split, dataset_kind):
        if sample.label:
            labels.append(sample.label)
        if limit and len(labels) >= limit:
            break
    return labels


def _vocab(labels: list[str], vocab_limit: int, extra_tokens: list[str] | None = None) -> list[str]:
    counts: dict[str, int] = {}
    for label in labels:
        for token in tokenize_expression(label):
            counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    if vocab_limit:
        ordered = ordered[:vocab_limit]
    for token in extra_tokens or []:
        if token not in SPECIAL_TOKENS and token not in ordered:
            ordered.append(token)
    return ["<pad>", "<bos>", "<eos>", "<unk>", *ordered]


def _checkpoint_vocab(path: Path, torch: Any) -> list[str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return [str(token) for token in checkpoint.get("vocab", [])]


def _initialize_from_checkpoint(model: Any, path: Path, vocab: list[str], torch: Any) -> dict[str, int]:
    """Transfer compatible weights and token rows into an expanded model."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    old_vocab = [str(token) for token in checkpoint.get("vocab", [])]
    old_state = checkpoint.get("model", {})
    state = model.state_dict()
    copied_tensors = 0
    copied_token_rows = 0
    old_ids = {token: index for index, token in enumerate(old_vocab)}
    row_keys = {"embedding.weight", "output.weight", "output.bias"}
    for key, target in state.items():
        source = old_state.get(key)
        if source is None:
            continue
        if key in row_keys and source.ndim == target.ndim and source.shape[1:] == target.shape[1:]:
            for new_index, token in enumerate(vocab):
                old_index = old_ids.get(token)
                if old_index is not None and old_index < source.shape[0]:
                    target[new_index].copy_(source[old_index])
                    copied_token_rows += 1
            continue
        if source.shape == target.shape:
            target.copy_(source)
            copied_tensors += 1
    model.load_state_dict(state)
    return {"copiedTensors": copied_tensors, "copiedTokenRows": copied_token_rows, "sourceVocabSize": len(old_vocab)}


def _manifest_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / f"{split}.json"


def _valid_manifest(path: Path, max_points: int, max_tokens: int, vocab: list[str], dataset_kind: str, feature_width: int = LEGACY_FEATURE_WIDTH, dataset_source_version: str = DATASET_SOURCE_VERSION) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        data.get("maxPoints") == max_points
        and data.get("maxTokens") == max_tokens
        and data.get("datasetKind") == dataset_kind
        and int(data.get("featureWidth", LEGACY_FEATURE_WIDTH)) == feature_width
        and data.get("datasetSourceVersion") == dataset_source_version
        and data.get("vocab") == vocab
        and int(data.get("samples", 0)) > 0
        and bool(data.get("shards"))
        and all((path.parent / shard["file"]).exists() for shard in data.get("shards", []))
    )


def _cached_vocab(cache_dir: Path, max_points: int, max_tokens: int, dataset_kind: str, feature_width: int = LEGACY_FEATURE_WIDTH, dataset_source_version: str = DATASET_SOURCE_VERSION) -> list[str] | None:
    """Reuse a compatible manifest without rescanning hundreds of thousands of XML files."""

    path = _manifest_path(cache_dir, "train")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    vocab = data.get("vocab")
    if (
        data.get("maxPoints") != max_points
        or data.get("maxTokens") != max_tokens
        or int(data.get("featureWidth", LEGACY_FEATURE_WIDTH)) != feature_width
        or data.get("datasetKind") != dataset_kind
        or data.get("datasetSourceVersion") != dataset_source_version
        or not isinstance(vocab, list)
        or not vocab
        or int(data.get("samples", 0)) <= 0
        or not data.get("shards")
        or not all((cache_dir / shard["file"]).exists() for shard in data.get("shards", []))
    ):
        return None
    return [str(token) for token in vocab]


def _build_cache(root: Path, cache_dir: Path, split: str, limit: int, max_points: int, max_tokens: int, vocab: list[str], shard_size: int, dataset_kind: str, feature_width: int = LEGACY_FEATURE_WIDTH, dataset_source_version: str = DATASET_SOURCE_VERSION) -> dict[str, Any]:
    manifest_path = _manifest_path(cache_dir, split)
    if _valid_manifest(manifest_path, max_points, max_tokens, vocab, dataset_kind, feature_width, dataset_source_version):
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    import numpy as np

    cache_dir.mkdir(parents=True, exist_ok=True)
    for old_shard in cache_dir.glob(f"{split}-*.npz"):
        old_shard.unlink()
    token_to_id = {token: index for index, token in enumerate(vocab)}
    feature_rows: list[list[list[float]]] = []
    feature_lengths: list[int] = []
    target_rows: list[list[int]] = []
    labels: list[str] = []
    shards: list[dict[str, Any]] = []
    sample_count = 0

    def flush() -> None:
        nonlocal feature_rows, feature_lengths, target_rows, labels, sample_count
        if not feature_rows:
            return
        shard_name = f"{split}-{len(shards):05d}.npz"
        features = np.zeros((len(feature_rows), max_points, feature_width), dtype=np.float16)
        targets = np.full((len(feature_rows), max_tokens), token_to_id["<pad>"], dtype=np.int32)
        for index, row in enumerate(feature_rows):
            features[index, : len(row)] = np.asarray(row, dtype=np.float16)
            token_row = target_rows[index]
            targets[index, : len(token_row)] = np.asarray(token_row, dtype=np.int32)
        np.savez(
            cache_dir / shard_name,
            features=features,
            feature_lengths=np.asarray(feature_lengths, dtype=np.int32),
            targets=targets,
            labels=np.asarray(labels, dtype=object),
        )
        shards.append({"file": shard_name, "samples": len(feature_rows)})
        sample_count += len(feature_rows)
        feature_rows, feature_lengths, target_rows, labels = [], [], [], []

    for sample in _samples(root, split, dataset_kind):
        if not sample.label:
            continue
        row = _sample_features(sample, max_points, feature_width)
        feature_rows.append(row)
        feature_lengths.append(len(row))
        target_rows.append(_token_ids(sample.label, token_to_id, max_tokens))
        labels.append(sample.label)
        if len(feature_rows) >= shard_size:
            flush()
        if limit and sample_count + len(feature_rows) >= limit:
            break
    flush()
    manifest = {
        "version": 1,
        "split": split,
        "samples": sample_count,
        "maxPoints": max_points,
        "maxTokens": max_tokens,
        "featureWidth": feature_width,
        "datasetKind": dataset_kind,
        "datasetSourceVersion": dataset_source_version,
        "vocab": vocab,
        "shards": shards,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _cached_batches(manifest: dict[str, Any], cache_dir: Path, batch_size: int, shuffle: bool, rng: random.Random, np: Any):
    for shard in manifest["shards"]:
        with np.load(cache_dir / shard["file"], allow_pickle=True) as data:
            order = list(range(int(shard["samples"])))
            if shuffle:
                rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                yield data["features"][indices], data["feature_lengths"][indices], data["targets"][indices], data["labels"][indices]


def _evaluate_cached(model, manifest, cache_dir, batch_size, token_to_id, vocab, max_tokens, torch, pack, unpack, device, np):
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for features_np, lengths_np, _targets_np, labels_np in _cached_batches(manifest, cache_dir, batch_size, False, random.Random(0), np):
            features = torch.from_numpy(features_np).float().to(device)
            lengths = torch.from_numpy(lengths_np).long().to(device)
            predicted = model.greedy_decode(features, lengths, token_to_id["<bos>"], token_to_id["<eos>"], max_tokens, pack, unpack)
            for label, prediction in zip(labels_np.tolist(), predicted.cpu().tolist()):
                predicted_text = normalize_operator_scripts(_decode_ids(prediction, vocab))
                reference_tokens = tokenize_expression(str(label))
                hypothesis_tokens = tokenize_expression(predicted_text) if predicted_text else []
                distance = edit_distance(reference_tokens, hypothesis_tokens)
                rows.append({
                    "target": str(label),
                    "prediction": predicted_text,
                    "tokenErrorRate": distance / max(1, len(reference_tokens)),
                    "exactMatch": str(label) == predicted_text,
                })
    return {
        "samples": len(rows),
        "exactMatchRate": sum(row["exactMatch"] for row in rows) / max(1, len(rows)),
        "meanTokenErrorRate": sum(row["tokenErrorRate"] for row in rows) / max(1, len(rows)),
        "results": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess and train on full MathWriting using disk-backed shards.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-kind", choices=("mathwriting", "mathwriting-synthetic", "mathwriting-symbols", "crohme", "crohme23", "synthetic-piecewise"), default="mathwriting")
    parser.add_argument("--dataset-source-version", default=DATASET_SOURCE_VERSION, help="Provenance identifier written into cache manifests")
    parser.add_argument("--init-checkpoint", type=Path, help="Optional checkpoint used for transfer learning")
    parser.add_argument("--vocab-checkpoint", type=Path, help="Reuse the vocabulary from a checkpoint without rescanning labels")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/mathwriting-cache"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/expression-torch-full.pt"))
    parser.add_argument("--report-out", type=Path, default=Path("artifacts/expression-torch-full.json"))
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--valid-limit", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-points", type=int, default=384)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--feature-width", type=int, choices=(LEGACY_FEATURE_WIDTH, GEOMETRY_FEATURE_WIDTH), default=LEGACY_FEATURE_WIDTH)
    parser.add_argument("--vocab-limit", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cache-only", action="store_true", help="Build or reuse disk-backed shards without training a checkpoint")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch, nn, pack, unpack = _require_torch()
    import numpy as np

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    started = time.perf_counter()
    vocab = _cached_vocab(args.cache_dir, args.max_points, args.max_tokens, args.dataset_kind, args.feature_width, args.dataset_source_version)
    init_vocab = _checkpoint_vocab(args.init_checkpoint, torch) if args.init_checkpoint else []
    checkpoint_vocab = _checkpoint_vocab(args.vocab_checkpoint, torch) if args.vocab_checkpoint else []
    if vocab is not None and not args.vocab_limit:
        print(f"Reusing cached vocabulary with {len(vocab)} tokens.", flush=True)
    elif checkpoint_vocab and not args.vocab_limit:
        vocab = checkpoint_vocab
        print(f"Reusing checkpoint vocabulary with {len(vocab)} tokens.", flush=True)
    else:
        print("Scanning training labels to build the complete vocabulary...", flush=True)
        train_labels = _labels(args.dataset_root, "train", args.train_limit, args.dataset_kind)
        vocab = _vocab(train_labels, args.vocab_limit, init_vocab)
    print(f"Vocabulary: {len(vocab)} tokens; building disk-backed shards...", flush=True)
    train_manifest = _build_cache(args.dataset_root, args.cache_dir, "train", args.train_limit, args.max_points, args.max_tokens, vocab, args.shard_size, args.dataset_kind, args.feature_width, args.dataset_source_version)
    valid_manifest = _build_cache(args.dataset_root, args.cache_dir, "valid", args.valid_limit, args.max_points, args.max_tokens, vocab, args.shard_size, args.dataset_kind, args.feature_width, args.dataset_source_version)
    print(f"Cached {train_manifest['samples']} train and {valid_manifest['samples']} validation expressions.", flush=True)

    if args.cache_only:
        print(json.dumps({
            "datasetKind": args.dataset_kind,
            "trainSamples": train_manifest["samples"],
            "validSamples": valid_manifest["samples"],
            "cacheDir": str(args.cache_dir),
            "vocabSize": len(vocab),
        }, indent=2), flush=True)
        return 0

    token_to_id = {token: index for index, token in enumerate(vocab)}
    model = _build_model(len(vocab), args.hidden_size, torch, nn, args.dropout, args.feature_width).to(device)
    initialization = None
    if args.init_checkpoint:
        initialization = _initialize_from_checkpoint(model, args.init_checkpoint, vocab, torch)
        print(f"Initialized from {args.init_checkpoint}: {initialization}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_id["<pad>"], label_smoothing=args.label_smoothing)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    rng = random.Random(args.seed)
    history = []
    evaluation = {"samples": 0, "exactMatchRate": 0.0, "meanTokenErrorRate": 0.0, "results": []}
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for features_np, lengths_np, targets_np, _labels_np in _cached_batches(train_manifest, args.cache_dir, args.batch_size, True, rng, np):
            features = torch.from_numpy(features_np).float().to(device)
            lengths = torch.from_numpy(lengths_np).long().to(device)
            targets = torch.from_numpy(targets_np).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with amp_context:
                logits = model(features, lengths, targets[:, :-1], pack, unpack)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            batches += 1
            if batches % 250 == 0:
                print(f"epoch {epoch + 1}/{args.epochs} batch {batches} loss {total_loss / batches:.4f}", flush=True)
        epoch_record = {"epoch": epoch + 1, "meanLoss": total_loss / max(1, batches), "batches": batches}
        evaluation = _evaluate_cached(model, valid_manifest, args.cache_dir, args.batch_size, token_to_id, vocab, args.max_tokens, torch, pack, unpack, device, np)
        epoch_record["validationExactMatchRate"] = evaluation["exactMatchRate"]
        epoch_record["validationMeanTokenErrorRate"] = evaluation["meanTokenErrorRate"]
        history.append(epoch_record)
        torch.save({
            "model": model.state_dict(),
            "vocab": vocab,
            "config": {"hiddenSize": args.hidden_size, "maxPoints": args.max_points, "maxTokens": args.max_tokens, "featureWidth": args.feature_width, "dropout": args.dropout, "modelVersion": "0.3"},
            "epoch": epoch + 1,
            "history": history,
        }, args.output)
        print(f"epoch {epoch + 1} complete: mean loss {epoch_record['meanLoss']:.4f}, validation token error {evaluation['meanTokenErrorRate']:.4f}, exact match {evaluation['exactMatchRate']:.4f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "vocab": vocab,
        "config": {"hiddenSize": args.hidden_size, "maxPoints": args.max_points, "maxTokens": args.max_tokens, "featureWidth": args.feature_width, "dropout": args.dropout, "modelVersion": "0.3"},
        "epoch": args.epochs,
        "history": history,
    }, args.output)
    report = {
        "model": "mathwriting-expression-gru-attention-0.2-disk-backed",
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "trainSamples": train_manifest["samples"],
        "validSamples": valid_manifest["samples"],
        "datasetKind": args.dataset_kind,
        "initCheckpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "initialization": initialization,
        "vocabSize": len(vocab),
        "history": history,
        "evaluation": evaluation,
        "elapsedSeconds": round(time.perf_counter() - started, 2),
        "cacheDir": str(args.cache_dir),
        "checkpoint": str(args.output),
        "amp": amp_enabled,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "evaluation"}, indent=2), flush=True)
    print(json.dumps({key: evaluation[key] for key in ("samples", "exactMatchRate", "meanTokenErrorRate")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
