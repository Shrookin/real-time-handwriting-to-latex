"""Mixed MathWriting/CROHME/CROHME23/piecewise replay training.

This runner reuses the disk-backed feature caches from both datasets. CROHME
is interleaved with MathWriting instead of replacing it, which gives the model
new writer and symbol coverage while reducing catastrophic forgetting. CROHME23
and piecewise caches can be interleaved as additional replay sources.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

from .evaluate import edit_distance
from .expression_full_train import _initialize_from_checkpoint
from .expression_torch import GEOMETRY_FEATURE_WIDTH, LEGACY_FEATURE_WIDTH, _build_model, _decode_ids, _require_torch, _token_ids
from .expression_inference import normalize_operator_scripts
from .mathwriting import tokenize_expression


def _manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("shards"):
        raise ValueError(f"Cache manifest has no shards: {path}")
    return data


def _validate_cache(manifest: dict[str, Any], cache_dir: Path) -> None:
    for shard in manifest["shards"]:
        if not (cache_dir / shard["file"]).exists():
            raise FileNotFoundError(cache_dir / shard["file"])


def _batches(
    manifest: dict[str, Any],
    cache_dir: Path,
    batch_size: int,
    token_to_id: dict[str, int],
    max_tokens: int,
    np: Any,
    rng: random.Random,
    shuffle: bool,
    limit: int = 0,
    length_bucket_size: int = 16,
) -> Iterable[tuple[Any, Any, Any]]:
    seen = 0
    for shard in manifest["shards"]:
        with np.load(cache_dir / shard["file"], allow_pickle=True) as data:
            order = list(range(int(shard["samples"])))
            if shuffle:
                rng.shuffle(order)
            if limit:
                remaining = max(0, limit - seen)
                order = order[:remaining]

            # Group nearby sequence lengths before forming a batch. The model
            # uses packed GRUs, but its attention tensors are still padded to
            # the longest sequence in each batch. Randomly mixing very short
            # and 384-point expressions creates many different CUDA allocation
            # sizes and eventually fragments the allocator during long runs.
            # Sorting only inside small shuffled pools preserves approximate
            # stochastic training while making the allocation sizes stable.
            pool_size = max(batch_size, batch_size * max(1, length_bucket_size))
            batched_order: list[int] = []
            for pool_start in range(0, len(order), pool_size):
                pool = order[pool_start : pool_start + pool_size]
                if length_bucket_size > 1:
                    pool.sort(key=lambda index: int(data["feature_lengths"][index]))
                batched_order.extend(pool)

            for start in range(0, len(batched_order), batch_size):
                indices = batched_order[start : start + batch_size]
                labels = data["labels"][indices].tolist()
                targets = np.full((len(indices), max_tokens), token_to_id["<pad>"], dtype=np.int64)
                for row_index, label in enumerate(labels):
                    token_row = _token_ids(str(label), token_to_id, max_tokens)
                    targets[row_index, : len(token_row)] = np.asarray(token_row, dtype=np.int64)
                yield data["features"][indices], data["feature_lengths"][indices], targets
                seen += len(indices)
                if limit and seen >= limit:
                    return


def _batch_count(samples: int, batch_size: int) -> int:
    return (samples + batch_size - 1) // batch_size


def _evaluate(model: Any, manifest: dict[str, Any], cache_dir: Path, batch_size: int, token_to_id: dict[str, int], vocab: list[str], max_tokens: int, torch: Any, pack: Any, unpack: Any, device: Any, np: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for shard in manifest["shards"]:
            with np.load(cache_dir / shard["file"], allow_pickle=True) as data:
                labels = data["labels"].tolist()
                for start in range(0, len(labels), batch_size):
                    indices = list(range(start, min(start + batch_size, len(labels))))
                    features = torch.from_numpy(data["features"][indices]).float().to(device)
                    lengths = torch.from_numpy(data["feature_lengths"][indices]).long().to(device)
                    predicted = model.greedy_decode(features, lengths, token_to_id["<bos>"], token_to_id["<eos>"], max_tokens, pack, unpack)
                    for row_index, prediction in enumerate(predicted.cpu().tolist()):
                        target = str(labels[indices[row_index]])
                        output = normalize_operator_scripts(_decode_ids(prediction, vocab))
                        reference_tokens = tokenize_expression(target)
                        hypothesis_tokens = tokenize_expression(output) if output else []
                        distance = edit_distance(reference_tokens, hypothesis_tokens)
                        rows.append({"target": target, "prediction": output, "exactMatch": target == output, "tokenErrorRate": distance / max(1, len(reference_tokens))})
    return {
        "samples": len(rows),
        "exactMatchRate": sum(row["exactMatch"] for row in rows) / max(1, len(rows)),
        "meanTokenErrorRate": sum(row["tokenErrorRate"] for row in rows) / max(1, len(rows)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train with MathWriting replay and CROHME online-ink data.")
    parser.add_argument("--mathwriting-cache-dir", type=Path, required=True)
    parser.add_argument("--crohme-cache-dir", type=Path, required=True)
    parser.add_argument("--crohme23-cache-dir", type=Path)
    parser.add_argument("--piecewise-cache-dir", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--mathwriting-limit", type=int, default=50000)
    parser.add_argument("--crohme-repeat", type=int, default=1)
    parser.add_argument("--mathwriting-per-crohme", type=int, default=4)
    parser.add_argument("--crohme23-repeat", type=int, default=1)
    parser.add_argument("--mathwriting-per-crohme23", type=int, default=8)
    parser.add_argument("--piecewise-per-mathwriting", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--feature-width", type=int, choices=(LEGACY_FEATURE_WIDTH, GEOMETRY_FEATURE_WIDTH))
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--length-bucket-size", type=int, default=16, help="Shuffled pool size in batches for length-aware bucketing.")
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=False, help="Use PyTorch's native recurrent implementation instead of cuDNN during training.")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _cuda_memory_snapshot(torch: Any, device: Any) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "allocatedMiB": round(torch.cuda.memory_allocated(device) / (1024 * 1024)),
        "reservedMiB": round(torch.cuda.memory_reserved(device) / (1024 * 1024)),
        "peakAllocatedMiB": round(torch.cuda.max_memory_allocated(device) / (1024 * 1024)),
        "peakReservedMiB": round(torch.cuda.max_memory_reserved(device) / (1024 * 1024)),
    }


def main() -> int:
    args = build_parser().parse_args()
    torch, nn, pack, unpack = _require_torch()
    import numpy as np

    math_train = _manifest(args.mathwriting_cache_dir / "train.json")
    math_valid = _manifest(args.mathwriting_cache_dir / "valid.json")
    crohme_train = _manifest(args.crohme_cache_dir / "train.json")
    crohme_valid = _manifest(args.crohme_cache_dir / "valid.json")
    crohme23_train = _manifest(args.crohme23_cache_dir / "train.json") if args.crohme23_cache_dir else None
    crohme23_valid = _manifest(args.crohme23_cache_dir / "valid.json") if args.crohme23_cache_dir else None
    piecewise_train = _manifest(args.piecewise_cache_dir / "train.json") if args.piecewise_cache_dir else None
    piecewise_valid = _manifest(args.piecewise_cache_dir / "valid.json") if args.piecewise_cache_dir else None
    cache_pairs = [(math_train, args.mathwriting_cache_dir), (math_valid, args.mathwriting_cache_dir), (crohme_train, args.crohme_cache_dir), (crohme_valid, args.crohme_cache_dir)]
    if crohme23_train is not None and crohme23_valid is not None:
        cache_pairs.extend(((crohme23_train, args.crohme23_cache_dir), (crohme23_valid, args.crohme23_cache_dir)))
    if piecewise_train is not None and piecewise_valid is not None:
        cache_pairs.extend(((piecewise_train, args.piecewise_cache_dir), (piecewise_valid, args.piecewise_cache_dir)))
    for manifest, directory in cache_pairs:
        _validate_cache(manifest, directory)

    cache_widths = {int(manifest.get("featureWidth", LEGACY_FEATURE_WIDTH)) for manifest, _directory in cache_pairs}
    if len(cache_widths) != 1:
        raise ValueError(f"Mixed replay caches use incompatible feature widths: {sorted(cache_widths)}")
    feature_width = args.feature_width if args.feature_width is not None else next(iter(cache_widths))
    if feature_width not in cache_widths:
        raise ValueError(f"Requested feature width {feature_width} does not match replay caches {sorted(cache_widths)}")

    vocab_sources = [math_train["vocab"], crohme_train["vocab"]]
    if crohme23_train is not None:
        vocab_sources.append(crohme23_train["vocab"])
    if piecewise_train is not None:
        vocab_sources.append(piecewise_train["vocab"])
    vocab = list(dict.fromkeys(token for source in vocab_sources for token in source))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    model = _build_model(len(vocab), args.hidden_size, torch, nn, args.dropout, feature_width).to(device)
    initialization = _initialize_from_checkpoint(model, args.init_checkpoint, vocab, torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_id["<pad>"], label_smoothing=args.label_smoothing)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

    def train_batch(features_np: Any, lengths_np: Any, targets_np: Any) -> float:
        """Run one update and release all batch-local GPU references."""

        features = torch.from_numpy(features_np).float().to(device)
        lengths = torch.from_numpy(lengths_np).long().to(device)
        targets = torch.from_numpy(targets_np).long().to(device)
        optimizer.zero_grad(set_to_none=True)
        amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
        with amp_context:
            logits = model(features, lengths, targets[:, :-1], pack, unpack)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1))
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach().cpu())
        del scaled_loss, logits, loss, targets, lengths, features
        return loss_value

    math_limit = min(args.mathwriting_limit, int(math_train["samples"])) if args.mathwriting_limit else int(math_train["samples"])
    math_steps = _batch_count(math_limit, args.batch_size)
    crohme_steps = _batch_count(int(crohme_train["samples"]), args.batch_size)
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        rng = random.Random(args.seed + epoch)
        bucket_size = max(1, args.length_bucket_size)
        math_iter = iter(_batches(math_train, args.mathwriting_cache_dir, args.batch_size, token_to_id, args.max_tokens, np, rng, True, math_limit, bucket_size))
        crohme_iter = iter(_batches(crohme_train, args.crohme_cache_dir, args.batch_size, token_to_id, args.max_tokens, np, rng, True, 0, bucket_size))
        crohme23_iter = iter(_batches(crohme23_train, args.crohme23_cache_dir, args.batch_size, token_to_id, args.max_tokens, np, rng, True, 0, bucket_size)) if crohme23_train is not None else None
        piecewise_iter = iter(_batches(piecewise_train, args.piecewise_cache_dir, args.batch_size, token_to_id, args.max_tokens, np, rng, True, 0, bucket_size)) if piecewise_train is not None else None
        total_loss = 0.0
        batches = 0
        crohme_batches = 0
        crohme23_batches = 0
        piecewise_batches = 0
        for math_step in range(math_steps):
            batch = next(math_iter, None)
            if batch is None:
                break
            for source, features_np, lengths_np, targets_np in (("mathwriting", *batch),):
                del source
                total_loss += train_batch(features_np, lengths_np, targets_np)
                batches += 1
            if (math_step + 1) % max(1, args.mathwriting_per_crohme) == 0:
                for _ in range(args.crohme_repeat):
                    crohme_batch = next(crohme_iter, None)
                    if crohme_batch is None:
                        crohme_iter = iter(_batches(crohme_train, args.crohme_cache_dir, args.batch_size, token_to_id, args.max_tokens, np, rng, True, 0, bucket_size))
                        crohme_batch = next(crohme_iter)
                    features_np, lengths_np, targets_np = crohme_batch
                    total_loss += train_batch(features_np, lengths_np, targets_np)
                    batches += 1
                    crohme_batches += 1
            if crohme23_iter is not None and args.mathwriting_per_crohme23 and (math_step + 1) % args.mathwriting_per_crohme23 == 0:
                for _ in range(args.crohme23_repeat):
                    crohme23_batch = next(crohme23_iter, None)
                    if crohme23_batch is None:
                        crohme23_iter = iter(_batches(crohme23_train, args.crohme23_cache_dir, args.batch_size, token_to_id, args.max_tokens, np, rng, True, 0, bucket_size))
                        crohme23_batch = next(crohme23_iter)
                    features_np, lengths_np, targets_np = crohme23_batch
                    total_loss += train_batch(features_np, lengths_np, targets_np)
                    batches += 1
                    crohme23_batches += 1
            if piecewise_iter is not None and args.piecewise_per_mathwriting and (math_step + 1) % args.piecewise_per_mathwriting == 0:
                piecewise_batch = next(piecewise_iter, None)
                if piecewise_batch is None:
                    piecewise_iter = iter(_batches(piecewise_train, args.piecewise_cache_dir, args.batch_size, token_to_id, args.max_tokens, np, rng, True, 0, bucket_size))
                    piecewise_batch = next(piecewise_iter)
                features_np, lengths_np, targets_np = piecewise_batch
                total_loss += train_batch(features_np, lengths_np, targets_np)
                batches += 1
                piecewise_batches += 1
            if batches % 250 == 0:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    memory = _cuda_memory_snapshot(torch, device)
                    torch.cuda.reset_peak_memory_stats(device)
                    memory_text = " " + " ".join(f"{key}={value}MiB" for key, value in memory.items())
                else:
                    memory = None
                    memory_text = ""
                print(f"epoch {epoch + 1}/{args.epochs} batch {batches} loss {total_loss / batches:.4f} crohmeBatches {crohme_batches} piecewiseBatches {piecewise_batches}{memory_text}", flush=True)
        math_evaluation = _evaluate(model, math_valid, args.mathwriting_cache_dir, args.batch_size, token_to_id, vocab, args.max_tokens, torch, pack, unpack, device, np)
        crohme_evaluation = _evaluate(model, crohme_valid, args.crohme_cache_dir, args.batch_size, token_to_id, vocab, args.max_tokens, torch, pack, unpack, device, np)
        crohme23_evaluation = _evaluate(model, crohme23_valid, args.crohme23_cache_dir, args.batch_size, token_to_id, vocab, args.max_tokens, torch, pack, unpack, device, np) if crohme23_valid is not None else None
        piecewise_evaluation = _evaluate(model, piecewise_valid, args.piecewise_cache_dir, args.batch_size, token_to_id, vocab, args.max_tokens, torch, pack, unpack, device, np) if piecewise_valid is not None else None
        record = {"epoch": epoch + 1, "meanLoss": total_loss / max(1, batches), "batches": batches, "crohmeBatches": crohme_batches, "crohme23Batches": crohme23_batches, "piecewiseBatches": piecewise_batches, "mathwritingValidation": math_evaluation, "crohmeValidation": crohme_evaluation, "crohme23Validation": crohme23_evaluation, "piecewiseValidation": piecewise_evaluation}
        history.append(record)
        # Persist a progress report before the checkpoint save as well as at
        # the end of the run. This keeps metrics available if a long-running
        # wrapper is detached immediately after an epoch completes.
        progress_report = {"model": "mathwriting-crohme-crohme23-piecewise-mixed-replay-0.3", "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None, "cudnnEnabled": torch.backends.cudnn.enabled if device.type == "cuda" else None, "disableCudnn": args.disable_cudnn, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "featureWidth": feature_width, "vocabSize": len(vocab), "mathwritingTrainSamples": math_limit, "crohmeTrainSamples": crohme_train["samples"], "crohme23TrainSamples": crohme23_train["samples"] if crohme23_train is not None else 0, "piecewiseTrainSamples": piecewise_train["samples"] if piecewise_train is not None else 0, "mathwritingPerCrohme": args.mathwriting_per_crohme, "mathwritingPerCROHME23": args.mathwriting_per_crohme23, "piecewisePerMathWriting": args.piecewise_per_mathwriting, "crohmeRepeat": args.crohme_repeat, "crohme23Repeat": args.crohme23_repeat, "initCheckpoint": str(args.init_checkpoint), "initialization": initialization, "history": history, "elapsedSeconds": round(time.perf_counter() - started, 2), "checkpoint": str(args.output)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(progress_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        torch.save({"model": model.state_dict(), "vocab": vocab, "config": {"hiddenSize": args.hidden_size, "maxPoints": math_train["maxPoints"], "maxTokens": args.max_tokens, "featureWidth": feature_width, "dropout": args.dropout, "modelVersion": "0.3"}, "epoch": epoch + 1, "history": history}, args.output)
        piecewise_text = f", piecewise exact {piecewise_evaluation['exactMatchRate']:.4f}" if piecewise_evaluation is not None else ""
        crohme23_text = f", CROHME23 exact {crohme23_evaluation['exactMatchRate']:.4f}" if crohme23_evaluation is not None else ""
        print(f"epoch {epoch + 1} complete: mean loss {record['meanLoss']:.4f}, MathWriting exact {math_evaluation['exactMatchRate']:.4f}, CROHME exact {crohme_evaluation['exactMatchRate']:.4f}{crohme23_text}{piecewise_text}", flush=True)

    report = {"model": "mathwriting-crohme-crohme23-piecewise-mixed-replay-0.3", "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None, "cudnnEnabled": torch.backends.cudnn.enabled if device.type == "cuda" else None, "disableCudnn": args.disable_cudnn, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "featureWidth": feature_width, "vocabSize": len(vocab), "mathwritingTrainSamples": math_limit, "crohmeTrainSamples": crohme_train["samples"], "crohme23TrainSamples": crohme23_train["samples"] if crohme23_train is not None else 0, "piecewiseTrainSamples": piecewise_train["samples"] if piecewise_train is not None else 0, "mathwritingPerCrohme": args.mathwriting_per_crohme, "mathwritingPerCROHME23": args.mathwriting_per_crohme23, "piecewisePerMathWriting": args.piecewise_per_mathwriting, "crohmeRepeat": args.crohme_repeat, "crohme23Repeat": args.crohme23_repeat, "initCheckpoint": str(args.init_checkpoint), "initialization": initialization, "history": history, "elapsedSeconds": round(time.perf_counter() - started, 2), "checkpoint": str(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
