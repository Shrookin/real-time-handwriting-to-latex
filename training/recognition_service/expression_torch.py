"""A small trainable full-expression LaTeX recognizer.

This is an experiment runner, intentionally separate from the dependency-free
HTTP service. It uses the MathWriting stroke sequence directly and predicts a
token sequence, which gives us a measurable path beyond isolated-symbol KNN.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .evaluate import edit_distance
from .mathwriting import InkPoint, InkStroke, MathWritingSample, iter_inkml, tokenize_expression


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
LEGACY_FEATURE_WIDTH = 6
GEOMETRY_FEATURE_WIDTH = 10


@dataclass(frozen=True)
class Example:
    sample_id: str
    features: list[list[float]]
    target: str


def _sample_features(sample: MathWritingSample, max_points: int, feature_width: int = LEGACY_FEATURE_WIDTH) -> list[list[float]]:
    """Flatten an expression into point features and optional stroke geometry.

    Width six is the original checkpoint contract. Width ten appends the
    normalized left/top/right/bottom bounds of the current stroke to every
    point, giving a candidate model explicit 2D grouping information without
    changing the active baseline checkpoint.
    """

    if feature_width not in {LEGACY_FEATURE_WIDTH, GEOMETRY_FEATURE_WIDTH}:
        raise ValueError(f"Unsupported feature width: {feature_width}")

    points = [point for stroke in sample.strokes for point in stroke.points]
    if not points:
        return [[0.0] * feature_width]

    stride = max(1, math.ceil(len(points) / max_points))
    selected: list[tuple[int, int, float, float]] = []
    offset = 0
    for stroke_index, stroke in enumerate(sample.strokes):
        for point_index, point in enumerate(stroke.points):
            if point_index == 0 or (offset + point_index) % stride == 0:
                selected.append((stroke_index, point_index, point.x, point.y))
        offset += len(stroke.points)
    selected = selected[:max_points]

    xs = [row[2] for row in selected]
    ys = [row[3] for row in selected]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = max(max_x - min_x, max_y - min_y, 1.0)
    stroke_bounds = {}
    for stroke_index, stroke in enumerate(sample.strokes):
        stroke_xs = [point.x for point in stroke.points]
        stroke_ys = [point.y for point in stroke.points]
        if stroke_xs and stroke_ys:
            stroke_bounds[stroke_index] = (min(stroke_xs), min(stroke_ys), max(stroke_xs), max(stroke_ys))

    result: list[list[float]] = []
    previous_x = selected[0][2]
    previous_y = selected[0][3]
    previous_stroke = selected[0][0]
    for stroke_index, _point_index, x, y in selected:
        is_new_stroke = 1.0 if stroke_index != previous_stroke else 0.0
        row = [
            (x - min_x) / scale * 2.0 - 1.0,
            (y - min_y) / scale * 2.0 - 1.0,
            (x - previous_x) / scale,
            (y - previous_y) / scale,
            is_new_stroke,
            1.0 - is_new_stroke,
        ]
        if feature_width == GEOMETRY_FEATURE_WIDTH:
            stroke_min_x, stroke_min_y, stroke_max_x, stroke_max_y = stroke_bounds.get(stroke_index, (x, y, x, y))
            row.extend([
                (stroke_min_x - min_x) / scale * 2.0 - 1.0,
                (stroke_min_y - min_y) / scale * 2.0 - 1.0,
                (stroke_max_x - min_x) / scale * 2.0 - 1.0,
                (stroke_max_y - min_y) / scale * 2.0 - 1.0,
            ])
        result.append(row)
        previous_x, previous_y, previous_stroke = x, y, stroke_index
    return result


def request_features(strokes: list[dict[str, Any]], max_points: int, feature_width: int = LEGACY_FEATURE_WIDTH) -> list[list[float]]:
    """Convert the HTTP online-ink shape into the model's feature sequence."""

    sample_strokes = []
    for stroke in strokes:
        points = []
        for point in stroke.get("points", []):
            if "x" in point and "y" in point:
                points.append(InkPoint(float(point["x"]), float(point["y"]), float(point["t"]) if "t" in point else None))
        if points:
            sample_strokes.append(InkStroke(tuple(points)))
    sample = MathWritingSample("request", tuple(sample_strokes), {})
    return _sample_features(sample, max_points, feature_width)


def _load_examples(root: Path, split: str, limit: int, max_points: int, feature_width: int = LEGACY_FEATURE_WIDTH) -> list[Example]:
    examples: list[Example] = []
    for sample in iter_inkml(root, split):
        if sample.label:
            examples.append(Example(sample.sample_id, _sample_features(sample, max_points, feature_width), sample.label))
        if limit and len(examples) >= limit:
            break
    return examples


def _make_vocab(examples: Iterable[Example], vocab_limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for example in examples:
        for token in tokenize_expression(example.target):
            counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    if vocab_limit:
        ordered = ordered[:vocab_limit]
    return [*SPECIAL_TOKENS, *ordered]


def _token_ids(target: str, token_to_id: dict[str, int], max_tokens: int) -> list[int]:
    unknown = token_to_id["<unk>"]
    tokens = tokenize_expression(target)[: max(1, max_tokens - 2)]
    return [token_to_id["<bos>"], *(token_to_id.get(token, unknown) for token in tokens), token_to_id["<eos>"]]


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    except ImportError as error:  # pragma: no cover - exercised only without optional ML deps
        raise RuntimeError("Install the optional ML environment before running expression_torch") from error
    return torch, nn, pack_padded_sequence, pad_packed_sequence


def _build_model(vocab_size: int, hidden_size: int, torch: Any, nn: Any, dropout_rate: float = 0.1, feature_width: int = LEGACY_FEATURE_WIDTH):
    class ExpressionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.encoder = nn.GRU(feature_width, hidden_size, batch_first=True, bidirectional=True)
            self.init_decoder = nn.Linear(hidden_size * 2, hidden_size)
            # Additive (Bahdanau-style) attention is more expressive than the
            # previous implementation, which reduced the attention features by
            # summing them into one score.
            self.encoder_attention = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.query_attention = nn.Linear(hidden_size, hidden_size, bias=False)
            self.attention_score = nn.Linear(hidden_size, 1, bias=False)
            self.decoder = nn.GRU(hidden_size * 3, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, vocab_size)
            self.dropout = nn.Dropout(dropout_rate)

        def encode(self, features, lengths, pack_padded_sequence, pad_packed_sequence):
            packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
            encoded, state = self.encoder(packed)
            encoded, _ = pad_packed_sequence(encoded, batch_first=True)
            initial = torch.tanh(self.init_decoder(torch.cat([state[-2], state[-1]], dim=1))).unsqueeze(0)
            return encoded, initial

        def forward(self, features, lengths, decoder_input, pack_padded_sequence, pad_packed_sequence):
            encoded, state = self.encode(features, lengths, pack_padded_sequence, pad_packed_sequence)
            mask = torch.arange(encoded.size(1), device=features.device)[None, :] < lengths[:, None]
            encoded_keys = self.encoder_attention(encoded)
            outputs = []
            for step in range(decoder_input.size(1)):
                embedded = self.dropout(self.embedding(decoder_input[:, step])).unsqueeze(1)
                query = self.query_attention(state[-1]).unsqueeze(1)
                scores = self.attention_score(torch.tanh(encoded_keys + query)).squeeze(-1)
                scores = scores.masked_fill(~mask, -1e4)
                weights = scores.softmax(dim=1).unsqueeze(1)
                context = weights.bmm(encoded)
                decoder_input_step = torch.cat([embedded, context], dim=2)
                decoded, state = self.decoder(decoder_input_step, state)
                outputs.append(self.output(decoded))
            return torch.cat(outputs, dim=1)

        @torch.no_grad()
        def greedy_decode(self, features, lengths, bos_id, eos_id, max_tokens, pack_padded_sequence, pad_packed_sequence, return_scores=False):
            encoded, state = self.encode(features, lengths, pack_padded_sequence, pad_packed_sequence)
            mask = torch.arange(encoded.size(1), device=features.device)[None, :] < lengths[:, None]
            encoded_keys = self.encoder_attention(encoded)
            current = torch.full((features.size(0),), bos_id, dtype=torch.long, device=features.device)
            result = []
            scores_out = []
            finished = torch.zeros(features.size(0), dtype=torch.bool, device=features.device)
            for _ in range(max_tokens):
                embedded = self.embedding(current).unsqueeze(1)
                query = self.query_attention(state[-1]).unsqueeze(1)
                scores = self.attention_score(torch.tanh(encoded_keys + query)).squeeze(-1)
                scores = scores.masked_fill(~mask, -1e4)
                context = scores.softmax(dim=1).unsqueeze(1).bmm(encoded)
                decoded, state = self.decoder(torch.cat([embedded, context], dim=2), state)
                logits = self.output(decoded[:, 0])
                probabilities = logits.softmax(dim=1)
                current = probabilities.argmax(dim=1)
                result.append(current)
                scores_out.append(probabilities.gather(1, current.unsqueeze(1)).squeeze(1))
                finished |= current == eos_id
                if bool(finished.all()):
                    break
            ids = torch.stack(result, dim=1) if result else torch.empty((features.size(0), 0), dtype=torch.long, device=features.device)
            if return_scores:
                scores = torch.stack(scores_out, dim=1) if scores_out else torch.empty((features.size(0), 0), dtype=torch.float32, device=features.device)
                return ids, scores
            return ids

    return ExpressionModel()


def _collate(batch: list[Example], token_to_id: dict[str, int], max_tokens: int, torch: Any):
    feature_lengths = torch.tensor([len(item.features) for item in batch], dtype=torch.long)
    target_rows = [_token_ids(item.target, token_to_id, max_tokens) for item in batch]
    feature_width = len(batch[0].features[0])
    max_features = int(feature_lengths.max())
    max_target = max(len(row) for row in target_rows)
    features = torch.zeros((len(batch), max_features, feature_width), dtype=torch.float32)
    targets = torch.full((len(batch), max_target), token_to_id["<pad>"], dtype=torch.long)
    for index, item in enumerate(batch):
        features[index, : len(item.features)] = torch.tensor(item.features, dtype=torch.float32)
        targets[index, : len(target_rows[index])] = torch.tensor(target_rows[index], dtype=torch.long)
    return features, feature_lengths, targets


def _batches(examples: list[Example], batch_size: int, shuffle: bool, rng: random.Random):
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [examples[index] for index in order[start : start + batch_size]]


def _decode_ids(row: list[int], id_to_token: list[str]) -> str:
    tokens: list[str] = []
    for token_id in row:
        token = id_to_token[token_id]
        if token == "<eos>":
            break
        if token not in {"<pad>", "<bos>"}:
            tokens.append(token)
    return "".join(tokens)


def _evaluate(model, examples, token_to_id, id_to_token, args, torch, pack, unpack, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in _batches(examples, args.batch_size, False, random.Random(0)):
            features, lengths, targets = _collate(batch, token_to_id, args.max_tokens, torch)
            predicted = model.greedy_decode(features.to(device), lengths.to(device), token_to_id["<bos>"], token_to_id["<eos>"], args.max_tokens, pack, unpack)
            for item, prediction in zip(batch, predicted.cpu().tolist()):
                predicted_text = _decode_ids(prediction, id_to_token)
                reference_tokens = tokenize_expression(item.target)
                hypothesis_tokens = tokenize_expression(predicted_text) if predicted_text else []
                distance = edit_distance(reference_tokens, hypothesis_tokens)
                rows.append({
                    "sampleId": item.sample_id,
                    "target": item.target,
                    "prediction": predicted_text,
                    "tokenErrorRate": distance / max(1, len(reference_tokens)),
                    "exactMatch": item.target == predicted_text,
                })
    return {
        "samples": len(rows),
        "exactMatchRate": sum(row["exactMatch"] for row in rows) / max(1, len(rows)),
        "meanTokenErrorRate": sum(row["tokenErrorRate"] for row in rows) / max(1, len(rows)),
        "results": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small PyTorch full-expression MathWriting recognizer.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/expression-torch-smoke.pt"))
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--valid-limit", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-points", type=int, default=384)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--vocab-limit", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--feature-width", type=int, choices=(LEGACY_FEATURE_WIDTH, GEOMETRY_FEATURE_WIDTH), default=LEGACY_FEATURE_WIDTH)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch, nn, pack, unpack = _require_torch()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device(device_name)
    started = time.perf_counter()
    train = _load_examples(args.dataset_root, "train", args.train_limit, args.max_points, args.feature_width)
    valid = _load_examples(args.dataset_root, "valid", args.valid_limit, args.max_points, args.feature_width)
    vocab = _make_vocab(train, args.vocab_limit)
    token_to_id = {token: index for index, token in enumerate(vocab)}
    model = _build_model(len(vocab), args.hidden_size, torch, nn, args.dropout, args.feature_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_id["<pad>"], label_smoothing=args.label_smoothing)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    rng = random.Random(args.seed)
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in _batches(train, args.batch_size, True, rng):
            features, lengths, targets = _collate(batch, token_to_id, args.max_tokens, torch)
            decoder_input = targets[:, :-1].to(device)
            decoder_target = targets[:, 1:].to(device)
            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
            with amp_context:
                logits = model(features.to(device), lengths.to(device), decoder_input, pack, unpack)
                loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            batches += 1
        history.append({"epoch": epoch + 1, "meanLoss": total_loss / max(1, batches), "batches": batches})

    evaluation = _evaluate(model, valid, token_to_id, vocab, args, torch, pack, unpack, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "vocab": vocab,
        "config": {"hiddenSize": args.hidden_size, "maxPoints": args.max_points, "maxTokens": args.max_tokens, "featureWidth": args.feature_width, "modelVersion": "0.3"},
    }, args.output)
    report = {
        "model": f"mathwriting-expression-gru-attention-0.3-geometry{args.feature_width}",
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "trainSamples": len(train),
        "validSamples": len(valid),
        "vocabSize": len(vocab),
        "history": history,
        "evaluation": evaluation,
        "elapsedSeconds": round(time.perf_counter() - started, 2),
        "checkpoint": str(args.output),
        "amp": amp_enabled,
        "featureWidth": args.feature_width,
    }
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "evaluation"}, indent=2))
    print(json.dumps({key: evaluation[key] for key in ("samples", "exactMatchRate", "meanTokenErrorRate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
