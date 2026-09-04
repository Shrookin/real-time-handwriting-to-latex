"""Optional PyTorch checkpoint loader for the local recognition service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .expression_torch import LEGACY_FEATURE_WIDTH, _build_model, _decode_ids, _require_torch, request_features
from .expression import normalize_stroke_order
from .piecewise_layout import compose_piecewise_latex, detect_piecewise_layout, score_piecewise_relations


def normalize_operator_scripts(latex: str) -> str:
    """Undo a narrow decoder/layout error without changing real exponents.

    A baseline operator can be emitted as a one-token script when its
    handwritten stroke is slightly above or below the baseline. This is
    especially common for ``+``, ``-``, and relation operators in handwritten
    expressions. Expressions such as ``x^{-1}`` are left alone because the
    script contains more than a standalone operator token.
    """

    # A single operator is almost always a baseline operator when it is drawn
    # slightly high or low. Keep this deliberately allow-listed: flattening
    # every one-token script would destroy legitimate exponents/subscripts
    # such as x^2, x^i, or a_{n}. Accept both braced and unbraced forms.
    operator = r"(?:[+<>=-]|\\(?:pm|mp|times|cdot|div|neq|ne|le|leq|ge|geq|approx|sim|in|notin|to|rightarrow|leftarrow))"
    def flatten(match: re.Match[str]) -> str:
        token = match.group("braced") or match.group("plain") or ""
        # TeX control words consume following letters, so preserve a small
        # delimiter when a command operator is moved back to the baseline.
        return f"{token} " if token.startswith("\\") else token

    return re.sub(
        rf"(?:\^|_)(?:\{{(?P<braced>{operator})\}}|(?P<plain>{operator}))",
        flatten,
        latex,
    )


@dataclass
class ExpressionCheckpoint:
    model: Any
    vocab: list[str]
    torch: Any
    device: Any
    max_points: int
    max_tokens: int
    feature_width: int
    model_version: str


def load_checkpoint(path: str | Path, device_name: str = "auto") -> ExpressionCheckpoint:
    torch, nn, _pack, _unpack = _require_torch()
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    vocab = [str(token) for token in checkpoint["vocab"]]
    requested = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    device = torch.device(requested)
    feature_width = int(config.get("featureWidth", LEGACY_FEATURE_WIDTH))
    model = _build_model(len(vocab), int(config.get("hiddenSize", 192)), torch, nn, float(config.get("dropout", 0.1)), feature_width)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    checkpoint_version = str(checkpoint.get("modelVersion", config.get("modelVersion", "0.2")))
    if checkpoint_version in {"0.1", "0.2"}:
        checkpoint_version = f"mathwriting-expression-gru-attention-{checkpoint_version}"
    # The training metadata intentionally keeps the architecture version
    # stable, but that is not enough to identify which checkpoint is live.
    # Include the artifact stem in the service-facing version so /health and
    # recognition responses prove which trained weights answered the request.
    checkpoint_version = f"{checkpoint_version}@{path.stem}"
    return ExpressionCheckpoint(
        model=model,
        vocab=vocab,
        torch=torch,
        device=device,
        max_points=int(config.get("maxPoints", 384)),
        max_tokens=int(config.get("maxTokens", 64)),
        feature_width=feature_width,
        model_version=checkpoint_version,
    )


def _recognize_sequence(checkpoint: ExpressionCheckpoint, strokes: list[dict[str, Any]]) -> dict[str, Any]:
    torch = checkpoint.torch
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    features = request_features(strokes, checkpoint.max_points, checkpoint.feature_width)
    tensor = torch.tensor(features, dtype=torch.float32, device=checkpoint.device).unsqueeze(0)
    lengths = torch.tensor([len(features)], dtype=torch.long, device=checkpoint.device)
    with torch.no_grad():
        ids, scores = checkpoint.model.greedy_decode(
            tensor,
            lengths,
            checkpoint.vocab.index("<bos>"),
            checkpoint.vocab.index("<eos>"),
            checkpoint.max_tokens,
            pack_padded_sequence,
            pad_packed_sequence,
            return_scores=True,
        )
    prediction = normalize_operator_scripts(_decode_ids(ids[0].cpu().tolist(), checkpoint.vocab))
    token_scores = scores[0].cpu().tolist()
    confidence = sum(token_scores) / max(1, len(token_scores))
    return {
        "latex": prediction,
        "display": prediction,
        "confidence": float(confidence),
        "alternatives": [],
        "symbols": [],
        "modelVersion": checkpoint.model_version,
        "inferenceDevice": str(checkpoint.device),
    }


def _recognize_piecewise(checkpoint: ExpressionCheckpoint, strokes: list[dict[str, Any]]) -> dict[str, Any] | None:
    layout = detect_piecewise_layout(strokes)
    if layout is None:
        return None
    rows: list[tuple[str, str | None]] = []
    confidences: list[float] = []
    for row in layout.rows:
        value_strokes = list(row.value_strokes)
        condition_strokes = list(row.condition_strokes)
        if condition_strokes:
            value_result = _recognize_sequence(checkpoint, value_strokes)
            condition_result = _recognize_sequence(checkpoint, condition_strokes)
            rows.append((str(value_result.get("latex", "")), str(condition_result.get("latex", ""))))
            confidences.extend([float(value_result.get("confidence", 0.0)), float(condition_result.get("confidence", 0.0))])
        else:
            row_result = _recognize_sequence(checkpoint, list(row.strokes))
            rows.append((str(row_result.get("latex", "")), None))
            confidences.append(float(row_result.get("confidence", 0.0)))
    latex = compose_piecewise_latex(rows)
    if not latex:
        return None
    relation_evidence = score_piecewise_relations(rows)
    structural_evidence = min(1.0, float(layout.score) + float(relation_evidence["scoreBoost"]))
    return {
        "latex": latex,
        "display": latex,
        "confidence": (min(confidences) * 0.85) if confidences else 0.0,
        "alternatives": [],
        "symbols": [],
        "modelVersion": checkpoint.model_version + "+piecewise-layout-1",
        "inferenceDevice": str(checkpoint.device),
        "piecewiseEvidence": structural_evidence,
        "piecewiseBraceOrientation": layout.brace_orientation,
        "piecewiseClosingBraces": layout.closing_braces,
        "piecewiseRelationEvidence": relation_evidence,
    }


def recognize(checkpoint: ExpressionCheckpoint, strokes: list[dict[str, Any]]) -> dict[str, Any]:
    """Recognize an expression using a scored ordinary/piecewise hypothesis."""

    prepared_strokes = normalize_stroke_order(strokes)
    sequence = _recognize_sequence(checkpoint, prepared_strokes)
    piecewise = _recognize_piecewise(checkpoint, prepared_strokes)
    if piecewise is None:
        return sequence

    evidence = float(piecewise.get("piecewiseEvidence", 0.0))
    piecewise_confidence = float(piecewise.get("confidence", 0.0))
    sequence_confidence = float(sequence.get("confidence", 0.0))
    # Strong brace-and-row evidence can select the specialist even when the
    # whole-expression decoder is more confident on generic symbols. For
    # ambiguous layouts, require the specialist to be competitive with the
    # ordinary hypothesis instead of forcing every brace into cases.
    relation_evidence = piecewise.get("piecewiseRelationEvidence", {})
    repeated_conditions = int(relation_evidence.get("relationRows", 0)) >= 2
    strong_layout = evidence >= 0.78 or (repeated_conditions and evidence >= 0.7)
    competitive = piecewise_confidence >= max(0.05, sequence_confidence * 0.72)
    if strong_layout or (evidence >= 0.58 and competitive):
        return piecewise
    return sequence
