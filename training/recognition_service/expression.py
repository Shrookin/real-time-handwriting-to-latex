"""Geometry-first composition of isolated MathWriting symbol predictions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Sequence

from .symbol_model import predict


LAYOUT_VERSION = "layout-1"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass
class SymbolCandidate:
    strokes: list[dict[str, Any]]
    bounds: Box
    latex: str
    display: str
    confidence: float
    alternatives: list[dict[str, Any]]


def _point_xy(point: Any) -> tuple[float, float] | None:
    if not isinstance(point, dict) or "x" not in point or "y" not in point:
        return None
    return float(point["x"]), float(point["y"])


def stroke_bounds(stroke: dict[str, Any]) -> Box | None:
    points = [_point_xy(point) for point in stroke.get("points", [])]
    valid = [point for point in points if point is not None]
    if not valid:
        return None
    xs = [point[0] for point in valid]
    ys = [point[1] for point in valid]
    return Box(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def bounds_for_strokes(strokes: Sequence[dict[str, Any]]) -> Box:
    boxes = [stroke_bounds(stroke) for stroke in strokes]
    valid = [box for box in boxes if box is not None]
    if not valid:
        return Box(0, 0, 0, 0)
    left = min(box.x for box in valid)
    top = min(box.y for box in valid)
    right = max(box.right for box in valid)
    bottom = max(box.bottom for box in valid)
    return Box(left, top, right - left, bottom - top)


def _nearby(first: Box, second: Box, threshold: float) -> bool:
    x_gap = max(first.x - second.right, second.x - first.right, 0.0)
    y_gap = max(first.y - second.bottom, second.y - first.bottom, 0.0)
    x_overlap = min(first.right, second.right) - max(first.x, second.x)
    y_overlap = min(first.bottom, second.bottom) - max(first.y, second.y)
    # Touching/crossing strokes are one symbol. A small gap is also allowed
    # for dots, crossbars, and multi-stroke handwritten glyphs.
    return (
        (x_gap <= threshold and y_overlap >= -threshold * 0.35)
        or (y_gap <= threshold and x_overlap >= -threshold * 0.35)
    )


def group_strokes(strokes: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group strokes into likely symbols using connected-component geometry."""

    usable = [stroke for stroke in strokes if stroke_bounds(stroke) is not None]
    if len(usable) < 2:
        return [usable] if usable else []
    boxes = [stroke_bounds(stroke) for stroke in usable]
    sizes = [max(box.width, box.height, 1.0) for box in boxes if box is not None]
    threshold = max(8.0, min(28.0, median(sizes) * 0.32))
    parents = list(range(len(usable)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first in range(len(usable)):
        for second in range(first + 1, len(usable)):
            if boxes[first] is not None and boxes[second] is not None and _nearby(boxes[first], boxes[second], threshold):
                union(first, second)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, stroke in enumerate(usable):
        grouped.setdefault(find(index), []).append(stroke)
    return sorted(grouped.values(), key=lambda group: bounds_for_strokes(group).x)


def normalize_stroke_order(strokes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair only a clear anchor-first/backward component jump.

    The expression checkpoint learned the native MathWriting trace order, so
    sorting every request by x-coordinate would damage ordinary writing and
    multi-stroke symbols.  When a user writes a large anchor first and then
    adds a separate component far to its left, however, the trace order has a
    strong backward jump. In that narrow case, preserve stroke order inside
    each connected component and order the components geometrically.
    """

    original = list(strokes)
    if len(original) < 3:
        return original
    groups = group_strokes(original)
    if len(groups) < 3 or sum(len(group) for group in groups) != len(original):
        return original

    input_index = {id(stroke): index for index, stroke in enumerate(original)}
    trace_groups = sorted(groups, key=lambda group: min(input_index[id(stroke)] for stroke in group))
    sizes = [max(bounds_for_strokes(group).width, bounds_for_strokes(group).height, 1.0) for group in groups]
    ordered_sizes = sorted(sizes)
    median_size = ordered_sizes[len(ordered_sizes) // 2]
    tolerance = max(24.0, median_size * 0.9)
    largest_backward_gap = max(
        (bounds_for_strokes(trace_groups[index - 1]).x - bounds_for_strokes(trace_groups[index]).x
         for index in range(1, len(trace_groups))),
        default=0.0,
    )
    if largest_backward_gap < tolerance:
        return original
    return [stroke for group in groups for stroke in group]


def _is_horizontal_bar(candidate: SymbolCandidate) -> bool:
    return candidate.latex in {"-", "−", r"\minus"} and (
        candidate.bounds.width >= max(12.0, candidate.bounds.height * 2.4)
        or candidate.bounds.height <= 2.0
    )


_BASELINE_OPERATOR_TOKENS = {
    "+", "-", "−", r"\minus", "=", r"\pm", r"\mp", r"\times", r"\cdot",
    r"\div", r"\neq", r"\le", r"\leq", r"\ge", r"\geq", r"\approx",
    r"\sim", r"\in", r"\notin", r"\to", r"\rightarrow", r"\leftarrow",
}


def _is_baseline_operator(candidate: SymbolCandidate) -> bool:
    """Operators should not become scripts because their stroke is off-center."""

    return candidate.latex.strip().lower() in _BASELINE_OPERATOR_TOKENS


def _within_x(candidate: SymbolCandidate, bar: SymbolCandidate) -> bool:
    padding = max(6.0, bar.bounds.width * 0.2)
    return bar.bounds.x - padding <= candidate.bounds.center_x <= bar.bounds.right + padding


def _assemble_linear(candidates: Sequence[SymbolCandidate], baseline: float, scale: float) -> str:
    ordered = sorted(candidates, key=lambda candidate: (candidate.bounds.x, candidate.bounds.y))
    consumed: set[int] = set()
    parts: list[str] = []
    previous: SymbolCandidate | None = None
    threshold = max(8.0, scale * 0.35)

    for index, candidate in enumerate(ordered):
        if index in consumed:
            continue

        if _is_horizontal_bar(candidate):
            above = [
                (other_index, other)
                for other_index, other in enumerate(ordered)
                if other_index != index
                and other_index not in consumed
                and _within_x(other, candidate)
                and other.bounds.bottom <= candidate.bounds.y + threshold
            ]
            below = [
                (other_index, other)
                for other_index, other in enumerate(ordered)
                if other_index != index
                and other_index not in consumed
                and _within_x(other, candidate)
                and other.bounds.y >= candidate.bounds.bottom - threshold
            ]
            if above and below:
                above_candidates = [other for _, other in above]
                below_candidates = [other for _, other in below]
                parts.append(
                    r"\frac{" + _assemble_linear(above_candidates, baseline, scale) + "}{"
                    + _assemble_linear(below_candidates, baseline, scale)
                    + "}"
                )
                consumed.add(index)
                consumed.update(other_index for other_index, _ in above + below)
                previous = candidate
                continue

        token = candidate.latex
        if _is_baseline_operator(candidate):
            parts.append(token)
            # An operator separates bases. A later small symbol should not be
            # attached to the base on the other side of the operator.
            previous = None
            continue

        normalized = token.lower()
        next_candidate = ordered[index + 1] if index + 1 < len(ordered) else None
        if normalized in {r"\sqrt", "√"} and next_candidate is not None:
            if next_candidate.bounds.x >= candidate.bounds.x - threshold and next_candidate.bounds.x <= candidate.bounds.right + scale * 1.8:
                token = r"\sqrt{" + next_candidate.latex + "}"
                consumed.add(index + 1)

        is_small = candidate.bounds.height <= max(8.0, scale * 0.82)
        is_right_of_previous = previous is not None and candidate.bounds.x >= previous.bounds.right - threshold
        if previous is not None and is_small and is_right_of_previous:
            if candidate.bounds.bottom < baseline - max(4.0, scale * 0.18):
                parts.append("^{" + token + "}")
            elif candidate.bounds.y > previous.bounds.y + previous.bounds.height * 0.35:
                parts.append("_{" + token + "}")
            else:
                parts.append(token)
        else:
            parts.append(token)
        previous = candidate

    return "".join(parts)


def assemble(candidates: Sequence[SymbolCandidate]) -> str:
    """Assemble recognized symbols into a conservative LaTeX expression."""

    if not candidates:
        return ""
    heights = [candidate.bounds.height for candidate in candidates if candidate.bounds.height > 0]
    base_candidates = [candidate for candidate in candidates if not _is_baseline_operator(candidate) and not _is_horizontal_bar(candidate) and candidate.bounds.height > 0]
    base_heights = [candidate.bounds.height for candidate in base_candidates]
    scale = median(base_heights or heights) if (base_heights or heights) else max(candidate.bounds.width for candidate in candidates)
    baseline_candidates = [candidate for candidate in base_candidates if candidate.bounds.height >= max(4.0, scale * 0.62)]
    baseline = median(candidate.bounds.bottom for candidate in (baseline_candidates or base_candidates or candidates))
    return _assemble_linear(candidates, baseline, max(scale, 1.0))


def _expression_alternatives(candidates: Sequence[SymbolCandidate], primary: str) -> list[dict[str, Any]]:
    alternatives: list[dict[str, Any]] = []
    ordered = sorted(candidates, key=lambda candidate: candidate.bounds.x)
    for index, candidate in enumerate(ordered):
        for alternative in candidate.alternatives:
            replacement = alternative.get("latex", "")
            if not replacement:
                continue
            changed = list(ordered)
            changed[index] = SymbolCandidate(
                strokes=candidate.strokes,
                bounds=candidate.bounds,
                latex=replacement,
                display=alternative.get("display", replacement),
                confidence=float(alternative.get("confidence", 0.0)),
                alternatives=[],
            )
            latex = assemble(changed)
            if latex and latex != primary and latex not in {item["latex"] for item in alternatives}:
                alternatives.append({"latex": latex, "display": latex, "confidence": float(alternative.get("confidence", 0.0))})
            if len(alternatives) >= 3:
                return alternatives
    return alternatives


def recognize_expression(model: dict[str, Any], strokes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Recognize a region by segmenting it, classifying symbols, then composing layout."""

    groups = group_strokes(strokes)
    candidates: list[SymbolCandidate] = []
    for group in groups:
        result = predict(model, group)
        candidates.append(
            SymbolCandidate(
                strokes=group,
                bounds=bounds_for_strokes(group),
                latex=result.get("latex", ""),
                display=result.get("display", result.get("latex", "")),
                confidence=float(result.get("confidence", 0.0)),
                alternatives=result.get("alternatives", []),
            )
        )
    candidates.sort(key=lambda candidate: (candidate.bounds.x, candidate.bounds.y))
    latex = assemble(candidates)
    confidence = min((candidate.confidence for candidate in candidates), default=0.0)
    symbols = [
        {
            "latex": candidate.latex,
            "display": candidate.display,
            "confidence": candidate.confidence,
            "bounds": {
                "x": candidate.bounds.x,
                "y": candidate.bounds.y,
                "width": candidate.bounds.width,
                "height": candidate.bounds.height,
            },
            "alternatives": candidate.alternatives,
        }
        for candidate in candidates
    ]
    return {
        "latex": latex,
        "display": latex,
        "confidence": confidence,
        "alternatives": _expression_alternatives(candidates, latex),
        "symbols": symbols,
        "modelVersion": f"{model.get('modelVersion', 'unknown')}+{LAYOUT_VERSION}",
    }
