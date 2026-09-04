"""Geometry helpers for recognizing handwritten piecewise expressions.

The sequence model is good at reading a single expression, but a piecewise
expression is also a layout problem: one tall delimiter owns several rows and
each row has a value/condition relationship.  This module intentionally keeps
that structural decision separate from symbol decoding so it can be tested and
trained independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import median
from typing import Any, Sequence


_RELATION_RE = re.compile(r"(?:=|<|>|\\(?:le|leq|ge|geq|ne|neq|approx|sim|lt|gt))")
_RELATION_COMMANDS = {"le", "leq", "ge", "geq", "ne", "neq", "approx", "sim", "lt", "gt"}


@dataclass(frozen=True)
class StrokeBox:
    stroke: dict[str, Any]
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class PiecewiseRow:
    strokes: tuple[dict[str, Any], ...]
    value_strokes: tuple[dict[str, Any], ...]
    condition_strokes: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PiecewiseLayout:
    delimiter: tuple[dict[str, Any], ...]
    rows: tuple[PiecewiseRow, ...]
    score: float = 0.0
    brace_orientation: str = "ambiguous"
    closing_braces: int = 0


def _stroke_box(stroke: dict[str, Any]) -> StrokeBox | None:
    points = [point for point in stroke.get("points", []) if isinstance(point, dict) and "x" in point and "y" in point]
    if not points:
        return None
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    return StrokeBox(stroke, min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _boxes(strokes: Sequence[dict[str, Any]]) -> list[StrokeBox]:
    return [box for stroke in strokes if (box := _stroke_box(stroke)) is not None]


def _brace_orientation(stroke: dict[str, Any]) -> str:
    """Estimate whether a tall stroke opens left, closes right, or is unclear.

    A handwritten left case brace generally bows left in its middle section,
    while a right brace bows right.  This is only a structural signal; it is
    deliberately not used as a symbol classifier by itself.
    """

    points = [
        point
        for point in stroke.get("points", [])
        if isinstance(point, dict) and "x" in point and "y" in point
    ]
    if len(points) < 3:
        return "ambiguous"
    ys = [float(point["y"]) for point in points]
    y_min, y_max = min(ys), max(ys)
    height = y_max - y_min
    if height <= 1:
        return "ambiguous"

    bands: list[list[float]] = [[], [], []]
    for point in points:
        position = (float(point["y"]) - y_min) / height
        band = 0 if position < 0.3 else 2 if position > 0.7 else 1
        bands[band].append(float(point["x"]))
    if not all(bands):
        return "ambiguous"
    top = median(bands[0])
    middle_min = min(bands[1])
    middle_max = max(bands[1])
    bottom = median(bands[2])
    width = max(float(point["x"]) for point in points) - min(float(point["x"]) for point in points)
    threshold = max(1.5, width * 0.08)
    # Real stylus braces often spend only a few samples at the far left/right
    # turn.  Comparing extrema instead of band medians preserves that signal.
    if min(top, bottom) - middle_min >= threshold:
        return "opening"
    if middle_max - max(top, bottom) >= threshold:
        return "closing"
    return "ambiguous"


def _cluster_rows(boxes: list[StrokeBox], row_tolerance: float) -> list[list[StrokeBox]]:
    rows: list[list[StrokeBox]] = []
    for box in sorted(boxes, key=lambda item: (item.center_y, item.x)):
        selected: list[StrokeBox] | None = None
        best_distance = float("inf")
        for row in rows:
            center = sum(item.center_y for item in row) / len(row)
            distance = abs(box.center_y - center)
            vertical_overlap = min(box.y + box.height, max(item.y + item.height for item in row)) - max(box.y, min(item.y for item in row))
            # A small negative gap is still a real row boundary.  Treating
            # near-touching boxes as overlap caused a wide glyph in one row
            # to absorb the next row, especially for piecewise conditions.
            if distance <= row_tolerance or vertical_overlap >= 0:
                if distance < best_distance:
                    selected = row
                    best_distance = distance
        if selected is None:
            rows.append([box])
        else:
            selected.append(box)
    return sorted(rows, key=lambda row: min(item.y for item in row))


def _split_columns(row: list[StrokeBox], split_gap: float) -> tuple[list[StrokeBox], list[StrokeBox]]:
    ordered = sorted(row, key=lambda item: item.x)
    if len(ordered) < 3:
        return ordered, []
    gaps = [ordered[index + 1].x - ordered[index].right for index in range(len(ordered) - 1)]
    largest_index = max(range(len(gaps)), key=gaps.__getitem__)
    if gaps[largest_index] < split_gap:
        return ordered, []
    return ordered[: largest_index + 1], ordered[largest_index + 1 :]


def detect_piecewise_layout(strokes: Sequence[dict[str, Any]]) -> PiecewiseLayout | None:
    """Detect a brace-like delimiter and at least two aligned body rows.

    This is intentionally conservative.  A candidate must be tall and narrow,
    sit to the left of the body, and leave a meaningful gap before the body so
    ordinary integrals and summation limits are not treated as case braces.
    """

    boxes = _boxes(strokes)
    if len(boxes) < 4:
        return None
    nonzero_heights = [box.height for box in boxes if box.height > 1]
    nonzero_widths = [box.width for box in boxes if box.width > 1]
    if not nonzero_heights:
        return None
    typical_height = median(nonzero_heights)
    typical_width = median(nonzero_widths or [typical_height])
    candidates = [
        box
        for box in boxes
        if box.height >= max(28.0, typical_height * 1.65)
        and box.width <= max(typical_height * 0.72, typical_width * 1.8)
    ]
    if not candidates:
        return None
    # A right-closing brace is evidence against a cases layout. Prefer an
    # opening/ambiguous candidate when one exists, but keep the old geometry
    # fallback for low-resolution or stylus-sampled braces.
    delimiter = min(
        candidates,
        key=lambda item: (0 if _brace_orientation(item.stroke) != "closing" else 1, item.x, -item.height),
    )
    brace_orientation = _brace_orientation(delimiter.stroke)
    if brace_orientation == "closing":
        return None
    body = [box for box in boxes if box is not delimiter and box.x >= delimiter.right]
    if len(body) < 3:
        return None
    body_left = min(box.x for box in body)
    gap = body_left - delimiter.right
    if gap < max(12.0, typical_width * 1.25):
        return None
    row_tolerance = max(14.0, min(34.0, typical_height * 0.72))
    row_groups = _cluster_rows(body, row_tolerance)
    if len(row_groups) < 2:
        return None
    row_centers = [sum(item.center_y for item in row) / len(row) for row in row_groups]
    if max(row_centers) - min(row_centers) < max(22.0, typical_height * 0.9):
        return None
    body_right = max(box.right for box in body)
    delimiter_top = delimiter.y
    delimiter_bottom = delimiter.y + delimiter.height
    closing_braces = 0
    for box in candidates:
        if box is delimiter or box.x <= body_right:
            continue
        overlap = min(delimiter_bottom, box.y + box.height) - max(delimiter_top, box.y)
        overlap_ratio = overlap / max(1.0, min(delimiter.height, box.height))
        if overlap_ratio >= 0.55 and _brace_orientation(box.stroke) == "closing":
            closing_braces += 1
    split_gap = max(18.0, typical_width * 1.8)
    rows: list[PiecewiseRow] = []
    for row in row_groups:
        ordered = sorted(row, key=lambda item: item.x)
        values, conditions = _split_columns(ordered, split_gap)
        row_strokes = tuple(item.stroke for item in ordered)
        rows.append(PiecewiseRow(
            strokes=row_strokes,
            value_strokes=tuple(item.stroke for item in values),
            condition_strokes=tuple(item.stroke for item in conditions),
        ))
    # This score is a prior for the recognizer, not a probability.  Multiple
    # aligned rows and an opening brace strengthen the cases hypothesis;
    # aligned closing braces weaken it because they more often indicate a
    # grouped/set expression.
    score = 0.52
    if brace_orientation == "opening":
        score += 0.18
    elif brace_orientation == "ambiguous":
        score += 0.04
    score += min(0.18, 0.08 * max(0, len(rows) - 1))
    score -= min(0.36, 0.18 * closing_braces)
    score = max(0.0, min(1.0, score))
    if score < 0.5:
        return None
    return PiecewiseLayout(
        delimiter=(delimiter.stroke,),
        rows=tuple(rows),
        score=score,
        brace_orientation=brace_orientation,
        closing_braces=closing_braces,
    )


def compose_piecewise_latex(rows: Sequence[tuple[str, str | None]]) -> str:
    """Compose row predictions in value/condition order."""

    body: list[str] = []
    for value, condition in rows:
        value = value.strip()
        condition = condition.strip() if condition else ""
        if not value and not condition:
            continue
        body.append(value + (f" & {condition}" if condition else ""))
    return r"\begin{cases}" + r"\\".join(body) + r"\end{cases}" if body else ""


def score_piecewise_relations(rows: Sequence[tuple[str, str | None]]) -> dict[str, Any]:
    """Score relation-bearing condition columns as piecewise evidence.

    Relations are useful only when they repeat across rows.  A single equality
    is common in ordinary expressions and set-builder notation, so it receives
    no meaningful boost by itself.  Repeated conditions sharing a variable are
    much stronger evidence that the right column is a cases condition column.
    """

    conditions = [str(condition).strip() for _value, condition in rows if condition and str(condition).strip()]
    relation_conditions = [condition for condition in conditions if _RELATION_RE.search(condition)]
    symbols_by_row: list[set[str]] = []
    for condition in relation_conditions:
        symbols = set(re.findall(r"\\([A-Za-z]+)", condition))
        symbols.update(re.findall(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", condition))
        symbols.discard("")
        symbols.difference_update(_RELATION_COMMANDS)
        symbols_by_row.append(symbols)
    shared_symbols = sorted(set.intersection(*symbols_by_row)) if symbols_by_row else []
    relation_ratio = len(relation_conditions) / max(1, len(conditions))
    boost = 0.0
    if len(relation_conditions) >= 2:
        boost += min(0.18, 0.09 * len(relation_conditions))
        boost += 0.12 * relation_ratio
    if len(relation_conditions) >= 2 and shared_symbols:
        boost += 0.12
    return {
        "conditionRows": len(conditions),
        "relationRows": len(relation_conditions),
        "relationRatio": relation_ratio,
        "sharedConditionSymbols": shared_symbols,
        "scoreBoost": min(0.42, boost),
    }
