"""Small, dependency-free reader for the MathWriting InkML format.

The parser follows Google's official example reader but keeps the output in
plain Python structures so it can feed training scripts and the NewNotes
versioned online-ink contract without importing NumPy or a deep-learning stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
from xml.etree import ElementTree


INKML_NAMESPACE = "http://www.w3.org/2003/InkML"
SCHEMA_VERSION = 1
_COMMAND_RE = re.compile(r"\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)")


@dataclass(frozen=True)
class InkPoint:
    x: float
    y: float
    t: float | None = None


@dataclass(frozen=True)
class InkStroke:
    points: tuple[InkPoint, ...]


@dataclass(frozen=True)
class MathWritingSample:
    sample_id: str
    strokes: tuple[InkStroke, ...]
    annotations: dict[str, str]

    @property
    def label(self) -> str:
        return self.annotations.get("normalizedLabel") or self.annotations.get("label") or ""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_trace(text: str | None) -> InkStroke:
    points: list[InkPoint] = []
    for raw_point in (text or "").split(","):
        values = raw_point.split()
        if len(values) < 2:
            continue
        timestamp = float(values[2]) if len(values) >= 3 else None
        points.append(InkPoint(float(values[0]), float(values[1]), timestamp))
    return InkStroke(tuple(points))


def read_inkml(path: str | Path) -> MathWritingSample:
    """Read one MathWriting `.inkml` file."""

    source = Path(path)
    root = ElementTree.parse(source).getroot()
    annotations: dict[str, str] = {}
    strokes: list[InkStroke] = []

    for element in root:
        name = _local_name(element.tag)
        if name == "annotation":
            annotation_type = element.attrib.get("type") or "label"
            annotations[annotation_type] = (element.text or "").strip()
        elif name == "trace":
            strokes.append(_parse_trace(element.text))

    return MathWritingSample(source.stem, tuple(strokes), annotations)


def _point_dict(point: InkPoint) -> dict[str, float]:
    result: dict[str, float] = {"x": point.x, "y": point.y}
    if point.t is not None:
        result["t"] = point.t
    return result


def to_request(sample: MathWritingSample, *, region_id: str | None = None, revision: int = 1) -> dict[str, Any]:
    """Convert a sample into the frontend's versioned recognition request."""

    return {
        "schemaVersion": SCHEMA_VERSION,
        "regionId": region_id or f"mathwriting-{sample.sample_id}",
        "revision": revision,
        "strokes": [{"points": [_point_dict(point) for point in stroke.points]} for stroke in sample.strokes],
    }


def to_training_record(sample: MathWritingSample) -> dict[str, Any]:
    """Return request-shaped online ink with the normalized training target."""

    return {**to_request(sample), "target": sample.label, "sampleId": sample.sample_id}


def tokenize_expression(expression: str) -> list[str]:
    """Tokenize a MathWriting LaTeX label using the official conventions."""

    tokens: list[str] = []
    remaining = expression
    while remaining:
        if remaining[0] == "\\":
            match = _COMMAND_RE.match(remaining)
            if not match:
                raise ValueError(f"Unrecognized LaTeX command near: {remaining[:20]!r}")
            token = match.group(0)
        else:
            token = remaining[0]
        tokens.append(token)
        remaining = remaining[len(token):]
    return tokens


def iter_inkml(root: str | Path, split: str = "train"):
    """Yield samples in stable filename order from a dataset split."""

    split_dir = Path(root) / split
    for path in sorted(split_dir.glob("*.inkml")):
        yield read_inkml(path)
