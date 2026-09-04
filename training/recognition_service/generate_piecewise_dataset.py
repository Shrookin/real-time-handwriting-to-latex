"""Generate a targeted online-ink piecewise-expression dataset.

There is no sufficiently large public corpus of online handwritten piecewise
functions in the current workspace.  This generator fills that gap with
synthetic layout examples while reusing human-written MathWriting symbol
strokes.  The result is ordinary InkML, so the existing expression cache and
training runners can consume it without a second data contract.

The generated labels are deliberately canonical ``cases`` expressions.  The
model should learn the structural signal first; alternate LaTeX spellings can
be added after the layout behavior is stable.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

from .mathwriting import InkPoint, InkStroke, MathWritingSample, read_inkml


INKML_NAMESPACE = "http://www.w3.org/2003/InkML"
ElementTree.register_namespace("", INKML_NAMESPACE)


@dataclass(frozen=True)
class Glyph:
    label: str
    strokes: tuple[InkStroke, ...]


@dataclass(frozen=True)
class Atom:
    label: str
    superscript: bool = False


@dataclass(frozen=True)
class RowTemplate:
    value_label: str
    value_atoms: tuple[Atom, ...]
    condition_label: str
    condition_atoms: tuple[Atom, ...]


_VALUE_TEMPLATES = (
    ("x", (Atom("x"),)),
    ("-x", (Atom("-"), Atom("x"))),
    ("x+1", (Atom("x"), Atom("+"), Atom("1"))),
    ("x-1", (Atom("x"), Atom("-"), Atom("1"))),
    ("x^2", (Atom("x"), Atom("2", True))),
    ("x^2+1", (Atom("x"), Atom("2", True), Atom("+"), Atom("1"))),
    ("0", (Atom("0"),)),
    ("1", (Atom("1"),)),
    ("a", (Atom("a"),)),
    ("b", (Atom("b"),)),
    ("a+b", (Atom("a"), Atom("+"), Atom("b"))),
)

_CONDITION_TEMPLATES = (
    ("x<0", (Atom("x"), Atom("<"), Atom("0"))),
    (r"x\ge0", (Atom("x"), Atom(r"\ge"), Atom("0"))),
    ("x>1", (Atom("x"), Atom(">"), Atom("1"))),
    (r"x\le0", (Atom("x"), Atom(r"\le"), Atom("0"))),
    ("-1<x", (Atom("-"), Atom("1"), Atom("<"), Atom("x"))),
    (r"0\le x", (Atom("0"), Atom(r"\le"), Atom("x"))),
    ("x=0", (Atom("x"), Atom("="), Atom("0"))),
    ("x=1", (Atom("x"), Atom("="), Atom("1"))),
)


def _point_dict(point: InkPoint) -> dict[str, float]:
    return {"x": point.x, "y": point.y, **({"t": point.t} if point.t is not None else {})}


def _bounds(strokes: Iterable[InkStroke]) -> tuple[float, float, float, float]:
    points = [point for stroke in strokes for point in stroke.points]
    if not points:
        return 0.0, 0.0, 1.0, 1.0
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _transform_glyph(glyph: Glyph, left: float, baseline: float, height: float, jitter: random.Random, superscript: bool) -> list[dict[str, list[dict[str, float]]]]:
    min_x, min_y, max_x, max_y = _bounds(glyph.strokes)
    source_height = max(max_y - min_y, 1.0)
    scale = height / source_height
    x_jitter = jitter.uniform(-2.0, 2.0)
    y_jitter = jitter.uniform(-2.0, 2.0)
    top = baseline - height + y_jitter - (24.0 if superscript else 0.0)
    result = []
    for stroke in glyph.strokes:
        points = []
        for point in stroke.points:
            points.append({
                "x": left + x_jitter + (point.x - min_x) * scale,
                "y": top + (point.y - min_y) * scale,
                **({"t": point.t} if point.t is not None else {}),
            })
        if points:
            result.append({"points": points})
    return result


def _glyph_width(glyph: Glyph, height: float) -> float:
    min_x, min_y, max_x, max_y = _bounds(glyph.strokes)
    return max(12.0, (max_x - min_x) * height / max(max_y - min_y, 1.0))


def _render_atoms(atoms: tuple[Atom, ...], library: dict[str, tuple[Glyph, ...]], left: float, baseline: float, jitter: random.Random) -> tuple[list[dict], float]:
    strokes: list[dict] = []
    cursor = left
    for atom in atoms:
        options = library.get(atom.label)
        if not options:
            raise KeyError(f"No glyph strokes available for {atom.label!r}")
        glyph = jitter.choice(options)
        height = 27.0 if atom.superscript else 42.0
        strokes.extend(_transform_glyph(glyph, cursor, baseline, height, jitter, atom.superscript))
        cursor += _glyph_width(glyph, height) + (5.0 if atom.superscript else 8.0)
    return strokes, cursor


def _brace_stroke(left: float, top: float, height: float, jitter: random.Random) -> dict:
    width = 22.0
    points = [
        (left + width, top),
        (left + 8.0, top + 5.0),
        (left + 2.0, top + height * 0.22),
        (left + width * 0.66, top + height * 0.37),
        (left + width * 0.66, top + height * 0.50),
        (left + 2.0, top + height * 0.64),
        (left + 8.0, top + height * 0.84),
        (left + width, top + height),
    ]
    return {"points": [{"x": x + jitter.uniform(-1.5, 1.5), "y": y + jitter.uniform(-1.5, 1.5)} for x, y in points]}


def _choose_row(rng: random.Random) -> RowTemplate:
    value_label, value_atoms = rng.choice(_VALUE_TEMPLATES)
    condition_label, condition_atoms = rng.choice(_CONDITION_TEMPLATES)
    return RowTemplate(value_label, value_atoms, condition_label, condition_atoms)


def render_piecewise_sample(sample_id: str, library: dict[str, tuple[Glyph, ...]], rng: random.Random, row_count: int | None = None) -> tuple[MathWritingSample, dict]:
    rows = [_choose_row(rng) for _ in range(row_count or rng.randint(2, 4))]
    top = 24.0
    row_gap = rng.uniform(48.0, 58.0)
    baseline = top + 42.0
    # Keep a deliberate visual gap after the case brace.  The detector is
    # conservative by design so that tall integrals are not misclassified.
    body_left = 92.0
    strokes = [_brace_stroke(24.0, top, row_gap * (len(rows) - 1) + 54.0, rng)]
    row_metadata = []
    for row_index, row in enumerate(rows):
        row_baseline = baseline + row_index * row_gap
        value_strokes, cursor = _render_atoms(row.value_atoms, library, body_left, row_baseline, rng)
        condition_left = cursor + rng.uniform(26.0, 42.0)
        condition_strokes, _ = _render_atoms(row.condition_atoms, library, condition_left, row_baseline, rng)
        value_indices = list(range(len(strokes), len(strokes) + len(value_strokes)))
        strokes.extend(value_strokes)
        condition_indices = list(range(len(strokes), len(strokes) + len(condition_strokes)))
        strokes.extend(condition_strokes)
        row_metadata.append({"valueStrokeIndices": value_indices, "conditionStrokeIndices": condition_indices})

    body = r"\\".join(f"{row.value_label} & {row.condition_label}" for row in rows)
    label = r"\begin{cases}" + body + r"\end{cases}"
    annotations = {
        "label": label,
        "normalizedLabel": label,
        "dataset": "piecewise-synthetic-v1",
        "layout": json.dumps({"delimiterStrokeIndex": 0, "rows": row_metadata}, separators=(",", ":")),
    }
    sample = MathWritingSample(sample_id, tuple(
        InkStroke(tuple(InkPoint(float(point["x"]), float(point["y"]), point.get("t")) for point in stroke["points"]))
        for stroke in strokes
    ), annotations)
    return sample, {"rows": row_metadata, "rowCount": len(rows)}


def load_glyph_library(symbols_root: Path, labels: Iterable[str]) -> dict[str, tuple[Glyph, ...]]:
    wanted = set(labels)
    found: dict[str, list[Glyph]] = {label: [] for label in wanted}
    for path in sorted(symbols_root.glob("*.inkml")):
        if all(found[label] for label in wanted):
            break
        try:
            sample = read_inkml(path)
        except (OSError, ValueError):
            continue
        if sample.label in found and sample.strokes:
            found[sample.label].append(Glyph(sample.label, sample.strokes))
    missing = sorted(label for label, options in found.items() if not options)
    if missing:
        raise RuntimeError(f"MathWriting symbol library is missing glyphs: {', '.join(missing)}")
    return {label: tuple(options) for label, options in found.items()}


def _write_inkml(path: Path, sample: MathWritingSample) -> None:
    root = ElementTree.Element(f"{{{INKML_NAMESPACE}}}ink")
    for annotation_type, value in sample.annotations.items():
        element = ElementTree.SubElement(root, f"{{{INKML_NAMESPACE}}}annotation", {"type": annotation_type})
        element.text = value
    for index, stroke in enumerate(sample.strokes):
        trace = ElementTree.SubElement(root, f"{{{INKML_NAMESPACE}}}trace", {"id": str(index)})
        trace.text = ",".join(
            f"{point.x:.3f} {point.y:.3f} {point.t:.3f}" if point.t is not None else f"{point.x:.3f} {point.y:.3f}"
            for point in stroke.points
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def generate_dataset(output_root: Path, symbols_root: Path, train_count: int, valid_count: int, seed: int, overwrite: bool = False) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_root}")
        for child in output_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    labels = [atom.label for _, atoms in _VALUE_TEMPLATES for atom in atoms] + [atom.label for _, atoms in _CONDITION_TEMPLATES for atom in atoms]
    library = load_glyph_library(symbols_root, labels)
    rng = random.Random(seed)
    report = {"dataset": "piecewise-synthetic-v1", "seed": seed, "train": train_count, "valid": valid_count, "symbols": {label: len(options) for label, options in library.items()}}
    for split, count in (("train", train_count), ("valid", valid_count)):
        split_root = output_root / split
        for index in range(count):
            sample_id = f"piecewise-{split}-{index:06d}"
            sample, _metadata = render_piecewise_sample(sample_id, library, rng)
            _write_inkml(split_root / f"{sample_id}.inkml", sample)
    (output_root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate targeted synthetic online-ink piecewise expressions.")
    parser.add_argument("--symbols-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=10000)
    parser.add_argument("--valid-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = generate_dataset(args.output_root, args.symbols_root, args.train_count, args.valid_count, args.seed, args.overwrite)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
