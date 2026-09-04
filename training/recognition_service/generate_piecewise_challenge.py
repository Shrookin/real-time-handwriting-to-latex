"""Generate a held-out structural challenge set for piecewise expressions.

The existing ``piecewise-synthetic-v1`` data is useful for training, but it
must not also be the only evidence that the cases specialist generalizes.  The
challenge set deliberately uses a different renderer profile: global scale,
shear, row drift, uneven row spacing, tighter columns, and longer conditions.
It reuses isolated human glyphs only as visual primitives; the layouts and
sample IDs are disjoint from the training dataset.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .generate_piecewise_dataset import (
    Glyph,
    _bounds,
    _glyph_width,
    _write_inkml,
    load_glyph_library,
)
from .mathwriting import InkPoint, InkStroke, MathWritingSample


@dataclass(frozen=True)
class Atom:
    label: str
    superscript: bool = False


@dataclass(frozen=True)
class ChallengeRow:
    value: str
    value_atoms: tuple[Atom, ...]
    condition: str
    condition_atoms: tuple[Atom, ...]


_VALUES = (
    ("x", (Atom("x"),)),
    ("0", (Atom("0"),)),
    ("1", (Atom("1"),)),
    ("x+1", (Atom("x"), Atom("+"), Atom("1"))),
    ("x-1", (Atom("x"), Atom("-"), Atom("1"))),
    ("x^2", (Atom("x"), Atom("2", True))),
    ("a+b", (Atom("a"), Atom("+"), Atom("b"))),
    ("-x", (Atom("-"), Atom("x"))),
)

_CONDITIONS = (
    ("x<0", (Atom("x"), Atom("<"), Atom("0"))),
    (r"x\ge0", (Atom("x"), Atom(r"\ge"), Atom("0"))),
    ("x>1", (Atom("x"), Atom(">"), Atom("1"))),
    (r"x\le1", (Atom("x"), Atom(r"\le"), Atom("1"))),
    (r"-1<x", (Atom("-"), Atom("1"), Atom("<"), Atom("x"))),
    (r"0\le x", (Atom("0"), Atom(r"\le"), Atom("x"))),
    ("x=0", (Atom("x"), Atom("="), Atom("0"))),
    ("x=1", (Atom("x"), Atom("="), Atom("1"))),
)


def _render_glyph(glyph: Glyph, left: float, baseline: float, height: float, rng: random.Random, superscript: bool) -> list[dict]:
    min_x, min_y, max_x, max_y = _bounds(glyph.strokes)
    scale = height / max(1.0, max_y - min_y)
    top = baseline - height - (height * 0.44 if superscript else 0.0)
    strokes: list[dict] = []
    for source in glyph.strokes:
        points = []
        for point in source.points:
            points.append({
                "x": left + (point.x - min_x) * scale + rng.uniform(-1.2, 1.2),
                "y": top + (point.y - min_y) * scale + rng.uniform(-1.2, 1.2),
            })
        if points:
            strokes.append({"points": points})
    return strokes


def _render_atoms(atoms: tuple[Atom, ...], library: dict[str, tuple[Glyph, ...]], left: float, baseline: float, rng: random.Random, scale: float) -> tuple[list[dict], float]:
    cursor = left
    strokes: list[dict] = []
    for atom in atoms:
        glyph = rng.choice(library[atom.label])
        height = (30.0 if atom.superscript else 48.0) * scale
        strokes.extend(_render_glyph(glyph, cursor, baseline, height, rng, atom.superscript))
        cursor += _glyph_width(glyph, height) + ((5.0 if atom.superscript else 10.0) * scale)
    return strokes, cursor


def _brace(left: float, top: float, height: float, rng: random.Random) -> dict:
    width = 25.0
    points = [
        (left + width, top),
        (left + 8.0, top + height * 0.07),
        (left + 2.0, top + height * 0.25),
        (left + width * 0.66, top + height * 0.42),
        (left + width * 0.66, top + height * 0.50),
        (left + 2.0, top + height * 0.68),
        (left + 8.0, top + height * 0.93),
        (left + width, top + height),
    ]
    return {"points": [{"x": x + rng.uniform(-2.0, 2.0), "y": y + rng.uniform(-2.0, 2.0)} for x, y in points]}


def _choose_rows(rng: random.Random, row_count: int) -> tuple[ChallengeRow, ...]:
    rows: list[ChallengeRow] = []
    for index in range(row_count):
        value, value_atoms = rng.choice(_VALUES)
        condition, condition_atoms = rng.choice(_CONDITIONS)
        rows.append(ChallengeRow(value, value_atoms, condition, condition_atoms))
    return tuple(rows)


def render_challenge_sample(sample_id: str, library: dict[str, tuple[Glyph, ...]], rng: random.Random, family: str, row_count: int) -> tuple[MathWritingSample, dict]:
    rows = _choose_rows(rng, row_count)
    scale = rng.uniform(0.82, 1.28)
    row_gap = rng.uniform(48.0, 72.0) * scale
    if family == "tight-rows":
        row_gap = rng.uniform(42.0, 52.0) * scale
    elif family == "uneven-rows":
        row_gap = rng.uniform(55.0, 82.0) * scale
    top = 28.0
    baseline = top + 52.0 * scale
    brace_height = row_gap * (len(rows) - 1) + 62.0 * scale
    brace_left = rng.uniform(18.0, 55.0)
    body_left = brace_left + rng.uniform(58.0, 92.0) * scale
    strokes: list[dict] = [_brace(brace_left, top, brace_height, rng)]
    row_metadata: list[dict] = []
    cursor_baseline = baseline
    for row_index, row in enumerate(rows):
        row_scale = scale * (rng.uniform(0.86, 1.14) if family == "uneven-rows" else 1.0)
        if family == "slanted-baseline":
            cursor_baseline += rng.uniform(-7.0, 7.0)
        value_strokes, cursor = _render_atoms(row.value_atoms, library, body_left, cursor_baseline, rng, row_scale)
        column_gap = rng.uniform(22.0, 54.0) * row_scale
        if family == "tight-columns":
            column_gap = rng.uniform(12.0, 24.0) * row_scale
        condition_strokes, _ = _render_atoms(row.condition_atoms, library, cursor + column_gap, cursor_baseline, rng, row_scale)
        value_indices = list(range(len(strokes), len(strokes) + len(value_strokes)))
        strokes.extend(value_strokes)
        condition_indices = list(range(len(strokes), len(strokes) + len(condition_strokes)))
        strokes.extend(condition_strokes)
        row_metadata.append({"valueStrokeIndices": value_indices, "conditionStrokeIndices": condition_indices})
        if family == "uneven-rows":
            cursor_baseline += rng.uniform(0.82, 1.18) * row_gap
        else:
            cursor_baseline += row_gap

    body = r"\\".join(f"{row.value} & {row.condition}" for row in rows)
    label = r"\begin{cases}" + body + r"\end{cases}"
    annotations = {
        "label": label,
        "normalizedLabel": label,
        "dataset": "piecewise-challenge-v1",
        "challengeFamily": family,
        "layout": json.dumps({"delimiterStrokeIndex": 0, "rows": row_metadata}, separators=(",", ":")),
    }
    sample = MathWritingSample(
        sample_id,
        tuple(InkStroke(tuple(InkPoint(float(point["x"]), float(point["y"])) for point in stroke["points"])) for stroke in strokes),
        annotations,
    )
    return sample, {"family": family, "rowCount": row_count, "rows": row_metadata}


def generate_challenge(output_root: Path, symbols_root: Path, count: int, seed: int, overwrite: bool = False) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_root}")
        for child in output_root.iterdir():
            if child.is_dir():
                import shutil
                shutil.rmtree(child)
            else:
                child.unlink()
    labels = {atom.label for _label, atoms in _VALUES + _CONDITIONS for atom in atoms}
    library = load_glyph_library(symbols_root, labels)
    rng = random.Random(seed)
    families = ("wide-relations", "tight-rows", "tight-columns", "uneven-rows", "slanted-baseline")
    cases: list[dict] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        family = families[index % len(families)]
        row_count = 2 + ((index * 7 + rng.randrange(3)) % 4)
        sample_id = f"piecewise-challenge-{index:06d}"
        sample, metadata = render_challenge_sample(sample_id, library, rng, family, row_count)
        _write_inkml(output_root / f"{sample_id}.inkml", sample)
        cases.append({
            "source": "piecewise-challenge",
            "split": ".",
            "sampleId": sample_id,
            "target": sample.label,
            "categories": ["piecewise", family, f"rows-{row_count}"],
            "metadata": metadata,
        })
    manifest = {
        "benchmark": "piecewise-challenge-v1",
        "dataset": "piecewise-challenge-v1",
        "seed": seed,
        "samples": count,
        "families": {family: sum(case["categories"][1] == family for case in cases) for family in families},
        "rowCounts": {str(row_count): sum(case["categories"][2] == f"rows-{row_count}" for case in cases) for row_count in range(2, 6)},
        "cases": cases,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in manifest.items() if key != "cases"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a held-out piecewise structural challenge set.")
    parser.add_argument("--symbols-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(generate_challenge(args.output_root, args.symbols_root, args.count, args.seed, args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
