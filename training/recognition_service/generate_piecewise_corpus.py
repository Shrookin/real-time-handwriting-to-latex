"""Build a large, reproducible piecewise-expression corpus.

This corpus is intentionally separate from the real-writer evaluation sets.
It reuses human MathWriting glyph samples while varying expression structure,
row geometry, spacing, baseline drift, scale, and stroke noise.  That makes it
useful for teaching the recognizer the cases/rows/conditions grammar, but it
must not be described as a new-writer corpus: the glyphs are recombined
synthetically and do not preserve a writer's complete expression.

The output is ordinary InkML and can be consumed by the existing cache builder.
The manifest records the provenance and the exact generation seed so that a
future real-writer or licensed corpus can be added without mixing evaluation
data into training by accident.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
from typing import Iterable

from .generate_piecewise_dataset import Glyph, _bounds, _glyph_width, _write_inkml, load_glyph_library
from .mathwriting import InkPoint, InkStroke, MathWritingSample


@dataclass(frozen=True)
class Atom:
    label: str
    superscript: bool = False


@dataclass(frozen=True)
class RowTemplate:
    name: str
    value_label: str
    value_atoms: tuple[Atom, ...]
    condition_label: str
    condition_atoms: tuple[Atom, ...]


@dataclass(frozen=True)
class LayoutProfile:
    name: str
    row_gap: tuple[float, float]
    condition_gap: tuple[float, float]
    baseline_slope: tuple[float, float]
    scale: tuple[float, float]
    row_jitter: float
    brace_width: tuple[float, float]
    point_jitter: float


# Keep this grammar within the symbol set already used by the v1/v3 corpus.
# The extra combinations are where the useful structural diversity comes from;
# the source glyphs themselves remain human MathWriting samples.
_VALUE_TEMPLATES: tuple[RowTemplate, ...] = (
    RowTemplate("x", "x", (Atom("x"),), "x<0", (Atom("x"), Atom("<"), Atom("0"))),
    RowTemplate("negative-x", "-x", (Atom("-"), Atom("x")), "x>0", (Atom("x"), Atom(">"), Atom("0"))),
    RowTemplate("x-plus-one", "x+1", (Atom("x"), Atom("+"), Atom("1")), "x=0", (Atom("x"), Atom("="), Atom("0"))),
    RowTemplate("x-minus-one", "x-1", (Atom("x"), Atom("-"), Atom("1")), "x=1", (Atom("x"), Atom("="), Atom("1"))),
    RowTemplate("x-plus-two", "x+2", (Atom("x"), Atom("+"), Atom("2")), "x<1", (Atom("x"), Atom("<"), Atom("1"))),
    RowTemplate("x-minus-two", "x-2", (Atom("x"), Atom("-"), Atom("2")), "x>1", (Atom("x"), Atom(">"), Atom("1"))),
    RowTemplate("x-square", "x^2", (Atom("x"), Atom("2", True)), r"0\le x", (Atom("0"), Atom(r"\le"), Atom("x"))),
    RowTemplate("x-square-plus-one", "x^2+1", (Atom("x"), Atom("2", True), Atom("+"), Atom("1")), r"x\ge0", (Atom("x"), Atom(r"\ge"), Atom("0"))),
    RowTemplate("x-square-minus-one", "x^2-1", (Atom("x"), Atom("2", True), Atom("-"), Atom("1")), r"x\le1", (Atom("x"), Atom(r"\le"), Atom("1"))),
    RowTemplate("two-x", "2x", (Atom("2"), Atom("x")), "-1<x", (Atom("-"), Atom("1"), Atom("<"), Atom("x"))),
    RowTemplate("two-x-plus-one", "2x+1", (Atom("2"), Atom("x"), Atom("+"), Atom("1")), r"1\le x", (Atom("1"), Atom(r"\le"), Atom("x"))),
    RowTemplate("zero", "0", (Atom("0"),), "x<0", (Atom("x"), Atom("<"), Atom("0"))),
    RowTemplate("one", "1", (Atom("1"),), "x>0", (Atom("x"), Atom(">"), Atom("0"))),
    RowTemplate("two", "2", (Atom("2"),), r"x\le0", (Atom("x"), Atom(r"\le"), Atom("0"))),
    RowTemplate("a", "a", (Atom("a"),), r"x\ge0", (Atom("x"), Atom(r"\ge"), Atom("0"))),
    RowTemplate("b", "b", (Atom("b"),), "x=1", (Atom("x"), Atom("="), Atom("1"))),
    RowTemplate("a-plus-b", "a+b", (Atom("a"), Atom("+"), Atom("b")), "0<x", (Atom("0"), Atom("<"), Atom("x"))),
    RowTemplate("a-minus-b", "a-b", (Atom("a"), Atom("-"), Atom("b")), r"x\le1", (Atom("x"), Atom(r"\le"), Atom("1"))),
)

_CONDITION_TEMPLATES: tuple[tuple[str, tuple[Atom, ...]], ...] = (
    ("x<0", (Atom("x"), Atom("<"), Atom("0"))),
    ("x>0", (Atom("x"), Atom(">"), Atom("0"))),
    ("x=0", (Atom("x"), Atom("="), Atom("0"))),
    ("x=1", (Atom("x"), Atom("="), Atom("1"))),
    ("x<1", (Atom("x"), Atom("<"), Atom("1"))),
    ("x>1", (Atom("x"), Atom(">"), Atom("1"))),
    (r"x\le0", (Atom("x"), Atom(r"\le"), Atom("0"))),
    (r"x\le1", (Atom("x"), Atom(r"\le"), Atom("1"))),
    (r"x\ge0", (Atom("x"), Atom(r"\ge"), Atom("0"))),
    (r"x\ge1", (Atom("x"), Atom(r"\ge"), Atom("1"))),
    (r"-1<x", (Atom("-"), Atom("1"), Atom("<"), Atom("x"))),
    (r"0\le x", (Atom("0"), Atom(r"\le"), Atom("x"))),
    (r"1\le x", (Atom("1"), Atom(r"\le"), Atom("x"))),
)

_LAYOUT_PROFILES: tuple[LayoutProfile, ...] = (
    LayoutProfile("clean", (48.0, 58.0), (24.0, 38.0), (-0.01, 0.01), (0.90, 1.08), 2.0, (18.0, 24.0), 0.8),
    LayoutProfile("wide-columns", (50.0, 62.0), (42.0, 72.0), (-0.02, 0.02), (0.86, 1.12), 2.5, (18.0, 26.0), 1.0),
    LayoutProfile("tight-columns", (46.0, 56.0), (8.0, 22.0), (-0.015, 0.015), (0.88, 1.05), 2.0, (17.0, 23.0), 0.9),
    LayoutProfile("slanted-baseline", (48.0, 60.0), (22.0, 44.0), (-0.12, 0.12), (0.84, 1.10), 4.0, (18.0, 25.0), 1.4),
    LayoutProfile("uneven-rows", (42.0, 72.0), (18.0, 52.0), (-0.04, 0.04), (0.82, 1.14), 7.0, (17.0, 26.0), 1.6),
    LayoutProfile("compressed", (34.0, 46.0), (14.0, 32.0), (-0.03, 0.03), (0.78, 0.98), 3.5, (16.0, 22.0), 1.2),
    LayoutProfile("expanded", (62.0, 84.0), (28.0, 64.0), (-0.025, 0.025), (0.96, 1.20), 5.0, (20.0, 28.0), 1.5),
    LayoutProfile("drifting-columns", (44.0, 68.0), (18.0, 48.0), (-0.08, 0.08), (0.84, 1.12), 8.0, (18.0, 26.0), 1.8),
)


def _transform_glyph(glyph: Glyph, left: float, baseline: float, height: float, rng: random.Random, profile: LayoutProfile, superscript: bool, slope: float) -> list[InkStroke]:
    min_x, min_y, max_x, max_y = _bounds(glyph.strokes)
    scale = height / max(max_y - min_y, 1.0)
    x_jitter = rng.uniform(-2.0, 2.0)
    y_jitter = rng.uniform(-profile.row_jitter, profile.row_jitter)
    top = baseline - height + y_jitter - (height * 0.56 if superscript else 0.0)
    strokes: list[InkStroke] = []
    for stroke in glyph.strokes:
        points: list[InkPoint] = []
        for point in stroke.points:
            relative_x = (point.x - min_x) * scale
            points.append(InkPoint(
                left + x_jitter + relative_x,
                top + (point.y - min_y) * scale + slope * relative_x + rng.uniform(-profile.point_jitter, profile.point_jitter),
                point.t,
            ))
        if points:
            strokes.append(InkStroke(tuple(points)))
    return strokes


def _render_atoms(atoms: tuple[Atom, ...], library: dict[str, tuple[Glyph, ...]], left: float, baseline: float, rng: random.Random, profile: LayoutProfile, slope: float) -> tuple[list[InkStroke], float]:
    strokes: list[InkStroke] = []
    cursor = left
    for atom in atoms:
        options = library.get(atom.label)
        if not options:
            raise KeyError(f"No glyph strokes available for {atom.label!r}")
        glyph = rng.choice(options)
        height = 25.0 if atom.superscript else 42.0
        strokes.extend(_transform_glyph(glyph, cursor, baseline, height, rng, profile, atom.superscript, slope))
        cursor += _glyph_width(glyph, height) + (4.0 if atom.superscript else rng.uniform(6.0, 10.0))
    return strokes, cursor


def _brace(left: float, top: float, height: float, rng: random.Random, profile: LayoutProfile) -> InkStroke:
    width = rng.uniform(*profile.brace_width)
    points = (
        (left + width, top),
        (left + width * 0.35, top + height * 0.05),
        (left + 2.0, top + height * 0.23),
        (left + width * 0.62, top + height * 0.40),
        (left + width * 0.62, top + height * 0.51),
        (left + 2.0, top + height * 0.68),
        (left + width * 0.35, top + height * 0.92),
        (left + width, top + height),
    )
    return InkStroke(tuple(InkPoint(x + rng.uniform(-profile.point_jitter, profile.point_jitter), y + rng.uniform(-profile.point_jitter, profile.point_jitter)) for x, y in points))


def _choose_row(rng: random.Random) -> RowTemplate:
    value = rng.choice(_VALUE_TEMPLATES)
    condition_label, condition_atoms = rng.choice(_CONDITION_TEMPLATES)
    return RowTemplate(value.name, value.value_label, value.value_atoms, condition_label, condition_atoms)


def render_curated_sample(sample_id: str, library: dict[str, tuple[Glyph, ...]], rng: random.Random, row_count: int | None = None, profile: LayoutProfile | None = None) -> tuple[MathWritingSample, dict]:
    selected_profile = profile or rng.choice(_LAYOUT_PROFILES)
    rows = [_choose_row(rng) for _ in range(row_count or rng.randint(2, 6))]
    scale = rng.uniform(*selected_profile.scale)
    row_gap = rng.uniform(*selected_profile.row_gap) * scale
    top = 26.0
    baseline = top + 44.0 * scale
    slope = rng.uniform(*selected_profile.baseline_slope)
    brace_height = row_gap * (len(rows) - 1) + 52.0 * scale
    strokes: list[InkStroke] = [_brace(24.0, top, brace_height, rng, selected_profile)]
    row_metadata: list[dict] = []
    value_template_counts: Counter[str] = Counter()
    condition_template_counts: Counter[str] = Counter()
    body_left = 64.0 + rng.uniform(-4.0, 8.0)
    for row_index, row in enumerate(rows):
        row_baseline = baseline + row_index * row_gap + rng.uniform(-selected_profile.row_jitter, selected_profile.row_jitter)
        value_strokes, value_cursor = _render_atoms(row.value_atoms, library, body_left + rng.uniform(-4.0, 4.0), row_baseline, rng, selected_profile, slope)
        condition_left = value_cursor + rng.uniform(*selected_profile.condition_gap)
        condition_strokes, _ = _render_atoms(row.condition_atoms, library, condition_left, row_baseline, rng, selected_profile, slope)
        value_indices = list(range(len(strokes), len(strokes) + len(value_strokes)))
        strokes.extend(value_strokes)
        condition_indices = list(range(len(strokes), len(strokes) + len(condition_strokes)))
        strokes.extend(condition_strokes)
        row_metadata.append({"valueStrokeIndices": value_indices, "conditionStrokeIndices": condition_indices})
        value_template_counts[row.name] += 1
        condition_template_counts[row.condition_label] += 1

    body = r"\\".join(f"{row.value_label} & {row.condition_label}" for row in rows)
    label = r"\begin{cases}" + body + r"\end{cases}"
    annotations = {
        "label": label,
        "normalizedLabel": label,
        "dataset": "piecewise-curated-v2",
        "layoutProfile": selected_profile.name,
        "layout": json.dumps({"delimiterStrokeIndex": 0, "rows": row_metadata}, separators=(",", ":")),
    }
    sample = MathWritingSample(sample_id, tuple(strokes), annotations)
    return sample, {
        "profile": selected_profile.name,
        "rowCount": len(rows),
        "rows": row_metadata,
        "valueTemplates": dict(value_template_counts),
        "conditionTemplates": dict(condition_template_counts),
    }


def _clear_output(root: Path, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {root}")
        shutil.rmtree(root)


def generate_corpus(
    output_root: Path,
    symbols_root: Path,
    train_count: int,
    valid_count: int,
    test_count: int,
    seed: int,
    overwrite: bool = False,
    profile_names: tuple[str, ...] | None = None,
) -> dict:
    _clear_output(output_root, overwrite)
    labels = sorted({atom.label for template in _VALUE_TEMPLATES for atom in template.value_atoms} | {atom.label for _, atoms in _CONDITION_TEMPLATES for atom in atoms})
    library = load_glyph_library(symbols_root, labels)
    profiles_by_name = {profile.name: profile for profile in _LAYOUT_PROFILES}
    selected_profiles = tuple(profiles_by_name[name] for name in profile_names) if profile_names else _LAYOUT_PROFILES
    if not selected_profiles:
        raise ValueError("At least one layout profile is required")
    report: dict = {
        "dataset": "piecewise-curated-v2",
        "generatorVersion": 2,
        "seed": seed,
        "source": "MathWriting 2024 symbols",
        "sourceLicense": "CC BY-NC-SA 4.0; verify compatibility before redistribution",
        "writerDiversity": "synthetic-glyph-recombination, not full-expression writer diversity",
        "purpose": "training structural cases/rows/condition alignment",
        "evaluationIsolation": "Do not evaluate on this corpus; use a held-out real-writer split.",
        "selectedProfiles": [profile.name for profile in selected_profiles],
        "counts": {"train": train_count, "valid": valid_count, "test": test_count},
        "profiles": {profile.name: 0 for profile in _LAYOUT_PROFILES},
        "rowCounts": Counter(),
        "valueTemplates": Counter(),
        "conditionTemplates": Counter(),
        "symbols": {label: len(options) for label, options in library.items()},
    }
    for split, count, split_offset in (("train", train_count, 0), ("valid", valid_count, 1_000_000), ("test", test_count, 2_000_000)):
        split_rng = random.Random(seed + split_offset)
        for index in range(count):
            sample_id = f"piecewise-v2-{split}-{index:07d}"
            profile = split_rng.choice(selected_profiles)
            sample, metadata = render_curated_sample(sample_id, library, split_rng, profile=profile)
            _write_inkml(output_root / split / f"{sample_id}.inkml", sample)
            report["profiles"][metadata["profile"]] += 1
            report["rowCounts"][str(metadata["rowCount"])] += 1
            for name, value in metadata["valueTemplates"].items():
                report["valueTemplates"][name] += value
            for name, value in metadata["conditionTemplates"].items():
                report["conditionTemplates"][name] += value
    report["rowCounts"] = dict(report["rowCounts"])
    report["valueTemplates"] = dict(report["valueTemplates"])
    report["conditionTemplates"] = dict(report["conditionTemplates"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a large, provenance-tracked piecewise online-ink corpus.")
    parser.add_argument("--symbols-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=100000)
    parser.add_argument("--valid-count", type=int, default=10000)
    parser.add_argument("--test-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        choices=[profile.name for profile in _LAYOUT_PROFILES],
        help="Restrict generation to one or more layout profiles; repeat for multiple profiles.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = generate_corpus(
        args.output_root,
        args.symbols_root,
        args.train_count,
        args.valid_count,
        args.test_count,
        args.seed,
        args.overwrite,
        tuple(args.profiles) if args.profiles else None,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
