"""Generate targeted synthetic replay data for the v4 failure categories.

The corpus is intentionally separate from evaluation data.  It reuses isolated
human MathWriting glyphs, but creates new expression layouts with controlled
variation for scripts, operators, fractions, functions, sequences, roots,
integrals, and long compositions.
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


OPERATOR_LABELS = frozenset({"+", "-", "=", "<", ">", r"\le", r"\ge"})


@dataclass(frozen=True)
class Atom:
    label: str
    script: str | None = None


@dataclass(frozen=True)
class Profile:
    scale: tuple[float, float]
    gap: tuple[float, float]
    baseline_slope: tuple[float, float]
    point_jitter: float
    baseline_jitter: float


@dataclass(frozen=True)
class Rendered:
    strokes: tuple[InkStroke, ...]
    right: float


PROFILES = (
    Profile((0.78, 1.20), (5.0, 13.0), (-0.02, 0.02), 0.6, 1.5),
    Profile((0.82, 1.16), (7.0, 18.0), (-0.06, 0.06), 1.0, 2.5),
    Profile((0.88, 1.10), (3.0, 10.0), (-0.12, 0.12), 1.4, 3.5),
    Profile((0.74, 1.24), (9.0, 22.0), (-0.03, 0.03), 1.8, 4.0),
)


CATEGORY_WEIGHTS = {
    "scripts": 18,
    "operators": 14,
    "fractions": 16,
    "functions": 12,
    "sequences": 10,
    "roots": 10,
    "integrals": 10,
    "long": 10,
}


def _transform(
    glyph: Glyph,
    left: float,
    baseline: float,
    height: float,
    rng: random.Random,
    profile: Profile,
    script: str | None = None,
    vertical_shift: float = 0.0,
) -> tuple[InkStroke, ...]:
    min_x, min_y, max_x, max_y = _bounds(glyph.strokes)
    scale = height / max(1.0, max_y - min_y)
    script_shift = -height * 0.58 if script == "sup" else height * 0.54 if script == "sub" else 0.0
    top = baseline - height + script_shift + rng.uniform(-profile.baseline_jitter, profile.baseline_jitter)
    strokes: list[InkStroke] = []
    for source in glyph.strokes:
        points = tuple(
            InkPoint(
                left + (point.x - min_x) * scale,
                top + (point.y - min_y) * scale + vertical_shift + rng.uniform(-profile.point_jitter, profile.point_jitter),
                point.t,
            )
            for point in source.points
        )
        if points:
            strokes.append(InkStroke(points))
    return tuple(strokes)


def _operator_shift(label: str, rng: random.Random) -> float:
    if label not in OPERATOR_LABELS:
        return 0.0
    # Most operators stay near the baseline, while a meaningful minority are
    # deliberately displaced to teach lenient ordinary-operator handling.
    return rng.choice((0.0, 0.0, 0.0, -4.0, 4.0, -8.0, 8.0))


def _atoms(
    atoms: Iterable[Atom],
    library: dict[str, tuple[Glyph, ...]],
    left: float,
    baseline: float,
    rng: random.Random,
    profile: Profile,
    height: float = 42.0,
) -> Rendered:
    strokes: list[InkStroke] = []
    cursor = left
    for atom in atoms:
        glyph = rng.choice(library[atom.label])
        atom_height = height * (0.58 if atom.script else 1.0)
        strokes.extend(_transform(glyph, cursor, baseline, atom_height, rng, profile, atom.script, _operator_shift(atom.label, rng)))
        cursor += _glyph_width(glyph, atom_height) + (rng.uniform(2.0, 5.0) if atom.script else rng.uniform(*profile.gap))
    return Rendered(tuple(strokes), cursor)


def _fraction(
    numerator: tuple[Atom, ...],
    denominator: tuple[Atom, ...],
    library: dict[str, tuple[Glyph, ...]],
    left: float,
    baseline: float,
    rng: random.Random,
    profile: Profile,
) -> Rendered:
    numerator_render = _atoms(numerator, library, left + 4.0, baseline - 23.0, rng, profile, 34.0)
    denominator_width_start = left + 4.0
    denominator_render = _atoms(denominator, library, denominator_width_start, baseline + 31.0, rng, profile, 34.0)
    right = max(numerator_render.right, denominator_render.right) + 4.0
    bar_y = baseline + 3.0 + rng.uniform(-2.0, 2.0)
    bar = InkStroke((InkPoint(left, bar_y), InkPoint(right, bar_y)))
    return Rendered(numerator_render.strokes + (bar,) + denominator_render.strokes, right)


def _root(
    content: tuple[Atom, ...],
    library: dict[str, tuple[Glyph, ...]],
    left: float,
    baseline: float,
    rng: random.Random,
    profile: Profile,
) -> Rendered:
    radical = rng.choice(library[r"\sqrt"])
    radical_strokes = _transform(radical, left, baseline, 48.0, rng, profile)
    content_render = _atoms(content, library, left + 25.0, baseline, rng, profile, 40.0)
    top_y = baseline - 43.0
    overbar = InkStroke((InkPoint(left + 17.0, top_y), InkPoint(content_render.right + 2.0, top_y)))
    return Rendered(radical_strokes + (overbar,) + content_render.strokes, content_render.right + 4.0)


def _render(category: str, library: dict[str, tuple[Glyph, ...]], rng: random.Random) -> tuple[str, tuple[InkStroke, ...]]:
    profile = rng.choice(PROFILES)
    baseline = 94.0
    left = 24.0

    if category == "scripts":
        choices = (
            (r"x^2+y_1", (Atom("x"), Atom("2", "sup"), Atom("+"), Atom("y"), Atom("1", "sub"))),
            (r"a_{n+1}+b^2", (Atom("a"), Atom("n", "sub"), Atom("+", "sub"), Atom("1", "sub"), Atom("+"), Atom("b"), Atom("2", "sup"))),
            (r"q_i+r^2", (Atom("q"), Atom("i", "sub"), Atom("+"), Atom("r"), Atom("2", "sup"))),
            (r"x_1^2+y_2^3", (Atom("x"), Atom("1", "sub"), Atom("2", "sup"), Atom("+"), Atom("y"), Atom("2", "sub"), Atom("3", "sup"))),
        )
        label, atoms = rng.choice(choices)
        rendered = _atoms(atoms, library, left, baseline, rng, profile)
        return label, rendered.strokes

    if category == "operators":
        choices = (
            ("a+b-c", (Atom("a"), Atom("+"), Atom("b"), Atom("-"), Atom("c"))),
            ("x+y=z", (Atom("x"), Atom("+"), Atom("y"), Atom("="), Atom("z"))),
            ("a-b+c", (Atom("a"), Atom("-"), Atom("b"), Atom("+"), Atom("c"))),
            (r"x\ge0", (Atom("x"), Atom(r"\ge"), Atom("0"))),
            (r"x\le1", (Atom("x"), Atom(r"\le"), Atom("1"))),
            ("x<0", (Atom("x"), Atom("<"), Atom("0"))),
        )
        label, atoms = rng.choice(choices)
        rendered = _atoms(atoms, library, left, baseline, rng, profile)
        return label, rendered.strokes

    if category == "fractions":
        choices = (
            (r"\frac{a}{b}", ("a",), ("b",)),
            (r"\frac{x+1}{y-2}", ("x", "+", "1"), ("y", "-", "2")),
            (r"\frac{1}{x}", ("1",), ("x",)),
            (r"\frac{a+b}{c+d}", ("a", "+", "b"), ("c", "+", "d")),
        )
        label, numerator, denominator = rng.choice(choices)
        rendered = _fraction(tuple(Atom(item) for item in numerator), tuple(Atom(item) for item in denominator), library, left, baseline, rng, profile)
        return label, rendered.strokes

    if category == "functions":
        choices = (
            ("sin(x)", ("s", "i", "n", "(", "x", ")")),
            ("cos(x)", ("c", "o", "s", "(", "x", ")")),
            ("log(x)", ("l", "o", "g", "(", "x", ")")),
            ("sin(x)+1", ("s", "i", "n", "(", "x", ")", "+", "1")),
            ("f(x)=x^2", ("f", "(", "x", ")", "=", "x", "2")),
        )
        label, atom_labels = rng.choice(choices)
        atoms = tuple(Atom(item, "sup") if item == "2" and label.endswith("x^2") else Atom(item) for item in atom_labels)
        rendered = _atoms(atoms, library, left, baseline, rng, profile)
        return label, rendered.strokes

    if category == "sequences":
        choices = (
            ("0,1,2,3,4", ("0", ",", "1", ",", "2", ",", "3", ",", "4")),
            ("1,2,3,4,5", ("1", ",", "2", ",", "3", ",", "4", ",", "5")),
            (r"a_1,a_2,a_3", ("a", "1", ",", "a", "2", ",", "a", "3")),
            (r"x_1+x_2+x_3", ("x", "1", "+", "x", "2", "+", "x", "3")),
            ("0,1,2,3,4,5,6", ("0", ",", "1", ",", "2", ",", "3", ",", "4", ",", "5", ",", "6")),
        )
        label, atom_labels = rng.choice(choices)
        atoms = tuple(Atom(item, "sub") if index % 3 == 1 and item in "123" and label.startswith(("a_", "x_")) else Atom(item) for index, item in enumerate(atom_labels))
        rendered = _atoms(atoms, library, left, baseline, rng, profile)
        return label, rendered.strokes

    if category == "roots":
        choices = (
            (r"\sqrt{x}", (Atom("x"),)),
            (r"\sqrt{x^2+1}", (Atom("x"), Atom("2", "sup"), Atom("+"), Atom("1"))),
            (r"\sqrt{a+b}", (Atom("a"), Atom("+"), Atom("b"))),
        )
        label, atoms = rng.choice(choices)
        rendered = _root(atoms, library, left, baseline, rng, profile)
        return label, rendered.strokes

    if category == "integrals":
        choices = (
            (r"\int_{0}^{1}f(x)dx", (Atom(r"\int"), Atom("0", "sub"), Atom("1", "sup"), Atom("f"), Atom("("), Atom("x"), Atom(")"), Atom("d"), Atom("x"))),
            (r"\int_{a}^{b}xdx", (Atom(r"\int"), Atom("a", "sub"), Atom("b", "sup"), Atom("x"), Atom("d"), Atom("x"))),
            (r"\int_{0}^{1}x^2dx", (Atom(r"\int"), Atom("0", "sub"), Atom("1", "sup"), Atom("x"), Atom("2", "sup"), Atom("d"), Atom("x"))),
        )
        label, atoms = rng.choice(choices)
        rendered = _atoms(atoms, library, left, baseline, rng, profile)
        return label, rendered.strokes

    if category == "long":
        choices = (
            (r"a+b-c+\frac{d}{e}", ("a", "+", "b", "-", "c"), r"\frac{d}{e}"),
            (r"x^2+y_1-\frac{a+b}{c}", ("x", "2", "+", "y", "1", "-"), r"\frac{a+b}{c}"),
            (r"sin(x)+a_1-b^2", ("s", "i", "n", "(", "x", ")", "+", "a", "1", "-", "b", "2"), None),
            (r"\int_{0}^{1}f(x)dx+\frac{a}{b}", (r"\int", "0", "1", "f", "(", "x", ")", "d", "x", "+"), r"\frac{a}{b}"),
        )
        label, prefix, suffix = rng.choice(choices)
        prefix_atoms = []
        for item in prefix:
            script = "sup" if item == "2" and ("x^2" in label or "b^2" in label) else "sub" if item == "1" and "y_1" in label or item == "1" and "a_1" in label else None
            prefix_atoms.append(Atom(item, script))
        first = _atoms(tuple(prefix_atoms), library, left, baseline, rng, profile)
        if suffix is None:
            return label, first.strokes
        if suffix == r"\frac{d}{e}":
            second = _fraction((Atom("d"),), (Atom("e"),), library, first.right + 5.0, baseline, rng, profile)
        elif suffix == r"\frac{a+b}{c}":
            second = _fraction((Atom("a"), Atom("+"), Atom("b")), (Atom("c"),), library, first.right + 5.0, baseline, rng, profile)
        else:
            second = _fraction((Atom("a"),), (Atom("b"),), library, first.right + 5.0, baseline, rng, profile)
        return label, first.strokes + second.strokes

    raise ValueError(f"Unknown weakness category: {category}")


def _category_plan(total: int, rng: random.Random) -> list[str]:
    categories: list[str] = []
    for category, weight in CATEGORY_WEIGHTS.items():
        categories.extend([category] * (total * weight // 100))
    while len(categories) < total:
        categories.append(rng.choice(tuple(CATEGORY_WEIGHTS)))
    rng.shuffle(categories)
    return categories[:total]


def _clear_output(root: Path, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {root}")
        shutil.rmtree(root)


def generate_dataset(output_root: Path, symbols_root: Path, train_count: int, valid_count: int, test_count: int, seed: int, overwrite: bool = False) -> dict:
    _clear_output(output_root, overwrite)
    labels = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "a", "b", "c", "d", "e", "f", "g", "i", "l", "n", "o", "q", "r", "s", "x", "y", "z",
        "+", "-", "=", "<", ">", r"\le", r"\ge", r"\int", r"\sqrt", "(", ")", ",",
    }
    library = load_glyph_library(symbols_root, labels)
    report: dict = {
        "dataset": "targeted-v4-weakness",
        "generatorVersion": 1,
        "seed": seed,
        "source": "MathWriting 2024 isolated human glyphs recombined into controlled expressions",
        "sourceLicense": "CC BY-NC-SA 4.0; verify compatibility before redistribution",
        "evaluationIsolation": "Do not evaluate on this corpus; use MathWriting held-out validation/test and NewNotes fixed cases.",
        "counts": {"train": train_count, "valid": valid_count, "test": test_count},
        "categoryWeights": CATEGORY_WEIGHTS,
        "categoryCounts": {"train": {}, "valid": {}, "test": {}},
        "symbols": {label: len(options) for label, options in library.items()},
    }
    for split, count, offset in (("train", train_count, 0), ("valid", valid_count, 1_000_000), ("test", test_count, 2_000_000)):
        split_rng = random.Random(seed + offset)
        plan = _category_plan(count, split_rng)
        category_counts = Counter(plan)
        report["categoryCounts"][split] = dict(sorted(category_counts.items()))
        for index, category in enumerate(plan):
            sample_id = f"weakness-v4-{split}-{index:07d}"
            label, strokes = _render(category, library, split_rng)
            sample = MathWritingSample(sample_id, strokes, {
                "label": label,
                "normalizedLabel": label,
                "dataset": "targeted-v4-weakness",
                "category": category,
                "generatorVersion": "1",
            })
            _write_inkml(output_root / split / f"{sample_id}.inkml", sample)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate targeted synthetic replay data for v4 expression weaknesses.")
    parser.add_argument("--symbols-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=100000)
    parser.add_argument("--valid-count", type=int, default=10000)
    parser.add_argument("--test-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(generate_dataset(args.output_root, args.symbols_root, args.train_count, args.valid_count, args.test_count, args.seed, args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
