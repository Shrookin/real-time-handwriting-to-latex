"""Generate a narrow replay corpus for integral and symbol-confusion failures.

This corpus is deliberately separate from the fixed regression suite.  It reuses
isolated human MathWriting glyphs but composes them into expressions that target
the current v4 failures: an integral followed by a continuation, integral
bounds and scripts, and visually similar symbols such as ``x/u``, ``a/o``,
``M/m``, ``I/l``, and ``t/f``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil

from .generate_piecewise_dataset import Glyph, _write_inkml, load_glyph_library
from .generate_weakness_dataset import (
    Atom,
    PROFILES,
    Profile,
    Rendered,
    _atoms,
    _fraction,
)
from .mathwriting import InkStroke, MathWritingSample


CATEGORY_WEIGHTS = {"integrals": 70, "symbol-confusion": 30}


def _render_integral(
    library: dict[str, tuple[Glyph, ...]], rng: random.Random, profile: Profile
) -> tuple[str, tuple[InkStroke, ...]]:
    baseline = 96.0
    left = 24.0
    choices = (
        (
            r"\int_{0}^{1}f(x)dx+1",
            (Atom(r"\int"), Atom("0", "sub"), Atom("1", "sup"), Atom("f"), Atom("("), Atom("x"), Atom(")"), Atom("d"), Atom("x"), Atom("+"), Atom("1")),
        ),
        (
            r"\int_{a}^{b}xdx+C",
            (Atom(r"\int"), Atom("a", "sub"), Atom("b", "sup"), Atom("x"), Atom("d"), Atom("x"), Atom("+"), Atom("C")),
        ),
        (
            r"I=\int_{0}^{1}I(x,y)dxdy+C",
            (Atom("I"), Atom("="), Atom(r"\int"), Atom("0", "sub"), Atom("1", "sup"), Atom("I"), Atom("("), Atom("x"), Atom(","), Atom("y"), Atom(")"), Atom("d"), Atom("x"), Atom("d"), Atom("y"), Atom("+"), Atom("C")),
        ),
        (
            r"P=\int_{a}^{b}f(x)dx+1",
            (Atom("P"), Atom("="), Atom(r"\int"), Atom("a", "sub"), Atom("b", "sup"), Atom("f"), Atom("("), Atom("x"), Atom(")"), Atom("d"), Atom("x"), Atom("+"), Atom("1")),
        ),
        (
            r"\int_{0}^{1}x^2dx+a",
            (Atom(r"\int"), Atom("0", "sub"), Atom("1", "sup"), Atom("x"), Atom("2", "sup"), Atom("d"), Atom("x"), Atom("+"), Atom("a")),
        ),
        (
            r"\int f(x)dx+a",
            (Atom(r"\int"), Atom("f"), Atom("("), Atom("x"), Atom(")"), Atom("d"), Atom("x"), Atom("+"), Atom("a")),
        ),
    )
    label, atoms = rng.choice(choices)
    rendered = _atoms(atoms, library, left, baseline, rng, profile, 42.0)
    return label, rendered.strokes


def _render_integral_fraction(
    library: dict[str, tuple[Glyph, ...]], rng: random.Random, profile: Profile
) -> tuple[str, tuple[InkStroke, ...]]:
    baseline = 96.0
    left = 24.0
    choices = (
        (
            r"\int f(x)dx+\frac{GM}{r^2}",
            (Atom(r"\int"), Atom("f"), Atom("("), Atom("x"), Atom(")"), Atom("d"), Atom("x"), Atom("+")),
            (Atom("G"), Atom("M")),
            (Atom("r"), Atom("2", "sup")),
        ),
        (
            r"\int_{0}^{1}I(x,y)dxdy+\frac{M}{a}",
            (Atom(r"\int"), Atom("0", "sub"), Atom("1", "sup"), Atom("I"), Atom("("), Atom("x"), Atom(","), Atom("y"), Atom(")"), Atom("d"), Atom("x"), Atom("d"), Atom("y"), Atom("+")),
            (Atom("M"),),
            (Atom("a"),),
        ),
        (
            r"\int_{a}^{b}xdx+\frac{C}{r}",
            (Atom(r"\int"), Atom("a", "sub"), Atom("b", "sup"), Atom("x"), Atom("d"), Atom("x"), Atom("+")),
            (Atom("C"),),
            (Atom("r"),),
        ),
    )
    label, prefix, numerator, denominator = rng.choice(choices)
    first = _atoms(prefix, library, left, baseline, rng, profile, 42.0)
    fraction = _fraction(numerator, denominator, library, first.right + 6.0, baseline, rng, profile)
    return label, first.strokes + fraction.strokes


def _render_symbol_confusion(
    library: dict[str, tuple[Glyph, ...]], rng: random.Random, profile: Profile
) -> tuple[str, tuple[InkStroke, ...]]:
    baseline = 96.0
    left = 24.0
    choices = (
        ("x+u", (Atom("x"), Atom("+"), Atom("u"))),
        ("u+x", (Atom("u"), Atom("+"), Atom("x"))),
        ("a+o", (Atom("a"), Atom("+"), Atom("o"))),
        ("o+a", (Atom("o"), Atom("+"), Atom("a"))),
        ("M+m", (Atom("M"), Atom("+"), Atom("m"))),
        ("m+M", (Atom("m"), Atom("+"), Atom("M"))),
        ("I+l", (Atom("I"), Atom("+"), Atom("l"))),
        ("l+I", (Atom("l"), Atom("+"), Atom("I"))),
        ("t+f", (Atom("t"), Atom("+"), Atom("f"))),
        ("f+t", (Atom("f"), Atom("+"), Atom("t"))),
        (r"x_{u}+u_{x}", (Atom("x"), Atom("u", "sub"), Atom("+"), Atom("u"), Atom("x", "sub"))),
        (r"a_{o}+o_{a}", (Atom("a"), Atom("o", "sub"), Atom("+"), Atom("o"), Atom("a", "sub"))),
        (r"M_{m}+m_{M}", (Atom("M"), Atom("m", "sub"), Atom("+"), Atom("m"), Atom("M", "sub"))),
        (r"I_{l}+l_{I}", (Atom("I"), Atom("l", "sub"), Atom("+"), Atom("l"), Atom("I", "sub"))),
        (r"t_{f}+f_{t}", (Atom("t"), Atom("f", "sub"), Atom("+"), Atom("f"), Atom("t", "sub"))),
    )
    label, atoms = rng.choice(choices)
    rendered = _atoms(atoms, library, left, baseline, rng, profile, 42.0)
    return label, rendered.strokes


def _render(category: str, library: dict[str, tuple[Glyph, ...]], rng: random.Random) -> tuple[str, tuple[InkStroke, ...]]:
    profile = rng.choice(PROFILES)
    if category == "integrals":
        if rng.random() < 0.28:
            return _render_integral_fraction(library, rng, profile)
        return _render_integral(library, rng, profile)
    if category == "symbol-confusion":
        return _render_symbol_confusion(library, rng, profile)
    raise ValueError(f"Unknown specialized category: {category}")


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
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def generate_dataset(
    output_root: Path,
    symbols_root: Path,
    train_count: int,
    valid_count: int,
    test_count: int,
    seed: int,
    overwrite: bool = False,
) -> dict:
    _clear_output(output_root, overwrite)
    labels = {
        "0", "1", "2", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "+", "-", "=", ",", "(", ")", r"\int", r"\frac",
    }
    library = load_glyph_library(symbols_root, labels)
    report = {
        "dataset": "targeted-v4-integral-symbol-confusion",
        "generatorVersion": 1,
        "seed": seed,
        "source": "MathWriting 2024 isolated human glyphs recombined into integral and symbol-confusion expressions",
        "sourceLicense": "CC BY-NC-SA 4.0; verify compatibility before redistribution",
        "evaluationIsolation": "The test split is evaluation-only; compare against MathWriting held-out validation and NewNotes fixed cases.",
        "counts": {"train": train_count, "valid": valid_count, "test": test_count},
        "categoryWeights": CATEGORY_WEIGHTS,
        "categoryCounts": {},
        "symbols": {label: len(options) for label, options in library.items()},
    }
    for split, count, offset in (("train", train_count, 0), ("valid", valid_count, 1_000_000), ("test", test_count, 2_000_000)):
        split_rng = random.Random(seed + offset)
        plan = _category_plan(count, split_rng)
        report["categoryCounts"][split] = dict(sorted(Counter(plan).items()))
        for index, category in enumerate(plan):
            sample_id = f"integral-symbol-{split}-{index:07d}"
            label, strokes = _render(category, library, split_rng)
            sample = MathWritingSample(sample_id, strokes, {
                "label": label,
                "normalizedLabel": label,
                "dataset": "targeted-v4-integral-symbol-confusion",
                "category": category,
                "generatorVersion": "1",
            })
            _write_inkml(output_root / split / f"{sample_id}.inkml", sample)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate integral and symbol-confusion replay data.")
    parser.add_argument("--symbols-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=50000)
    parser.add_argument("--valid-count", type=int, default=5000)
    parser.add_argument("--test-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = generate_dataset(args.output_root, args.symbols_root, args.train_count, args.valid_count, args.test_count, args.seed, args.overwrite)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
