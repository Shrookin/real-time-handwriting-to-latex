"""Generate targeted online-ink augmentation for the v3 expression model.

The generator deliberately combines two safe forms of augmentation:

* controlled symbol templates for weakly represented short sequences such as
  ``sin(x)`` and ``0,1,2,3,4``;
* real MathWriting expressions translated onto a common baseline and joined by
  a real handwritten operator glyph.

The source labels and strokes remain tied together. This is different from
concatenating arbitrary rendered formulas without moving their geometry.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .generate_piecewise_dataset import Atom, Glyph, _bounds, _render_atoms, _write_inkml, load_glyph_library
from .mathwriting import InkPoint, InkStroke, MathWritingSample, iter_inkml, read_inkml, tokenize_expression


@dataclass(frozen=True)
class Template:
    name: str
    label: str
    atoms: tuple[str, ...]


_TEMPLATES = (
    Template("sin", "sin(x)", tuple("sin(x)")),
    Template("cos", "cos(x)", tuple("cos(x)")),
    Template("log", "log(x)", tuple("log(x)")),
    Template("digit-list", "0,1,2,3,4", tuple("0,1,2,3,4")),
    Template("digit-list-long", "0,1,2,3,4,5,6", tuple("0,1,2,3,4,5,6")),
    Template("short-addition", "a+b=c", tuple("a+b=c")),
    Template("short-subtraction", "a-b=c", tuple("a-b=c")),
    Template("short-product", "x+y=z", tuple("x+y=z")),
)


def _as_strokes(raw_strokes: Iterable[dict]) -> tuple[InkStroke, ...]:
    return tuple(
        InkStroke(tuple(InkPoint(float(point["x"]), float(point["y"]), point.get("t")) for point in stroke["points"]))
        for stroke in raw_strokes
        if stroke.get("points")
    )


def _transform_group(strokes: tuple[InkStroke, ...], left: float, baseline: float, height: float, jitter: random.Random) -> tuple[InkStroke, ...]:
    min_x, min_y, max_x, max_y = _bounds(strokes)
    scale = height / max(1.0, max_y - min_y)
    x_jitter = jitter.uniform(-2.0, 2.0)
    y_jitter = jitter.uniform(-2.0, 2.0)
    top = baseline - height + y_jitter
    transformed: list[InkStroke] = []
    for stroke in strokes:
        points = tuple(
            InkPoint(
                left + x_jitter + (point.x - min_x) * scale,
                top + (point.y - min_y) * scale,
                point.t,
            )
            for point in stroke.points
        )
        if points:
            transformed.append(InkStroke(points))
    return tuple(transformed)


def _render_template(template: Template, library: dict[str, tuple[Glyph, ...]], rng: random.Random, sample_id: str) -> MathWritingSample:
    raw, _cursor = _render_atoms(tuple(Atom(atom) for atom in template.atoms), library, 32.0, 72.0, rng)
    strokes = _as_strokes(raw)
    return MathWritingSample(
        sample_id,
        strokes,
        {"label": template.label, "normalizedLabel": template.label, "dataset": "targeted-v3", "augmentation": template.name},
    )


def _eligible_source(sample: MathWritingSample) -> bool:
    if not sample.label or len(sample.strokes) < 1:
        return False
    if any(fragment in sample.label for fragment in (r"\begin{matrix}", r"\begin{cases}", r"\begin{aligned}")):
        return False
    try:
        tokens = tokenize_expression(sample.label)
    except ValueError:
        return False
    return 3 <= len(tokens) <= 20


def _compose_samples(left: MathWritingSample, right: MathWritingSample, operator: Glyph, rng: random.Random, sample_id: str) -> MathWritingSample:
    baseline = 88.0
    height = rng.uniform(34.0, 52.0)
    left_strokes = _transform_group(left.strokes, 24.0, baseline, height, rng)
    _min_x, _min_y, left_max_x, _left_max_y = _bounds(left_strokes)
    operator_left = left_max_x + rng.uniform(18.0, 34.0)
    operator_strokes = _transform_group(operator.strokes, operator_left, baseline, height * 0.75, rng)
    _op_min_x, _op_min_y, op_max_x, _op_max_y = _bounds(operator_strokes)
    right_strokes = _transform_group(right.strokes, op_max_x + rng.uniform(18.0, 34.0), baseline, height, rng)
    label = f"{left.label}+{right.label}"
    return MathWritingSample(
        sample_id,
        left_strokes + operator_strokes + right_strokes,
        {"label": label, "normalizedLabel": label, "dataset": "targeted-v3", "augmentation": "real-expression-plus"},
    )


def _load_source_pool(root: Path, rng: random.Random, limit: int) -> list[MathWritingSample]:
    paths = sorted((root / "train").glob("*.inkml"))
    if limit and len(paths) > limit:
        paths = rng.sample(paths, limit)
    pool: list[MathWritingSample] = []
    for path in paths:
        try:
            sample = read_inkml(path)
        except (OSError, ValueError):
            continue
        if _eligible_source(sample):
            pool.append(sample)
    return pool


def generate_dataset(
    output_root: Path,
    human_root: Path,
    symbols_root: Path,
    train_count: int,
    valid_count: int,
    source_pool_limit: int,
    seed: int,
    overwrite: bool = False,
) -> dict:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_root}")
        for child in output_root.iterdir():
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file():
                        nested.unlink()
                child.rmdir()
            else:
                child.unlink()
    wanted = sorted({atom for template in _TEMPLATES for atom in template.atoms} | {"+", "-", "="})
    library = load_glyph_library(symbols_root, wanted)
    rng = random.Random(seed)
    pool = _load_source_pool(human_root, rng, source_pool_limit)
    if len(pool) < 2:
        raise RuntimeError("Not enough eligible MathWriting source expressions for composition")
    report = {
        "dataset": "targeted-v3",
        "seed": seed,
        "train": train_count,
        "valid": valid_count,
        "sourcePool": len(pool),
        "compositionMix": {"templates": 0, "realExpressionPlus": 0},
    }
    for split, count in (("train", train_count), ("valid", valid_count)):
        for index in range(count):
            sample_id = f"targeted-{split}-{index:06d}"
            if rng.random() < 0.5:
                sample = _render_template(rng.choice(_TEMPLATES), library, rng, sample_id)
                report["compositionMix"]["templates"] += 1
            else:
                left, right = rng.sample(pool, 2)
                sample = _compose_samples(left, right, rng.choice(library["+"]), rng, sample_id)
                report["compositionMix"]["realExpressionPlus"] += 1
            _write_inkml(output_root / split / f"{sample_id}.inkml", sample)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate targeted MathWriting augmentation for expression training.")
    parser.add_argument("--human-root", type=Path, required=True)
    parser.add_argument("--symbols-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument("--valid-count", type=int, default=1000)
    parser.add_argument("--source-pool-limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = generate_dataset(args.output_root, args.human_root, args.symbols_root, args.train_count, args.valid_count, args.source_pool_limit, args.seed, args.overwrite)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
