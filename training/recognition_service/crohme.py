"""Reader for CROHME ground-truth InkML files.

CROHME releases use the standard InkML trace format but vary slightly in
directory layout and metadata. This adapter deliberately returns the same
MathWritingSample shape used by the existing feature pipeline, without
copying CROHME into the training dataset or modifying any checkpoint.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

from .mathwriting import InkPoint, InkStroke, MathWritingSample, tokenize_expression


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_TRUTH_RE = re.compile(
    rb'<annotation[^>]*type=["\']truth["\'][^>]*>(.*?)</annotation>',
    re.DOTALL | re.IGNORECASE,
)
_NORMALIZED_LABEL_RE = re.compile(
    rb'<annotation[^>]*type=["\']normalizedLabel["\'][^>]*>(.*?)</annotation>',
    re.DOTALL | re.IGNORECASE,
)
_TRACE_RE = re.compile(rb"<trace\b[^>]*>(.*?)</trace>", re.DOTALL | re.IGNORECASE)


def _valid_label(label: str) -> bool:
    if not label:
        return False
    try:
        tokenize_expression(label)
    except ValueError:
        return False
    return True


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _clean_truth(value: str) -> str:
    truth = "".join(value.strip().split())
    if truth.startswith("$") and truth.endswith("$"):
        truth = truth[1:-1].strip()
    if truth.startswith(r"\[") and truth.endswith(r"\]"):
        truth = truth[2:-2].strip()
    if truth.startswith(r"\(") and truth.endswith(r"\)"):
        truth = truth[2:-2].strip()
    return truth


def _parse_trace(text: str | None) -> InkStroke:
    points: list[InkPoint] = []
    for raw_point in (text or "").split(","):
        # CROHME's standard trace format is comma-separated whitespace
        # triples: x y [timestamp].  Avoiding a regex for every point makes
        # cache generation substantially faster on the full release.
        values = raw_point.strip().split()
        if len(values) < 2:
            continue
        try:
            points.append(InkPoint(float(values[0]), float(values[1]), float(values[2]) if len(values) >= 3 else None))
        except ValueError:
            fallback = _NUMBER_RE.findall(raw_point)
            if len(fallback) >= 2:
                points.append(InkPoint(float(fallback[0]), float(fallback[1]), float(fallback[2]) if len(fallback) >= 3 else None))
    return InkStroke(tuple(points))


def read_crohme_inkml(path: str | Path) -> MathWritingSample:
    """Read one CROHME InkML file with a LaTeX ``truth`` annotation.

    CROHME files contain large embedded MathML trees.  The online recognizer
    only needs the truth annotation and trace coordinates, so a targeted byte
    scan is both faster and more tolerant of historical XML quirks than
    parsing the complete document tree.
    """

    source = Path(path)
    return read_crohme_inkml_bytes(content=source.read_bytes(), sample_id=source.stem)


def read_crohme_inkml_bytes(content: bytes, *, sample_id: str) -> MathWritingSample:
    """Read one CROHME InkML document already loaded into memory."""

    annotations: dict[str, str] = {}
    traces: list[InkStroke] = []

    truth_match = _TRUTH_RE.search(content) or _NORMALIZED_LABEL_RE.search(content)
    if truth_match:
        annotations["truth"] = _clean_truth(truth_match.group(1).decode("utf-8", errors="replace"))
    for match in _TRACE_RE.finditer(content):
        stroke = _parse_trace(match.group(1).decode("utf-8", errors="replace"))
        if stroke.points:
            traces.append(stroke)

    # Older releases use `truth`; a few converted copies use `normalizedLabel`.
    label = annotations.get("truth") or annotations.get("normalizedLabel") or annotations.get("label") or ""
    annotations["truth"] = _clean_truth(label)
    annotations["normalizedLabel"] = annotations["truth"]
    return MathWritingSample(sample_id, tuple(traces), annotations)


def iter_crohme_inkml(root: str | Path) -> Iterator[MathWritingSample]:
    """Yield all truth-bearing InkML files below a CROHME release directory."""

    for path in sorted(Path(root).rglob("*.inkml")):
        try:
            sample = read_crohme_inkml(path)
        except (ElementTree.ParseError, OSError):
            # Some historical CROHME mirrors contain malformed XML files.
            # They are excluded from iteration rather than aborting a whole
            # training/evaluation run; the raw archive remains untouched.
            continue
        if sample.label:
            yield sample


_SPLIT_DIRECTORIES = {
    "train": (
        "CROHME2011_data/CROHME_training",
        "CROHME2012_data/trainData",
        "CROHME2013_data/TrainINKML",
    ),
    "valid": ("CROHME2014_data/TestEM2014GT",),
}


def iter_crohme_split(root: str | Path, split: str) -> Iterator[MathWritingSample]:
    """Yield only the explicitly assigned CROHME expression split."""

    try:
        directories = _SPLIT_DIRECTORIES[split]
    except KeyError as error:
        raise ValueError(f"Unsupported CROHME split: {split!r}") from error

    base = Path(root)
    for relative_directory in directories:
        for path in sorted((base / relative_directory).rglob("*.inkml")):
            try:
                sample = read_crohme_inkml(path)
            except (ElementTree.ParseError, OSError):
                continue
            if _valid_label(sample.label):
                yield sample


def iter_crohme_labels(root: str | Path, split: str) -> Iterator[str]:
    """Yield labels without parsing traces; used for fast vocabulary scans."""

    try:
        directories = _SPLIT_DIRECTORIES[split]
    except KeyError as error:
        raise ValueError(f"Unsupported CROHME split: {split!r}") from error

    base = Path(root)
    for relative_directory in directories:
        for path in sorted((base / relative_directory).rglob("*.inkml")):
            try:
                content = path.read_bytes()
            except OSError:
                continue
            match = _TRUTH_RE.search(content) or _NORMALIZED_LABEL_RE.search(content)
            if match:
                label = _clean_truth(match.group(1).decode("utf-8", errors="replace"))
                if _valid_label(label):
                    yield label
