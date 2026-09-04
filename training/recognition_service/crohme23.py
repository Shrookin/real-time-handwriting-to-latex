"""Reader for the CROHME 2023 archive.

CROHME23 is distributed as one large archive with several inherited datasets,
new 2023 data, and artificial online-ink data.  The default training split
uses the artificial data and the new CROHME2023 train set.  The inherited
CROHME2019 directory is intentionally excluded because it overlaps the
separately managed CROHME source already used by NewNotes.

The reader accepts either the downloaded ``CROHME23.zip`` or an extracted
``TC11_CROHME23`` directory, so cache generation does not require a second
copy of the 1.7 GB archive.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Iterator
from zipfile import ZipFile

from .crohme import read_crohme_inkml, read_crohme_inkml_bytes
from .mathwriting import MathWritingSample, tokenize_expression


_ZIP_ROOT = "TC11_CROHME23/INKML/"
_PREFIXES = {
    "train": (
        f"{_ZIP_ROOT}train/Artificial_data/",
        f"{_ZIP_ROOT}train/CROHME2023_train/",
    ),
    "valid": (f"{_ZIP_ROOT}val/CROHME2023_val/",),
    "test": (f"{_ZIP_ROOT}test/CROHME2023_test/",),
}


def _valid_label(label: str) -> bool:
    if not label:
        return False
    try:
        tokenize_expression(label)
    except ValueError:
        return False
    return True


def _directory_root(root: Path) -> Path:
    if (root / "INKML").is_dir():
        return root / "INKML"
    if (root / "TC11_CROHME23" / "INKML").is_dir():
        return root / "TC11_CROHME23" / "INKML"
    if root.name == "INKML" and root.is_dir():
        return root
    raise FileNotFoundError(
        f"Could not find CROHME23 INKML directory below {root}; "
        "pass CROHME23.zip or the extracted TC11_CROHME23 directory"
    )


def _directory_paths(root: Path, split: str) -> Iterator[Path]:
    base = _directory_root(root)
    relative_prefixes = tuple(prefix.removeprefix(_ZIP_ROOT) for prefix in _PREFIXES[split])
    for relative_prefix in relative_prefixes:
        yield from sorted((base / relative_prefix).rglob("*.inkml"))


def _archive_members(archive: Path, split: str) -> Iterator[tuple[str, bytes]]:
    prefixes = _PREFIXES[split]
    with ZipFile(archive) as source:
        names = sorted(
            name
            for name in source.namelist()
            if name.lower().endswith(".inkml") and name.startswith(prefixes)
        )
        for name in names:
            yield name, source.read(name)


def iter_crohme23_split(root: str | Path, split: str) -> Iterator[MathWritingSample]:
    """Yield normalized CROHME23 expression samples for a split."""

    if split not in _PREFIXES:
        raise ValueError(f"Unsupported CROHME23 split: {split!r}")
    source = Path(root)
    if source.is_file() and source.suffix.lower() == ".zip":
        for member, content in _archive_members(source, split):
            sample = read_crohme_inkml_bytes(
                content,
                sample_id=f"crohme23:{PurePosixPath(member).with_suffix('').as_posix()}",
            )
            if _valid_label(sample.label):
                yield sample
        return

    for path in _directory_paths(source, split):
        try:
            sample = read_crohme_inkml(path)
        except (OSError, ValueError):
            continue
        if _valid_label(sample.label):
            yield MathWritingSample(
                f"crohme23:{path.relative_to(_directory_root(source)).with_suffix('').as_posix()}",
                sample.strokes,
                sample.annotations,
            )


def iter_crohme23_labels(root: str | Path, split: str) -> Iterator[str]:
    """Yield valid labels without retaining parsed samples."""

    for sample in iter_crohme23_split(root, split):
        yield sample.label
