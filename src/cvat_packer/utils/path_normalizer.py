"""Cross-platform path normalization helpers.

All paths written into an output zip must be relative, use POSIX ('/')
separators, and never escape the archive root via '..'. This module is the
single place that enforces those rules so every format adapter behaves the
same way on Windows, macOS and Linux.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def to_posix(path: str | Path) -> str:
    """Convert any OS-native path into a POSIX-style ('/') string."""
    return PurePosixPath(Path(str(path)).as_posix()).as_posix()


def is_safe_relative_path(path: str) -> bool:
    """Reject absolute paths, Windows drive letters, '..' traversal, and
    empty/current-dir paths (an empty or '.' arcname is never a valid file
    entry to write into the archive)."""
    posix = to_posix(path)
    if not posix or posix == "." or posix.startswith("/"):
        return False
    parts = posix.split("/")
    first = parts[0]
    if len(first) == 2 and first[1] == ":":
        return False  # e.g. "C:" drive letter
    return ".." not in parts


def normalize_arcname(root: Path, file: Path) -> str:
    """Compute the POSIX-style archive name of `file` relative to `root`."""
    rel = file.relative_to(root)
    return to_posix(rel)


def basename(path_field: str) -> str:
    """Extract the final path component from an annotation-file path string,
    regardless of whether it uses '/' or '\\\\' separators.

    Annotation JSON/XML `file_name`/`name` fields are written by whatever OS
    produced the dataset, so a Windows-authored COCO export may contain
    `images\\img1.jpg`. `Path(...).name` alone only splits on the *host*
    OS's separator, so on macOS/Linux it would treat the whole Windows-style
    string as a single filename. Normalize to POSIX first so basename
    extraction is correct on every platform regardless of the file's origin.
    """
    return PurePosixPath(path_field.replace("\\", "/")).name
