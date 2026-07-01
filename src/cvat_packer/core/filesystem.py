"""Filesystem walking, junk filtering and safe-copy helpers.

Centralizing this here means every format adapter automatically ignores
`.DS_Store`, `Thumbs.db`, `__MACOSX`, etc. without having to remember to do
so itself.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterator

IGNORED_FILENAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
IGNORED_DIR_NAMES = {"__MACOSX"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_junk(path: Path) -> bool:
    if path.name in IGNORED_FILENAMES:
        return True
    return any(part in IGNORED_DIR_NAMES for part in path.parts)


def iter_files(root: Path | None) -> Iterator[Path]:
    """Yield every non-junk file under root, in a stable sorted order."""
    if root is None or not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and not is_junk(path):
            yield path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_staging_dir(prefix: str = "cvat_pack_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def safe_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def safe_copytree(src: Path | None, dst: Path, copy_images: bool = True) -> None:
    """Best-effort recursive copy of `src` into `dst`, skipping junk files.

    Used by skeleton (Phase 3 stub) format adapters that do not yet have a
    format-specific packaging routine.
    """
    if src is None or not src.exists():
        return
    ensure_dir(dst)
    for file in iter_files(src):
        rel = file.relative_to(src)
        if not copy_images and file.suffix.lower() in IMAGE_EXTENSIONS:
            continue
        safe_copy(file, dst / rel)
