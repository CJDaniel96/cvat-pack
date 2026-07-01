"""Small validation helpers shared by multiple format adapters."""

from __future__ import annotations

from pathlib import Path

from cvat_packer.core.filesystem import IMAGE_EXTENSIONS, iter_files


def find_images(root: Path | None) -> dict[str, Path]:
    """Map filename (basename, as it would appear in annotations) -> Path,
    for every image file found under root."""
    if root is None:
        return {}
    return {f.name: f for f in iter_files(root) if f.suffix.lower() in IMAGE_EXTENSIONS}


def check_required_paths(root: Path, required: list[str]) -> list[str]:
    """Return the subset of `required` (relative paths) that do not exist under root."""
    return [rel for rel in required if not (root / rel).exists()]
