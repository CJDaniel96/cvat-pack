"""Optional image-content validation.

Pillow is an optional dependency (see pyproject.toml `[project.optional-dependencies].images`).
When it is not installed, `get_image_size` simply returns None and callers
should fall back to filename/existence-only checks.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:  # pragma: no cover - exercised only when Pillow is absent
    _HAS_PIL = False


def pil_available() -> bool:
    return _HAS_PIL


def get_image_size(path: Path) -> tuple[int, int] | None:
    """Return (width, height) if Pillow is available and the file is readable, else None."""
    if not _HAS_PIL:
        return None
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None
