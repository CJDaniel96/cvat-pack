"""Generic annotation shape helpers (bbox / normalized-coordinate checks)."""

from __future__ import annotations

from collections.abc import Sequence


def is_valid_bbox_xywh(bbox: Sequence[float]) -> bool:
    """COCO-style [x, y, width, height] bbox: must have positive width/height."""
    if len(bbox) != 4:
        return False
    _, _, w, h = bbox
    return w > 0 and h > 0


def is_normalized(value: float, eps: float = 1e-6) -> bool:
    """Whether value lies in [0, 1] (with a small floating-point tolerance)."""
    return -eps <= value <= 1 + eps


def all_normalized(values: Sequence[float], eps: float = 1e-6) -> bool:
    return all(is_normalized(v, eps) for v in values)
