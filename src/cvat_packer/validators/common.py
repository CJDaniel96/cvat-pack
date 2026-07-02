"""Small validation helpers shared by multiple format adapters."""

from __future__ import annotations

from collections.abc import Iterable
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


def unmatched_stems(image_names: Iterable[str], annotation_stems: Iterable[str]) -> tuple[set[str], set[str]]:
    """Compare image basenames against annotation-file stems (both matched by
    filename-without-extension, CVAT's convention for pairing images with
    per-image label/annotation files).

    Returns `(images_without_annotation, annotations_without_image)`, each a
    set of stems. Used by every per-file-pair format (YOLO, Ultralytics YOLO,
    Pascal VOC, ...) so the "orphan annotation" / "background image" warning
    logic isn't reimplemented per adapter.
    """
    image_stems = {Path(name).stem for name in image_names}
    annotation_stem_set = set(annotation_stems)
    return image_stems - annotation_stem_set, annotation_stem_set - image_stems
