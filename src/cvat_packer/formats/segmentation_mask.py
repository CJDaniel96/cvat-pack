"""Segmentation Mask 1.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout (Pascal-VOC-style pixel masks)::

    <root>/
      SegmentationClass/
        img1.png              # class-id color-mapped PNG mask
      JPEGImages/
        img1.jpg
      labelmap.txt             # "class:R,G,B::" per line

TODO (Phase 3): parse labelmap.txt, verify each mask's color palette against
it, and verify mask dimensions match the corresponding source image (via
cvat_packer.validators.image when Pillow is installed).
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class SegmentationMaskAdapter(SkeletonFormatAdapter):
    format_name = "segmentation-mask"
    aliases = ["segmentation_mask", "segmentation-mask-1.0"]
    version = "1.0"
    supported_tasks = ["segmentation"]
    supported_shapes = ["mask"]
    supports_attributes = False
    supports_tracks = False
    display_name = "Segmentation Mask 1.0"

    required_any = ["SegmentationClass/*", "labelmap.txt"]
    structure_hint = "SegmentationClass/*.png masks + labelmap.txt class-color map (+ optional JPEGImages/)"
