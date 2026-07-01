"""CamVid 1.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout::

    <root>/
      <images>/*.png            # source frames
      <labels>/*.png            # per-pixel class-color-coded masks
      label_colors.txt           # "R G B class_name" per line

TODO (Phase 3): parse label_colors.txt, verify each label mask's palette
against it, and verify mask/image dimensions match.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class CamVidAdapter(SkeletonFormatAdapter):
    format_name = "camvid"
    aliases = ["camvid-1.0"]
    version = "1.0"
    supported_tasks = ["segmentation"]
    supported_shapes = ["mask"]
    supports_attributes = False
    supports_tracks = False
    display_name = "CamVid 1.0"

    required_any = ["label_colors.txt", "*.png", "*/*.png"]
    structure_hint = "image + label PNG pairs plus a label_colors.txt class-color map"
