"""MOTS PNG 1.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout::

    <root>/
      img1/000001.jpg
      instances/000001.png     # per-pixel instance-id-encoded PNG masks
                               # (pixel value = class_id * 1000 + instance_id)
      seqinfo.ini

TODO (Phase 3): decode the instance-id-encoded PNGs and cross-check
instance/track ids across frames, and verify mask/image dimensions match.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class MotsAdapter(SkeletonFormatAdapter):
    format_name = "mots"
    aliases = ["mots-png", "mots-1.0"]
    version = "1.0"
    supported_tasks = ["tracking", "segmentation"]
    supported_shapes = ["mask"]
    supports_attributes = False
    supports_tracks = True
    display_name = "MOTS PNG 1.0"

    required_any = ["instances/*.png", "instances_txt/*.txt"]
    structure_hint = "sequence folder with img1/ frames and instances/ (PNG masks) or instances_txt/ (RLE)"
