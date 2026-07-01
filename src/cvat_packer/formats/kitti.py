"""KITTI 1.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout (detection variant)::

    <root>/
      image_2/000000.png
      label_2/000000.txt   # "type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry" per line

Segmentation variant uses instance/semantic PNG masks instead of label_2/*.txt.

TODO (Phase 3): parse the 15-field KITTI label line schema, and for the
segmentation variant, verify mask <-> image dimension correspondence.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class KittiAdapter(SkeletonFormatAdapter):
    format_name = "kitti"
    aliases = ["kitti-1.0"]
    version = "1.0"
    supported_tasks = ["detection", "segmentation"]
    supported_shapes = ["bbox", "mask"]
    supports_attributes = True
    supports_tracks = False
    display_name = "KITTI 1.0"

    required_any = ["label_2/*.txt", "*/label_2/*.txt", "instance/*.png", "semantic/*.png"]
    structure_hint = "image_2/*.png + label_2/*.txt (detection), or instance/semantic mask PNGs (segmentation)"
