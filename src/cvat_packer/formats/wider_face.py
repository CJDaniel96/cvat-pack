"""WIDER Face 1.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout::

    <root>/
      wider_face_train_bbx_gt.txt   # "<file>\\n<num_boxes>\\n<x y w h ...> per line" blocks
      images/
        0--Parade/0_Parade_..._0001.jpg

TODO (Phase 3): parse the WIDER Face block format (filename, box count, then
that many "x y w h blur expression illumination invalid occlusion pose"
lines) and cross-check every referenced image exists.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class WiderFaceAdapter(SkeletonFormatAdapter):
    format_name = "wider-face"
    aliases = ["wider_face", "wider-face-1.0"]
    version = "1.0"
    supported_tasks = ["detection"]
    supported_shapes = ["bbox"]
    supports_attributes = True
    supports_tracks = False
    display_name = "WIDER Face 1.0"

    required_any = ["*_bbx_gt.txt", "**/*_bbx_gt.txt"]
    structure_hint = (
        "wider_face_train_bbx_gt.txt / wider_face_val_bbx_gt.txt annotation file(s) plus images/ folder"
    )
