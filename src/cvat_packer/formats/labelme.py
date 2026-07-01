"""LabelMe 3.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout::

    <root>/
      img1.jpg
      img1.xml            # LabelMe XML: <imagePath>, <object><polygon>/<box>...
      (or img1.json for the newer JSON-based LabelMe format)

TODO (Phase 3): parse the LabelMe XML/JSON schema, validate <imagePath>
correspondence, and validate polygon/box point structure.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class LabelMeAdapter(SkeletonFormatAdapter):
    format_name = "labelme"
    aliases = ["labelme-3.0"]
    version = "3.0"
    supported_tasks = ["detection", "segmentation"]
    supported_shapes = ["bbox", "polygon"]
    supports_attributes = True
    supports_tracks = False
    display_name = "LabelMe 3.0"

    required_any = ["*.xml", "*.json"]
    structure_hint = "one LabelMe .xml (or .json) file per image, each referencing imagePath"
