"""LFW (Labeled Faces in the Wild) 1.0 format adapter (Phase 3: structural
validation only).

Expected CVAT upload layout::

    <root>/
      <person_name>/
        <person_name>_0001.jpg
      pairs.txt              # optional matched/mismatched pair listing

TODO (Phase 3): parse pairs.txt and cross-check referenced identities/images.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class LfwAdapter(SkeletonFormatAdapter):
    format_name = "lfw"
    aliases = ["lfw-1.0"]
    version = "1.0"
    supported_tasks = ["classification", "verification"]
    supported_shapes: list[str] = []
    supports_attributes = False
    supports_tracks = False
    display_name = "LFW 1.0"

    required_any = ["*/*.jpg", "*/*.png", "pairs.txt"]
    structure_hint = "one subfolder per identity containing face images, optional pairs.txt"
