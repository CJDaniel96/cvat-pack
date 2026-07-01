"""Cityscapes 1.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout::

    <root>/
      leftImg8bit/<split>/<city>/<city>_..._leftImg8bit.png
      gtFine/<split>/<city>/<city>_..._gtFine_labelIds.png
      gtFine/<split>/<city>/<city>_..._gtFine_polygons.json

TODO (Phase 3): validate leftImg8bit <-> gtFine filename correspondence
(same <city>_<seq>_<frame> prefix) and verify labelIds mask dimensions match
the source image.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class CityscapesAdapter(SkeletonFormatAdapter):
    format_name = "cityscapes"
    aliases = ["cityscapes-1.0"]
    version = "1.0"
    supported_tasks = ["segmentation"]
    supported_shapes = ["mask", "polygon"]
    supports_attributes = False
    supports_tracks = False
    display_name = "Cityscapes 1.0"

    required_any = ["gtFine/**/*_labelIds.png", "leftImg8bit/**/*.png"]
    structure_hint = "leftImg8bit/<split>/<city>/*.png images + gtFine/<split>/<city>/*_labelIds.png masks"
