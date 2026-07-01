"""VGGFace2 1.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout::

    <root>/
      <identity_id>/
        <identity_id>_0001.jpg
      loose_bb_train.csv          # optional bounding boxes: NAME_ID,X,Y,W,H
      loose_landmark_train.csv    # optional 5-point landmarks

TODO (Phase 3): parse the bounding-box/landmark CSVs and cross-check them
against the identity subfolders of images.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class VggFace2Adapter(SkeletonFormatAdapter):
    format_name = "vggface2"
    aliases = ["vgg-face2", "vggface2-1.0"]
    version = "1.0"
    supported_tasks = ["classification", "detection"]
    supported_shapes = ["bbox", "points"]
    supports_attributes = True
    supports_tracks = False
    display_name = "VGGFace2 1.0"

    required_any = ["*/*.jpg", "*/*.png", "loose_bb_*.csv", "loose_landmark_*.csv"]
    structure_hint = "identity subfolders of face images, optional loose_bb_*.csv / loose_landmark_*.csv"
