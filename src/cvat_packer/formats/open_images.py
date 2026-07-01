"""Open Images 1.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout::

    <root>/
      images/<subset>/<ImageID>.jpg
      annotations/
        <subset>-annotations-bbox.csv     # ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,...
        <subset>-annotations-human-imagelabels.csv   # optional, tag-level labels
      metadata/
        class-descriptions-boxable.csv     # LabelName,DisplayName

TODO (Phase 3): parse the bbox CSV schema, validate ImageID <-> image file
correspondence, and validate the LabelName <-> class-descriptions mapping.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class OpenImagesAdapter(SkeletonFormatAdapter):
    format_name = "open-images"
    aliases = ["open_images", "open-images-1.0"]
    version = "1.0"
    supported_tasks = ["detection", "classification"]
    supported_shapes = ["bbox"]
    supports_attributes = True
    supports_tracks = False
    display_name = "Open Images 1.0"

    required_any = ["*-annotations-bbox.csv", "**/*-annotations-bbox.csv", "class-descriptions*.csv"]
    structure_hint = (
        "annotations/<subset>-annotations-bbox.csv + metadata/class-descriptions-boxable.csv "
        "+ images/<subset>/*.jpg"
    )
