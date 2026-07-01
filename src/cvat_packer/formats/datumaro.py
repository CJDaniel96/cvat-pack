"""Datumaro 1.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout::

    <root>/
      annotations/
        default.json        # Datumaro project-format JSON (items, categories, ...)
      images/
        default/*.jpg

TODO (Phase 3): parse the Datumaro JSON schema (categories/items/annotations
per item), validate item <-> image correspondence, and support Datumaro's
richer annotation types (cuboid_3d, hash_key, etc.).
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class DatumaroAdapter(SkeletonFormatAdapter):
    format_name = "datumaro"
    aliases = ["datumaro-1.0"]
    version = "1.0"
    supported_tasks = ["detection", "segmentation", "keypoints", "classification"]
    supported_shapes = ["bbox", "polygon", "points", "mask"]
    supports_attributes = True
    supports_tracks = True
    display_name = "Datumaro 1.0"

    required_any = ["annotations/*.json", "*/annotations/*.json"]
    structure_hint = "annotations/<subset>.json plus images/<subset>/ folders (Datumaro project layout)"
