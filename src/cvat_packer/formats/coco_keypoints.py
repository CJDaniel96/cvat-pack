"""COCO Keypoints 1.0 format adapter.

Same on-disk layout as plain COCO (see cvat_packer.formats.coco), but each
annotation is expected to carry a `keypoints` field (flat x, y, visibility
triples) and each category should declare its own `keypoints` name list (and
optionally a `skeleton` edge list) for CVAT to reconstruct point labels.
"""

from __future__ import annotations

import json

from cvat_packer.core.config import PackConfig
from cvat_packer.core.models import ValidationReport
from cvat_packer.core.registry import register
from cvat_packer.formats.coco import CocoAdapter, find_coco_json


@register
class CocoKeypointsAdapter(CocoAdapter):
    format_name = "coco-keypoints"
    aliases = ["coco_keypoints", "coco-keypoints-1.0"]
    version = "1.0"
    supported_tasks = ["keypoints"]
    supported_shapes = ["keypoints", "skeleton"]
    supports_attributes = True
    supports_tracks = False
    display_name = "COCO Keypoints 1.0"

    def validate(self, config: PackConfig) -> ValidationReport:
        report = super().validate(config)

        json_path = find_coco_json(config)
        if json_path is None:
            return report

        data = json.loads(json_path.read_text(encoding="utf-8"))
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        has_keypoints = any("keypoints" in ann for ann in annotations)
        if annotations and not has_keypoints:
            report.add_error(
                "[coco-keypoints] No annotation contains a 'keypoints' field. This format "
                "requires per-instance keypoints (flat x, y, visibility triples)."
            )

        for cat in categories:
            if "keypoints" not in cat:
                report.add_warning(
                    f"[coco-keypoints] Category '{cat.get('name')}' has no 'keypoints' name "
                    "list (recommended so CVAT can label each skeleton point)"
                )

        return report
