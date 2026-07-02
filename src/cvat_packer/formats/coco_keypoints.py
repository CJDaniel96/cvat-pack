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

        category_num_keypoints: dict[object, int] = {}
        for cat in categories:
            if "keypoints" not in cat:
                report.add_warning(
                    f"[coco-keypoints] Category '{cat.get('name')}' has no 'keypoints' name "
                    "list (recommended so CVAT can label each skeleton point)"
                )
            else:
                category_num_keypoints[cat.get("id")] = len(cat["keypoints"])

        for ann in annotations:
            if "keypoints" not in ann:
                continue
            ann_id = ann.get("id")
            keypoints = ann["keypoints"]

            if ann.get("bbox") is None:
                report.add_error(f"[coco-keypoints] Annotation {ann_id} is missing required 'bbox'")

            num_keypoints = ann.get("num_keypoints")
            if num_keypoints is None:
                report.add_error(f"[coco-keypoints] Annotation {ann_id} is missing required 'num_keypoints'")
            elif len(keypoints) != num_keypoints * 3:
                report.add_error(
                    f"[coco-keypoints] Annotation {ann_id} 'keypoints' length "
                    f"({len(keypoints)}) must equal num_keypoints * 3 ({num_keypoints} * 3 = "
                    f"{num_keypoints * 3})"
                )

            category_kpt_count = category_num_keypoints.get(ann.get("category_id"))
            if category_kpt_count is not None and len(keypoints) != category_kpt_count * 3:
                report.add_error(
                    f"[coco-keypoints] Annotation {ann_id} 'keypoints' length ({len(keypoints)}) "
                    f"does not match its category's keypoint name list length "
                    f"({category_kpt_count} points = {category_kpt_count * 3} values)"
                )

            if len(keypoints) % 3 == 0:
                visibilities = keypoints[2::3]
                bad = [v for v in visibilities if v not in (0, 1, 2)]
                if bad:
                    report.add_error(
                        f"[coco-keypoints] Annotation {ann_id} has invalid visibility value(s) "
                        f"{bad} (each keypoint's visibility must be 0, 1, or 2)"
                    )

        return report
