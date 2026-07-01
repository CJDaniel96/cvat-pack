"""COCO 1.0 format adapter.

Expected CVAT upload layout::

    <root>/
      images/
        img1.jpg
        img2.jpg
      annotations/
        instances_default.json   # {"images": [...], "annotations": [...], "categories": [...]}

`--dataset` should point at <root>. Alternatively pass `--annotations` at the
JSON file (or a folder containing it) and `--images` at the images folder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cvat_packer.core.archive import build_zip
from cvat_packer.core.config import PackConfig
from cvat_packer.core.errors import ValidationFailedError
from cvat_packer.core.filesystem import ensure_dir, make_staging_dir, safe_copy
from cvat_packer.core.models import PackageResult, Status, ValidationReport
from cvat_packer.core.registry import register
from cvat_packer.formats.base import FormatAdapter
from cvat_packer.validators.common import find_images


def find_coco_json(config: PackConfig) -> Path | None:
    candidates: list[Path] = []
    if config.annotations:
        if config.annotations.is_file() and config.annotations.suffix == ".json":
            candidates.append(config.annotations)
        elif config.annotations.is_dir():
            candidates.extend(sorted(config.annotations.glob("*.json")))
    if config.dataset:
        candidates.extend(sorted((config.dataset / "annotations").glob("*.json")))
        candidates.extend(sorted(config.dataset.glob("*.json")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _images_root(config: PackConfig) -> Path | None:
    if config.images and config.images.exists():
        return config.images
    if config.dataset and (config.dataset / "images").exists():
        return config.dataset / "images"
    if config.dataset:
        return config.dataset
    return None


@register
class CocoAdapter(FormatAdapter):
    format_name = "coco"
    aliases = ["coco-1.0", "coco1.0"]
    version = "1.0"
    supported_tasks = ["detection", "instance_segmentation"]
    supported_shapes = ["bbox", "polygon", "rle"]
    supports_attributes = True
    supports_tracks = False
    display_name = "COCO 1.0"

    def detect(self, input_path: Path) -> bool:
        if input_path is None or not input_path.exists():
            return False
        if input_path.is_file() and input_path.suffix == ".json":
            return self._looks_like_coco(input_path)
        for json_file in input_path.rglob("*.json"):
            if self._looks_like_coco(json_file):
                return True
        return False

    @staticmethod
    def _looks_like_coco(json_file: Path) -> bool:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return False
        return isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys()

    def validate(self, config: PackConfig) -> ValidationReport:
        report = ValidationReport(format=self.format_name)
        json_path = find_coco_json(config)
        if json_path is None:
            report.add_error(
                "[coco] Could not find a COCO annotation JSON file. Pass --annotations "
                "pointing at the JSON file, or --dataset with an 'annotations/*.json' "
                "file inside (e.g. annotations/instances_default.json)."
            )
            return report

        try:
            data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.add_error(f"[coco] Failed to parse {json_path}: {exc}")
            return report

        for key in ("images", "annotations", "categories"):
            if key not in data:
                report.add_error(f"[coco] Missing required top-level key '{key}' in {json_path.name}")
        if report.status == Status.FAILED:
            return report

        images = data["images"]
        annotations = data["annotations"]
        categories = data["categories"]

        report.images_count = len(images)
        report.annotations_count = len(annotations)

        image_ids: dict[Any, dict[str, Any]] = {}
        for img in images:
            if "id" not in img or "file_name" not in img:
                report.add_error(f"[coco] Image entry missing 'id' or 'file_name': {img}")
                continue
            image_ids[img["id"]] = img

        category_ids = set()
        for cat in categories:
            if "id" not in cat or "name" not in cat:
                report.add_error(f"[coco] Category entry missing 'id' or 'name': {cat}")
                continue
            category_ids.add(cat["id"])

        images_root = _images_root(config)
        available_images = find_images(images_root)

        for img in images:
            file_name = img.get("file_name")
            if not file_name:
                continue
            base_name = Path(file_name).name
            if images_root is not None and base_name not in available_images:
                report.missing_images.append(file_name)
                report.add_warning(f"[coco] Referenced image not found on disk: {file_name}")

        for ann in annotations:
            ann_id = ann.get("id")
            image_id = ann.get("image_id")
            category_id = ann.get("category_id")

            if image_id not in image_ids:
                report.orphan_annotations.append(str(ann_id))
                report.add_warning(f"[coco] Annotation {ann_id} references unknown image_id {image_id}")
            if category_id not in category_ids:
                report.add_error(f"[coco] Annotation {ann_id} references unknown category_id {category_id}")

            bbox = ann.get("bbox")
            if bbox is not None:
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    report.add_error(f"[coco] Annotation {ann_id} has invalid bbox (expected [x,y,w,h]): {bbox}")
                elif bbox[2] <= 0 or bbox[3] <= 0:
                    report.add_error(f"[coco] Annotation {ann_id} has non-positive width/height in bbox: {bbox}")

            segmentation = ann.get("segmentation")
            if segmentation is not None:
                if isinstance(segmentation, list):
                    for poly in segmentation:
                        if not isinstance(poly, list) or len(poly) % 2 != 0 or len(poly) < 6:
                            report.add_error(
                                f"[coco] Annotation {ann_id} has a malformed polygon in 'segmentation' "
                                "(expected a flat list of an even number of >=6 coordinates)"
                            )
                            break
                elif isinstance(segmentation, dict):
                    if "counts" not in segmentation or "size" not in segmentation:
                        report.add_error(
                            f"[coco] Annotation {ann_id} has malformed RLE segmentation "
                            "(expected keys 'counts' and 'size')"
                        )
                else:
                    report.add_error(
                        f"[coco] Annotation {ann_id} 'segmentation' must be a polygon list or an RLE dict"
                    )

            keypoints = ann.get("keypoints")
            if keypoints is not None and len(keypoints) % 3 != 0:
                report.add_error(
                    f"[coco] Annotation {ann_id} 'keypoints' length must be a multiple of 3 "
                    f"(x, y, visibility per point), got {len(keypoints)}"
                )

        if not images:
            report.add_warning("[coco] Dataset contains zero images")
        if not annotations:
            report.add_warning("[coco] Dataset contains zero annotations")

        return report

    def build_package(self, config: PackConfig, report: ValidationReport) -> PackageResult:
        json_path = find_coco_json(config)
        if json_path is None:
            raise ValidationFailedError("[coco] Cannot build package: annotation JSON not found")

        staging = make_staging_dir()
        ann_dir = ensure_dir(staging / "annotations")
        target_name = json_path.name if json_path.name.startswith("instances") else "instances_default.json"
        safe_copy(json_path, ann_dir / target_name)

        if config.copy_images:
            images_root = _images_root(config)
            available_images = find_images(images_root)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            images_dir = ensure_dir(staging / "images")
            for img in data.get("images", []):
                file_name = img.get("file_name")
                if not file_name:
                    continue
                base_name = Path(file_name).name
                src = available_images.get(base_name)
                if src is not None:
                    safe_copy(src, images_dir / base_name)

        output_zip = build_zip(staging, config.output)
        report.output_zip = str(output_zip)
        return PackageResult(success=True, report=report, staging_dir=staging, output_zip=output_zip)
