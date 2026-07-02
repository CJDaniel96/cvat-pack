"""Ultralytics YOLO format adapters: Detection, Segmentation, Pose, OBB, Classification.

Detection/Segmentation/Pose/OBB share the same directory layout::

    <root>/
      data.yaml              # {"names": {0: "cat", 1: "dog", ...}, "train": ..., "val": ...}
      images/
        train/*.jpg
        val/*.jpg
        test/*.jpg            # optional
      labels/
        train/*.txt
        val/*.txt
        test/*.txt            # optional

Only the label line format differs per task:
  - Detection:     class x_center y_center width height                (5 fields)
  - Segmentation:  class x1 y1 x2 y2 x3 y3 ...                          (>=7, odd count after class)
  - Pose:          class x_center y_center width height (x y [v])*n_kpt (5 + groups of 2 or 3)
  - OBB:           class x1 y1 x2 y2 x3 y3 x4 y4                        (9 fields, 4 corner points)

Classification uses a different, folder-per-class layout::

    <root>/
      train/<class_name>/*.jpg
      val/<class_name>/*.jpg
      test/<class_name>/*.jpg   # optional

`--dataset` should point at <root> in all cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cvat_packer.core.archive import build_zip
from cvat_packer.core.config import PackConfig
from cvat_packer.core.errors import ValidationFailedError
from cvat_packer.core.filesystem import IMAGE_EXTENSIONS, iter_files, make_staging_dir, safe_copy
from cvat_packer.core.models import PackageResult, ValidationReport
from cvat_packer.core.registry import register
from cvat_packer.formats.base import FormatAdapter
from cvat_packer.validators.common import find_images, unmatched_stems

SPLITS = ("train", "val", "test")


def _dataset_root(config: PackConfig) -> Path | None:
    if config.dataset and config.dataset.exists():
        return config.dataset
    return None


def _load_data_yaml(root: Path) -> dict[str, Any] | None:
    candidate = root / "data.yaml"
    if not candidate.exists():
        return None
    try:
        return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None


def _class_names(data: dict[str, Any] | None, config: PackConfig) -> list[str] | None:
    if config.labels and config.labels.exists():
        lines = [l.strip() for l in config.labels.read_text(encoding="utf-8").splitlines() if l.strip()]
        return lines or None
    if data is None:
        return None
    names = data.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return names
    return None


class UltralyticsYoloBoxBase(FormatAdapter):
    """Shared logic for Ultralytics YOLO detection/segmentation/pose/OBB."""

    #: exact number of fields expected per label line, or None for variable-length
    fields_per_object: int | None = 5
    min_fields: int = 5

    def detect(self, input_path: Path) -> bool:
        if input_path is None or not input_path.exists():
            return False
        return (input_path / "data.yaml").exists() and (input_path / "images").exists()

    def validate(self, config: PackConfig) -> ValidationReport:
        report = ValidationReport(format=self.format_name)
        root = _dataset_root(config)
        if root is None:
            report.add_error(
                f"[{self.format_name}] Pass --dataset pointing at a folder containing "
                "data.yaml, images/<split>/ and labels/<split>/."
            )
            return report

        data = _load_data_yaml(root)
        classes = _class_names(data, config)
        if classes is None:
            report.add_error(
                f"[{self.format_name}] Could not determine class names. Add a 'names' "
                "mapping/list to data.yaml, or pass --labels with a classes.txt file."
            )
            return report

        if not self._validate_root(data, report):
            return report

        images_dir = root / "images"
        labels_dir = root / "labels"
        if not images_dir.exists():
            report.add_error(f"[{self.format_name}] Missing 'images/' directory under {root}")
        if not labels_dir.exists():
            report.add_error(f"[{self.format_name}] Missing 'labels/' directory under {root}")
        if not images_dir.exists() or not labels_dir.exists():
            return report

        found_split = False
        for split in SPLITS:
            split_images = images_dir / split
            split_labels = labels_dir / split
            if not split_images.exists():
                continue
            found_split = True

            available_images = find_images(split_images)
            report.images_count += len(available_images)

            label_files = {f.stem: f for f in split_labels.glob("*.txt")} if split_labels.exists() else {}
            images_without_label, labels_without_image = unmatched_stems(available_images, label_files.keys())

            for stem, label_file in label_files.items():
                if stem in labels_without_image:
                    report.orphan_annotations.append(f"{split}/{label_file.name}")
                    report.add_warning(
                        f"[{self.format_name}] Label {split}/{label_file.name} has no matching image"
                    )
                lines = [
                    l.strip() for l in label_file.read_text(encoding="utf-8").splitlines() if l.strip()
                ]
                for line_no, line in enumerate(lines, start=1):
                    parts = line.split()
                    ok, msg = self._validate_line(parts, classes, data)
                    if ok:
                        report.annotations_count += 1
                    else:
                        report.add_error(f"[{self.format_name}] {split}/{label_file.name}:{line_no} {msg}")

            for stem in images_without_label:
                report.add_warning(
                    f"[{self.format_name}] Image {split}/{stem} has no label file (background image)"
                )

        if not found_split:
            report.add_error(
                f"[{self.format_name}] No split folders found under images/ (expected one of {SPLITS})"
            )

        return report

    def _validate_root(self, data: dict[str, Any] | None, report: ValidationReport) -> bool:
        """Hook for subclasses to validate extra data.yaml keys (e.g. Pose's
        `kpt_shape`) before per-line validation starts. Return False to abort."""
        return True

    def _validate_line(
        self, parts: list[str], classes: list[str], data: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        if len(parts) < self.min_fields:
            return False, f"expected at least {self.min_fields} fields, got {len(parts)}"
        if self.fields_per_object is not None and len(parts) != self.fields_per_object:
            return False, f"expected exactly {self.fields_per_object} fields, got {len(parts)}"

        class_id_str = parts[0]
        if not class_id_str.lstrip("-").isdigit():
            return False, f"class id must be an integer, got '{class_id_str}'"
        if int(class_id_str) >= len(classes) or int(class_id_str) < 0:
            return False, f"class id {class_id_str} out of range ({len(classes)} classes defined)"

        try:
            values = [float(v) for v in parts[1:]]
        except ValueError:
            return False, f"coordinates must be numeric: {parts[1:]}"
        if any(v < -1e-6 or v > 1 + 1e-6 for v in values):
            return False, f"coordinates must be normalized in [0,1], got {values}"
        return True, ""

    def build_package(self, config: PackConfig, report: ValidationReport) -> PackageResult:
        root = _dataset_root(config)
        if root is None:
            raise ValidationFailedError(f"[{self.format_name}] Cannot build package: --dataset not found")

        staging = make_staging_dir()
        for split in SPLITS:
            split_images = root / "images" / split
            split_labels = root / "labels" / split
            if split_images.exists() and config.copy_images:
                for img in find_images(split_images).values():
                    safe_copy(img, staging / "images" / split / img.name)
            if split_labels.exists():
                for label in split_labels.glob("*.txt"):
                    safe_copy(label, staging / "labels" / split / label.name)

        data_yaml = root / "data.yaml"
        if data_yaml.exists():
            safe_copy(data_yaml, staging / "data.yaml")

        output_zip = build_zip(staging, config.output)
        report.output_zip = str(output_zip)
        return PackageResult(success=True, report=report, staging_dir=staging, output_zip=output_zip)


@register
class UltralyticsYoloDetectionAdapter(UltralyticsYoloBoxBase):
    format_name = "ultralytics-yolo-detection"
    aliases = ["yolo-detection", "ultralytics-detection"]
    version = "1.0"
    supported_tasks = ["detection"]
    supported_shapes = ["bbox"]
    supports_attributes = False
    supports_tracks = False
    display_name = "Ultralytics YOLO Detection 1.0"
    fields_per_object = 5
    min_fields = 5


@register
class UltralyticsYoloSegmentationAdapter(UltralyticsYoloBoxBase):
    format_name = "ultralytics-yolo-segmentation"
    aliases = ["yolo-segmentation", "ultralytics-segmentation"]
    version = "1.0"
    supported_tasks = ["instance_segmentation"]
    supported_shapes = ["polygon"]
    supports_attributes = False
    supports_tracks = False
    display_name = "Ultralytics YOLO Segmentation 1.0"
    fields_per_object = None  # variable-length polygon
    min_fields = 7  # class + at least 3 (x,y) points

    def _validate_line(
        self, parts: list[str], classes: list[str], data: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        ok, msg = super()._validate_line(parts, classes, data)
        if not ok:
            return ok, msg
        if (len(parts) - 1) % 2 != 0:
            return False, f"polygon coordinates must come in (x,y) pairs, got {len(parts) - 1} values"
        return True, ""


@register
class UltralyticsYoloPoseAdapter(UltralyticsYoloBoxBase):
    format_name = "ultralytics-yolo-pose"
    aliases = ["yolo-pose", "ultralytics-pose"]
    version = "1.0"
    supported_tasks = ["keypoints"]
    supported_shapes = ["skeleton"]
    supports_attributes = False
    supports_tracks = False
    display_name = "Ultralytics YOLO Pose 1.0"
    fields_per_object = None  # 5 (bbox) + n_kpts * dims, per data.yaml `kpt_shape`
    min_fields = 5

    def _validate_root(self, data: dict[str, Any] | None, report: ValidationReport) -> bool:
        kpt_shape = (data or {}).get("kpt_shape")
        if not (isinstance(kpt_shape, list) and len(kpt_shape) == 2):
            report.add_error(
                f"[{self.format_name}] data.yaml must define 'kpt_shape: [n_keypoints, dims]' "
                "(dims is 2 for (x,y) or 3 for (x,y,visibility))"
            )
            return False
        n_kpts, dims = kpt_shape
        if not (isinstance(n_kpts, int) and n_kpts > 0):
            report.add_error(f"[{self.format_name}] kpt_shape[0] (n_keypoints) must be a positive integer, got {n_kpts}")
            return False
        if dims not in (2, 3):
            report.add_error(f"[{self.format_name}] kpt_shape[1] (dims) must be 2 or 3, got {dims}")
            return False
        self._n_kpts = n_kpts
        self._kpt_dims = dims
        return True

    def _validate_line(
        self, parts: list[str], classes: list[str], data: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        # bbox portion (class + 4 coords) uses the normal class-id/normalized-coord checks
        ok, msg = super()._validate_line(parts[:5], classes, data)
        if not ok:
            return ok, msg

        n_kpts, dims = self._n_kpts, self._kpt_dims
        kpt_parts = parts[5:]
        if len(kpt_parts) != n_kpts * dims:
            return False, (
                f"expected {n_kpts * dims} keypoint values ({n_kpts} keypoints x {dims} dims, "
                f"per data.yaml kpt_shape), got {len(kpt_parts)}"
            )

        try:
            kpt_values = [float(v) for v in kpt_parts]
        except ValueError:
            return False, f"keypoint values must be numeric: {kpt_parts}"

        for i in range(n_kpts):
            group = kpt_values[i * dims : (i + 1) * dims]
            x, y = group[0], group[1]
            if x < -1e-6 or x > 1 + 1e-6 or y < -1e-6 or y > 1 + 1e-6:
                return False, f"keypoint {i} (x,y)=({x},{y}) must be normalized in [0,1]"
            if dims == 3 and group[2] not in (0.0, 1.0, 2.0):
                return False, f"keypoint {i} visibility must be 0, 1, or 2, got {group[2]}"

        return True, ""


@register
class UltralyticsYoloOBBAdapter(UltralyticsYoloBoxBase):
    format_name = "ultralytics-yolo-obb"
    aliases = ["yolo-obb", "ultralytics-obb"]
    version = "1.0"
    supported_tasks = ["oriented_detection"]
    supported_shapes = ["rotated_bbox"]
    supports_attributes = False
    supports_tracks = False
    display_name = "Ultralytics YOLO Oriented Bounding Boxes 1.0"
    fields_per_object = 9  # class + 4 (x, y) corner points
    min_fields = 9


@register
class UltralyticsYoloClassificationAdapter(FormatAdapter):
    format_name = "ultralytics-yolo-classification"
    aliases = ["yolo-classification", "ultralytics-classification"]
    version = "1.0"
    supported_tasks = ["classification"]
    supported_shapes: list[str] = []
    supports_attributes = False
    supports_tracks = False
    display_name = "Ultralytics YOLO Classification 1.0"

    def detect(self, input_path: Path) -> bool:
        if input_path is None or not input_path.exists():
            return False
        for split in SPLITS:
            split_dir = input_path / split
            if split_dir.exists() and any(p.is_dir() for p in split_dir.iterdir()):
                return True
        return False

    def validate(self, config: PackConfig) -> ValidationReport:
        report = ValidationReport(format=self.format_name)
        root = _dataset_root(config)
        if root is None:
            report.add_error(
                f"[{self.format_name}] Pass --dataset pointing at a folder with "
                "train/<class>/, val/<class>/ (and optionally test/<class>/) subfolders."
            )
            return report

        found_split = False
        for split in SPLITS:
            split_dir = root / split
            if not split_dir.exists():
                continue
            class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
            if not class_dirs:
                report.add_warning(f"[{self.format_name}] Split '{split}' has no class subfolders")
                continue
            found_split = True
            for class_dir in class_dirs:
                images = find_images(class_dir)
                if not images:
                    report.add_warning(
                        f"[{self.format_name}] Class folder has no images: {split}/{class_dir.name}"
                    )
                report.images_count += len(images)
                report.annotations_count += len(images)

                invalid_ext = [
                    f.name for f in iter_files(class_dir) if f.suffix.lower() not in IMAGE_EXTENSIONS
                ]
                if invalid_ext:
                    report.add_warning(
                        f"[{self.format_name}] Class folder {split}/{class_dir.name} contains "
                        f"file(s) with unrecognized image extensions (ignored): {invalid_ext}"
                    )

        if not found_split:
            report.add_error(
                f"[{self.format_name}] No split folders found under {root} "
                f"(expected one of {SPLITS}, each containing one subfolder per class)"
            )

        return report

    def build_package(self, config: PackConfig, report: ValidationReport) -> PackageResult:
        root = _dataset_root(config)
        if root is None:
            raise ValidationFailedError(f"[{self.format_name}] Cannot build package: --dataset not found")

        staging = make_staging_dir()
        if config.copy_images:
            for split in SPLITS:
                split_dir = root / split
                if not split_dir.exists():
                    continue
                for class_dir in split_dir.iterdir():
                    if not class_dir.is_dir():
                        continue
                    for img in find_images(class_dir).values():
                        safe_copy(img, staging / split / class_dir.name / img.name)

        output_zip = build_zip(staging, config.output)
        report.output_zip = str(output_zip)
        return PackageResult(success=True, report=report, staging_dir=staging, output_zip=output_zip)
