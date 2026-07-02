"""YOLO 1.0 format adapter (CVAT's "YOLO 1.0", Darknet-style export).

Expected CVAT upload layout::

    <root>/
      obj.names            # one class name per line, index == class id
      obj.data             # classes/train/names/backup pointers
      train.txt            # list of "obj_train_data/<image>" relative paths
      obj_train_data/
        img1.jpg
        img1.txt           # "<class_id> <x_center> <y_center> <width> <height>" per line, normalized [0,1]
        img2.jpg
        img2.txt

`--dataset` should point at <root>. Alternatively pass `--images`/`--annotations`
separately and `--labels` at obj.names/classes.txt.
"""

from __future__ import annotations

from pathlib import Path

from cvat_packer.core.archive import build_zip
from cvat_packer.core.config import PackConfig
from cvat_packer.core.errors import ValidationFailedError
from cvat_packer.core.filesystem import ensure_dir, make_staging_dir, safe_copy
from cvat_packer.core.models import PackageResult, ValidationReport
from cvat_packer.core.registry import register
from cvat_packer.formats.base import FormatAdapter
from cvat_packer.validators.common import find_images, unmatched_stems

NON_LABEL_FILES = {"train.txt", "val.txt", "test.txt"}


def _load_classes(config: PackConfig) -> list[str] | None:
    candidates: list[Path] = []
    if config.labels:
        candidates.append(config.labels)
    if config.dataset:
        candidates += [config.dataset / "obj.names", config.dataset / "classes.txt"]
    if config.annotations and config.annotations.is_dir():
        candidates += [config.annotations / "obj.names", config.annotations / "classes.txt"]
    for candidate in candidates:
        if candidate and candidate.exists():
            lines = [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines()]
            return [line for line in lines if line]
    return None


def _labels_root(config: PackConfig) -> Path | None:
    if config.annotations and config.annotations.exists():
        return config.annotations
    if config.dataset:
        for sub in ("obj_train_data", "labels"):
            candidate = config.dataset / sub
            if candidate.exists() and list(candidate.glob("*.txt")):
                return candidate
        if list(config.dataset.glob("*.txt")):
            return config.dataset
    return None


def _images_root(config: PackConfig) -> Path | None:
    if config.images and config.images.exists():
        return config.images
    if config.dataset:
        for sub in ("obj_train_data", "images"):
            candidate = config.dataset / sub
            if candidate.exists():
                return candidate
        return config.dataset
    return None


@register
class YoloAdapter(FormatAdapter):
    format_name = "yolo"
    aliases = ["yolo-1.0", "darknet"]
    version = "1.0"
    supported_tasks = ["detection"]
    supported_shapes = ["bbox"]
    supports_attributes = False
    supports_tracks = False
    display_name = "YOLO 1.0"

    def detect(self, input_path: Path) -> bool:
        if input_path is None or not input_path.exists():
            return False
        names = {p.name for p in input_path.glob("*")}
        if "obj.names" in names or "obj.data" in names:
            return True
        has_txt = bool(list(input_path.rglob("*.txt")))
        has_image = any(f.suffix.lower() in (".jpg", ".jpeg", ".png") for f in input_path.rglob("*"))
        return has_txt and has_image

    def validate(self, config: PackConfig) -> ValidationReport:
        report = ValidationReport(format=self.format_name)

        classes = _load_classes(config)
        if classes is None:
            report.add_error(
                "[yolo] Could not find class names. Pass --labels pointing at a "
                "classes.txt/obj.names file, or include obj.names inside --dataset."
            )
            return report

        labels_root = _labels_root(config)
        images_root = _images_root(config)
        if labels_root is None:
            report.add_error(
                "[yolo] Could not find label .txt files. Pass --annotations pointing at "
                "the folder with per-image .txt files, or include an 'obj_train_data/' "
                "folder inside --dataset."
            )
            return report
        if images_root is None:
            report.add_error(
                "[yolo] Could not find image files. Pass --images, or include images "
                "alongside labels in --dataset."
            )
            return report

        available_images = find_images(images_root)
        report.images_count = len(available_images)

        label_files = [f for f in sorted(labels_root.rglob("*.txt")) if f.name not in NON_LABEL_FILES]
        if not label_files:
            report.add_error(f"[yolo] No label .txt files found under {labels_root}")
            return report

        label_stems = {f.stem for f in label_files}
        images_without_label, labels_without_image = unmatched_stems(available_images, label_stems)

        annotation_count = 0
        for label_file in label_files:
            if label_file.stem in labels_without_image:
                report.orphan_annotations.append(label_file.name)
                report.add_warning(f"[yolo] Label file has no matching image: {label_file.name}")

            lines = [
                line.strip()
                for line in label_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for line_no, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    report.add_error(
                        f"[yolo] {label_file.name}:{line_no} expected 5 fields "
                        f"'class x_center y_center width height', got {len(parts)}"
                    )
                    continue

                class_id_str, *coords = parts
                if not class_id_str.lstrip("-").isdigit():
                    report.add_error(
                        f"[yolo] {label_file.name}:{line_no} class id must be an integer, "
                        f"got '{class_id_str}'"
                    )
                elif int(class_id_str) >= len(classes) or int(class_id_str) < 0:
                    report.add_error(
                        f"[yolo] {label_file.name}:{line_no} class id {class_id_str} out of "
                        f"range (only {len(classes)} classes defined in obj.names)"
                    )

                try:
                    values = [float(v) for v in coords]
                except ValueError:
                    report.add_error(
                        f"[yolo] {label_file.name}:{line_no} coordinates must be numeric: {coords}"
                    )
                    continue

                if any(v < -1e-6 or v > 1 + 1e-6 for v in values):
                    report.add_error(
                        f"[yolo] {label_file.name}:{line_no} coordinates must be normalized "
                        f"in [0,1], got {values}"
                    )
                annotation_count += 1

        report.annotations_count = annotation_count

        for stem in sorted(images_without_label):
            report.add_warning(f"[yolo] Image has no label file (treated as background): {stem}")

        return report

    def build_package(self, config: PackConfig, report: ValidationReport) -> PackageResult:
        classes = _load_classes(config)
        labels_root = _labels_root(config)
        images_root = _images_root(config)
        if classes is None or labels_root is None or images_root is None:
            raise ValidationFailedError("[yolo] Cannot build package: missing classes, labels, or images")

        staging = make_staging_dir()
        data_dir = ensure_dir(staging / "obj_train_data")

        (staging / "obj.names").write_text("\n".join(classes) + "\n", encoding="utf-8")
        (staging / "obj.data").write_text(
            f"classes = {len(classes)}\n"
            "train = train.txt\n"
            "names = obj.names\n"
            "backup = backup/\n",
            encoding="utf-8",
        )

        for label_file in sorted(labels_root.rglob("*.txt")):
            if label_file.name in NON_LABEL_FILES:
                continue
            safe_copy(label_file, data_dir / label_file.name)

        available_images = find_images(images_root)
        train_lines = []
        for name, src in sorted(available_images.items()):
            if config.copy_images:
                safe_copy(src, data_dir / name)
            train_lines.append(f"obj_train_data/{name}")

        (staging / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

        output_zip = build_zip(staging, config.output)
        report.output_zip = str(output_zip)
        return PackageResult(success=True, report=report, staging_dir=staging, output_zip=output_zip)
