"""PASCAL VOC 1.0 format adapter.

Expected CVAT upload layout::

    <root>/
      JPEGImages/
        img1.jpg
      Annotations/
        img1.xml            # <annotation><filename>...<object><name>...<bndbox>...
      ImageSets/
        Main/train.txt       # optional
      SegmentationClass/     # optional, pixel masks
      SegmentationObject/    # optional, pixel masks

`--dataset` should point at <root>. Alternatively pass `--images` at
JPEGImages and `--annotations` at Annotations.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from cvat_packer.core.archive import build_zip
from cvat_packer.core.config import PackConfig
from cvat_packer.core.errors import ValidationFailedError
from cvat_packer.core.filesystem import ensure_dir, make_staging_dir, safe_copy
from cvat_packer.core.models import PackageResult, ValidationReport
from cvat_packer.core.registry import register
from cvat_packer.formats.base import FormatAdapter
from cvat_packer.validators.common import find_images, unmatched_stems


def _images_dir(config: PackConfig) -> Path | None:
    if config.images and config.images.exists():
        return config.images
    if config.dataset and (config.dataset / "JPEGImages").exists():
        return config.dataset / "JPEGImages"
    return None


def _annotations_dir(config: PackConfig) -> Path | None:
    if config.annotations and config.annotations.exists():
        return config.annotations
    if config.dataset and (config.dataset / "Annotations").exists():
        return config.dataset / "Annotations"
    return None


@register
class PascalVocAdapter(FormatAdapter):
    format_name = "pascal-voc"
    aliases = ["voc", "pascal_voc", "voc-1.0", "pascal-voc-1.0"]
    version = "1.0"
    supported_tasks = ["detection", "segmentation"]
    supported_shapes = ["bbox", "polygon", "mask"]
    supports_attributes = True
    supports_tracks = False
    display_name = "PASCAL VOC 1.0"

    def detect(self, input_path: Path) -> bool:
        if input_path is None or not input_path.exists():
            return False
        return (input_path / "Annotations").exists() and (input_path / "JPEGImages").exists()

    def validate(self, config: PackConfig) -> ValidationReport:
        report = ValidationReport(format=self.format_name)
        images_dir = _images_dir(config)
        annotations_dir = _annotations_dir(config)

        if annotations_dir is None:
            report.add_error(
                "[pascal-voc] Could not find an 'Annotations/' folder with XML files. "
                "Pass --annotations, or --dataset with an Annotations/ subfolder."
            )
            return report
        if images_dir is None:
            report.add_error(
                "[pascal-voc] Could not find a 'JPEGImages/' folder. Pass --images, "
                "or --dataset with a JPEGImages/ subfolder."
            )
            return report

        available_images = find_images(images_dir)
        report.images_count = len(available_images)

        xml_files = sorted(annotations_dir.glob("*.xml"))
        if not xml_files:
            report.add_error(f"[pascal-voc] No XML annotation files found in {annotations_dir}")
            return report

        object_count = 0
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
            except ET.ParseError as exc:
                report.add_error(f"[pascal-voc] Failed to parse {xml_file.name}: {exc}")
                continue

            root = tree.getroot()
            if root.tag != "annotation":
                report.add_error(f"[pascal-voc] {xml_file.name}: root element must be <annotation>")
                continue

            filename_el = root.find("filename")
            if filename_el is None or not (filename_el.text or "").strip():
                report.add_error(f"[pascal-voc] {xml_file.name}: missing <filename> element")
            else:
                file_name = filename_el.text.strip()
                if file_name not in available_images:
                    report.missing_images.append(file_name)
                    report.add_warning(f"[pascal-voc] {xml_file.name}: referenced image not found: {file_name}")

            objects = root.findall("object")
            if not objects:
                report.add_warning(f"[pascal-voc] {xml_file.name}: has no <object> elements")

            for obj in objects:
                name_el = obj.find("name")
                if name_el is None or not (name_el.text or "").strip():
                    report.add_error(f"[pascal-voc] {xml_file.name}: <object> missing <name>")
                    continue

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    if obj.find("polygon") is None:
                        report.add_error(
                            f"[pascal-voc] {xml_file.name}: object '{name_el.text}' missing <bndbox>"
                        )
                    continue

                try:
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                except (AttributeError, TypeError, ValueError):
                    report.add_error(
                        f"[pascal-voc] {xml_file.name}: <bndbox> has missing/non-numeric coordinates"
                    )
                    continue

                if xmax <= xmin or ymax <= ymin:
                    report.add_error(
                        f"[pascal-voc] {xml_file.name}: object '{name_el.text}' has invalid bndbox "
                        f"(xmax<=xmin or ymax<=ymin): ({xmin},{ymin},{xmax},{ymax})"
                    )
                object_count += 1

        report.annotations_count = object_count

        xml_stems = {f.stem for f in xml_files}
        images_without_annotation, _ = unmatched_stems(available_images, xml_stems)
        for stem in sorted(images_without_annotation):
            report.add_warning(f"[pascal-voc] Image has no matching annotation XML: {stem}")

        return report

    def build_package(self, config: PackConfig, report: ValidationReport) -> PackageResult:
        images_dir = _images_dir(config)
        annotations_dir = _annotations_dir(config)
        if images_dir is None or annotations_dir is None:
            raise ValidationFailedError("[pascal-voc] Cannot build package: missing JPEGImages or Annotations")

        staging = make_staging_dir()
        ann_out = ensure_dir(staging / "Annotations")
        for xml_file in annotations_dir.glob("*.xml"):
            safe_copy(xml_file, ann_out / xml_file.name)

        if config.copy_images:
            img_out = ensure_dir(staging / "JPEGImages")
            for img in find_images(images_dir).values():
                safe_copy(img, img_out / img.name)

        dataset_root = config.dataset
        if dataset_root is not None:
            image_sets = dataset_root / "ImageSets"
            if image_sets.exists():
                for f in image_sets.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(dataset_root)
                        safe_copy(f, staging / rel)
            for seg_dir_name in ("SegmentationClass", "SegmentationObject"):
                seg_dir = dataset_root / seg_dir_name
                if seg_dir.exists():
                    for f in seg_dir.rglob("*"):
                        if f.is_file():
                            rel = f.relative_to(dataset_root)
                            safe_copy(f, staging / rel)

        output_zip = build_zip(staging, config.output)
        report.output_zip = str(output_zip)
        return PackageResult(success=True, report=report, staging_dir=staging, output_zip=output_zip)
