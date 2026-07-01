"""CVAT for images 1.1 format adapter.

Expected CVAT upload layout::

    <root>/
      annotations.xml
      images/                 # optional if --copy-images is used
        img1.jpg

annotations.xml structure (CVAT's native XML dump)::

    <annotations>
      <version>1.1</version>
      <meta>...</meta>
      <image id="0" name="img1.jpg" width="1920" height="1080">
        <box label="car" xtl="10" ytl="10" xbr="100" ybr="100" occluded="0">
          <attribute name="color">red</attribute>
        </box>
        <polygon label="lane" points="1,2;3,4;5,6" .../>
        <polyline .../> <points .../> <cuboid .../> <ellipse .../> <mask .../>
      </image>
      <track id="0" label="car"> <box frame="0" .../> ... </track>
    </annotations>

`--dataset` should point at <root>. Alternatively pass `--annotations` at
annotations.xml (or a folder containing it) and `--images` at the images
folder.
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
from cvat_packer.validators.common import find_images

SHAPE_TAGS = ("box", "polygon", "polyline", "points", "cuboid", "ellipse", "mask", "skeleton", "tag")


def find_xml(config: PackConfig) -> Path | None:
    candidates: list[Path] = []
    if config.annotations:
        if config.annotations.is_file() and config.annotations.suffix == ".xml":
            candidates.append(config.annotations)
        elif config.annotations.is_dir():
            candidates.append(config.annotations / "annotations.xml")
            candidates.extend(sorted(config.annotations.glob("*.xml")))
    if config.dataset:
        candidates.append(config.dataset / "annotations.xml")
        candidates.extend(sorted(config.dataset.glob("*.xml")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _images_dir(config: PackConfig) -> Path | None:
    if config.images and config.images.exists():
        return config.images
    if config.dataset and (config.dataset / "images").exists():
        return config.dataset / "images"
    return None


@register
class CvatImageAdapter(FormatAdapter):
    format_name = "cvat-image"
    aliases = ["cvat-images", "cvat_for_images", "cvat-images-1.1", "cvat"]
    version = "1.1"
    supported_tasks = ["detection", "segmentation", "keypoints"]
    supported_shapes = list(SHAPE_TAGS)
    supports_attributes = True
    supports_tracks = True
    display_name = "CVAT for images 1.1"

    def detect(self, input_path: Path) -> bool:
        if input_path is None or not input_path.exists():
            return False
        xml_path = input_path / "annotations.xml" if input_path.is_dir() else input_path
        if not xml_path.exists() or xml_path.suffix != ".xml":
            return False
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            return False
        return root.tag == "annotations"

    def validate(self, config: PackConfig) -> ValidationReport:
        report = ValidationReport(format=self.format_name)
        xml_path = find_xml(config)
        if xml_path is None:
            report.add_error(
                "[cvat-image] Could not find annotations.xml. Pass --annotations pointing "
                "at the XML file, or --dataset containing annotations.xml."
            )
            return report

        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            report.add_error(f"[cvat-image] Failed to parse {xml_path.name}: {exc}")
            return report

        root = tree.getroot()
        if root.tag != "annotations":
            report.add_error(f"[cvat-image] Root element must be <annotations>, got <{root.tag}>")
            return report

        images_dir = _images_dir(config)
        available_images = find_images(images_dir)

        image_elements = root.findall("image")
        report.images_count = len(image_elements)

        shape_count = 0
        for image_el in image_elements:
            name = image_el.get("name")
            if not name:
                report.add_error("[cvat-image] <image> element missing 'name' attribute")
                continue

            base_name = Path(name).name
            if images_dir is not None and base_name not in available_images:
                report.missing_images.append(name)
                report.add_warning(f"[cvat-image] Referenced image not found on disk: {name}")

            for tag in SHAPE_TAGS:
                for shape in image_el.findall(tag):
                    shape_count += 1
                    if not shape.get("label"):
                        report.add_error(
                            f"[cvat-image] image '{name}' has a <{tag}> without a 'label' attribute"
                        )
                    for attr in shape.findall("attribute"):
                        if not attr.get("name"):
                            report.add_warning(
                                f"[cvat-image] image '{name}' <{tag}> has an <attribute> without 'name'"
                            )

        track_elements = root.findall("track")
        for track in track_elements:
            if not track.get("label"):
                report.add_error("[cvat-image] <track> missing 'label' attribute")
            for tag in SHAPE_TAGS:
                shape_count += len(track.findall(tag))

        report.annotations_count = shape_count

        if not image_elements and not track_elements:
            report.add_warning("[cvat-image] No <image> or <track> elements found in annotations.xml")

        return report

    def build_package(self, config: PackConfig, report: ValidationReport) -> PackageResult:
        xml_path = find_xml(config)
        if xml_path is None:
            raise ValidationFailedError("[cvat-image] Cannot build package: annotations.xml not found")

        staging = make_staging_dir()
        safe_copy(xml_path, staging / "annotations.xml")

        if config.copy_images:
            images_dir = _images_dir(config)
            if images_dir is not None:
                img_out = ensure_dir(staging / "images")
                for img in find_images(images_dir).values():
                    safe_copy(img, img_out / img.name)

        output_zip = build_zip(staging, config.output)
        report.output_zip = str(output_zip)
        return PackageResult(success=True, report=report, staging_dir=staging, output_zip=output_zip)
