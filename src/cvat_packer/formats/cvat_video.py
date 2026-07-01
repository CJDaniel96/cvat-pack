"""CVAT for video 1.1 format adapter.

Expected CVAT upload layout::

    <root>/
      annotations.xml     # same schema as CVAT for images, but annotations
                          # are expressed as <track> elements with per-shape
                          # 'frame' attributes rather than one <image> per frame
      <video file>         # optional source video, e.g. clip.mp4 (if --copy-images)

`--dataset` should point at <root>. Pass `--images` at the source video file
if you want it bundled into the output zip.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from cvat_packer.core.archive import build_zip
from cvat_packer.core.config import PackConfig
from cvat_packer.core.errors import ValidationFailedError
from cvat_packer.core.filesystem import make_staging_dir, safe_copy
from cvat_packer.core.models import PackageResult, ValidationReport
from cvat_packer.core.registry import register
from cvat_packer.formats.base import FormatAdapter
from cvat_packer.formats.cvat_image import SHAPE_TAGS, find_xml

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _find_video(config: PackConfig) -> Path | None:
    if config.images and config.images.is_file() and config.images.suffix.lower() in VIDEO_EXTENSIONS:
        return config.images
    if config.dataset:
        for ext in VIDEO_EXTENSIONS:
            matches = sorted(config.dataset.glob(f"*{ext}"))
            if matches:
                return matches[0]
    return None


@register
class CvatVideoAdapter(FormatAdapter):
    format_name = "cvat-video"
    aliases = ["cvat-videos", "cvat_for_video", "cvat-video-1.1"]
    version = "1.1"
    supported_tasks = ["detection", "segmentation", "tracking"]
    supported_shapes = list(SHAPE_TAGS)
    supports_attributes = True
    supports_tracks = True
    display_name = "CVAT for video 1.1"

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
        return root.tag == "annotations" and root.find("track") is not None

    def validate(self, config: PackConfig) -> ValidationReport:
        report = ValidationReport(format=self.format_name)
        xml_path = find_xml(config)
        if xml_path is None:
            report.add_error(
                "[cvat-video] Could not find annotations.xml. Pass --annotations pointing "
                "at the XML file, or --dataset containing annotations.xml."
            )
            return report

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            report.add_error(f"[cvat-video] Failed to parse {xml_path.name}: {exc}")
            return report

        if root.tag != "annotations":
            report.add_error(f"[cvat-video] Root element must be <annotations>, got <{root.tag}>")
            return report

        video_path = _find_video(config)
        if video_path is None:
            report.add_warning(
                "[cvat-video] No source video file found (pass --images pointing at the "
                "video file if you want it bundled into the package)"
            )
        else:
            report.images_count = 1

        tracks = root.findall("track")
        if not tracks:
            report.add_error("[cvat-video] No <track> elements found; CVAT for video requires tracks")
            return report

        shape_count = 0
        max_frame = -1
        for track in tracks:
            if not track.get("label"):
                report.add_error("[cvat-video] <track> missing 'label' attribute")
            for tag in SHAPE_TAGS:
                for shape in track.findall(tag):
                    shape_count += 1
                    frame_attr = shape.get("frame")
                    if frame_attr is None:
                        report.add_error(
                            f"[cvat-video] track '{track.get('id')}' <{tag}> missing 'frame' attribute"
                        )
                        continue
                    try:
                        max_frame = max(max_frame, int(frame_attr))
                    except ValueError:
                        report.add_error(
                            f"[cvat-video] track '{track.get('id')}' <{tag}> has non-integer "
                            f"frame: {frame_attr}"
                        )

        report.annotations_count = shape_count
        if max_frame < 0:
            report.add_warning("[cvat-video] Could not determine frame range from track shapes")

        return report

    def build_package(self, config: PackConfig, report: ValidationReport) -> PackageResult:
        xml_path = find_xml(config)
        if xml_path is None:
            raise ValidationFailedError("[cvat-video] Cannot build package: annotations.xml not found")

        staging = make_staging_dir()
        safe_copy(xml_path, staging / "annotations.xml")

        if config.copy_images:
            video_path = _find_video(config)
            if video_path is not None:
                safe_copy(video_path, staging / video_path.name)

        output_zip = build_zip(staging, config.output)
        report.output_zip = str(output_zip)
        return PackageResult(success=True, report=report, staging_dir=staging, output_zip=output_zip)
