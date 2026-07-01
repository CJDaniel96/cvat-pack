"""Abstract base classes every CVAT format adapter builds on.

`FormatAdapter` is the contract each format (COCO, YOLO, Pascal VOC, ...)
must implement. `SkeletonFormatAdapter` is a convenience base for formats
that only have structural (folder/file existence) validation implemented so
far -- see the Phase 3 stub adapters under cvat_packer/formats/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from cvat_packer.core.archive import build_zip
from cvat_packer.core.config import PackConfig
from cvat_packer.core.filesystem import IMAGE_EXTENSIONS, iter_files, make_staging_dir, safe_copytree
from cvat_packer.core.models import PackageResult, ValidationReport


class FormatAdapter(ABC):
    """Base contract for a single CVAT dataset format."""

    format_name: str = ""
    aliases: Sequence[str] = ()
    version: str = "1.0"
    supported_tasks: Sequence[str] = ()
    supported_shapes: Sequence[str] = ()
    supports_attributes: bool = False
    supports_tracks: bool = False

    #: Human-friendly name shown in CLI output, e.g. "COCO 1.0".
    display_name: str = ""

    @abstractmethod
    def detect(self, input_path: Path) -> bool:
        """Return True if input_path looks like this format."""

    @abstractmethod
    def validate(self, config: PackConfig) -> ValidationReport:
        """Validate input data against this format's rules.

        Must never raise for "expected" problems (missing files, malformed
        annotations, etc.) -- those should be reported as errors/warnings on
        the returned ValidationReport so the CLI can print/act on them.
        """

    @abstractmethod
    def build_package(self, config: PackConfig, report: ValidationReport) -> PackageResult:
        """Assemble a staging directory in the target layout and zip it.

        `report` is the ValidationReport produced by a prior call to
        validate() -- implementations should update it (e.g. `output_zip`)
        rather than build a new one.
        """

    def resolve_root(self, config: PackConfig) -> Path | None:
        """Best-effort single root directory for formats that don't need to
        distinguish between --images/--annotations/--dataset."""
        if config.dataset:
            return config.dataset
        if config.images:
            return config.images
        if config.annotations:
            return config.annotations
        return None


class SkeletonFormatAdapter(FormatAdapter):
    """Base class for formats with structural validation + best-effort
    packaging only (Phase 3 candidates not yet fully implemented).

    Subclasses set `required_any` (glob patterns relative to the resolved
    root, at least one of which must match) and `structure_hint` (a short,
    human-readable description of the expected directory layout) and get a
    working detect()/validate()/build_package() for free.
    """

    required_any: Sequence[str] = ()
    structure_hint: str = ""

    def detect(self, input_path: Path) -> bool:
        if input_path is None or not input_path.exists():
            return False
        for pattern in self.required_any:
            if list(input_path.glob(pattern)):
                return True
        return False

    def validate(self, config: PackConfig) -> ValidationReport:
        report = ValidationReport(format=self.format_name)
        root = self.resolve_root(config)
        if root is None or not root.exists():
            report.add_error(
                f"[{self.format_name}] No input found. Provide --dataset, --images "
                f"or --annotations pointing at a folder with: {self.structure_hint}"
            )
            return report

        found_any = any(list(root.glob(pattern)) for pattern in self.required_any)
        if not found_any:
            report.add_error(
                f"[{self.format_name}] Expected structure not found under {root}. "
                f"Required (any of): {list(self.required_any)}. Hint: {self.structure_hint}"
            )
            return report

        report.images_count = sum(
            1 for f in iter_files(root) if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        report.add_warning(
            f"[{self.format_name}] Only structural validation is implemented for this "
            "format so far (Phase 3 TODO). Files will be packaged as-is without "
            "content-level checks (bbox ranges, label/image pairing, etc.)."
        )
        return report

    def build_package(self, config: PackConfig, report: ValidationReport) -> PackageResult:
        root = self.resolve_root(config)
        staging = make_staging_dir()
        safe_copytree(root, staging, copy_images=config.copy_images)
        output_zip = build_zip(staging, config.output)
        report.output_zip = str(output_zip)
        return PackageResult(success=True, report=report, staging_dir=staging, output_zip=output_zip)
