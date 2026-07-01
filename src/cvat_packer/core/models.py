"""Shared result/report data structures used across all format adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Status(str, Enum):
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class ValidationReport:
    """Result of FormatAdapter.validate().

    `status` starts as SUCCESS and is only ever escalated (never downgraded)
    by add_warning()/add_error(), so adapters can call them in any order.
    """

    format: str
    status: Status = Status.SUCCESS
    images_count: int = 0
    annotations_count: int = 0
    missing_images: list[str] = field(default_factory=list)
    orphan_annotations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    output_zip: str | None = None

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        if self.status == Status.SUCCESS:
            self.status = Status.WARNING

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.status = Status.FAILED

    def ok(self, strict: bool = False) -> bool:
        """Whether this report should be treated as passing.

        In strict mode, warnings are treated the same as errors.
        """
        if self.status == Status.FAILED:
            return False
        if strict and self.status == Status.WARNING:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "status": self.status.value,
            "images_count": self.images_count,
            "annotations_count": self.annotations_count,
            "missing_images": self.missing_images,
            "orphan_annotations": self.orphan_annotations,
            "warnings": self.warnings,
            "errors": self.errors,
            "output_zip": self.output_zip,
        }


@dataclass
class PackageResult:
    """Result of FormatAdapter.build_package()."""

    success: bool
    report: ValidationReport
    staging_dir: Path | None = None
    output_zip: Path | None = None
    files_written: list[str] = field(default_factory=list)
