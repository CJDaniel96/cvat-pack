"""Manifest (JSON) writing and terminal summary formatting."""

from __future__ import annotations

import json
from pathlib import Path

from cvat_packer.core.models import ValidationReport


def write_manifest(report: ValidationReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def format_summary(report: ValidationReport, display_name: str) -> str:
    lines = [
        f"Format: {display_name}",
        f"Images: {report.images_count}",
        f"Annotations: {report.annotations_count}",
        f"Warnings: {len(report.warnings)}",
        f"Errors: {len(report.errors)}",
        f"Output: {report.output_zip or '(not generated)'}",
    ]
    return "\n".join(lines)
