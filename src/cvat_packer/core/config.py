"""Configuration object passed to every FormatAdapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PackConfig:
    """Everything a FormatAdapter needs to validate and package a dataset."""

    format: str
    output: Path

    images: Path | None = None
    annotations: Path | None = None
    dataset: Path | None = None
    labels: Path | None = None

    dry_run: bool = False
    validate_only: bool = False
    copy_images: bool = True
    strict: bool = False
    force: bool = False
    manifest: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        self.output = Path(self.output)
        if self.images is not None:
            self.images = Path(self.images)
        if self.annotations is not None:
            self.annotations = Path(self.annotations)
        if self.dataset is not None:
            self.dataset = Path(self.dataset)
        if self.labels is not None:
            self.labels = Path(self.labels)
