"""Post-build validation of the produced zip's internal structure."""

from __future__ import annotations

from pathlib import Path

from cvat_packer.core.archive import list_zip_entries
from cvat_packer.core.filesystem import IGNORED_DIR_NAMES, IGNORED_FILENAMES
from cvat_packer.utils.path_normalizer import is_safe_relative_path


def validate_zip_structure(zip_path: Path) -> list[str]:
    """Return a list of problems found in the zip (empty list == clean)."""
    problems: list[str] = []
    for name in list_zip_entries(zip_path):
        if not is_safe_relative_path(name):
            problems.append(f"Unsafe path in zip: {name}")
        parts = name.split("/")
        if parts[-1] in IGNORED_FILENAMES:
            problems.append(f"Junk file present in zip: {name}")
        if any(part in IGNORED_DIR_NAMES for part in parts):
            problems.append(f"Junk directory present in zip: {name}")
    return problems
