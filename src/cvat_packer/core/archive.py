"""Zip archive creation for the final CVAT upload package."""

from __future__ import annotations

import zipfile
from pathlib import Path

from cvat_packer.core.filesystem import ensure_dir, iter_files
from cvat_packer.utils.path_normalizer import is_safe_relative_path, normalize_arcname


def build_zip(staging_dir: Path, output_path: Path) -> Path:
    """Zip every file under staging_dir into output_path.

    - Skips junk files (already filtered out by iter_files).
    - Uses POSIX-style relative arcnames (Unicode-safe).
    - Refuses to write unsafe (absolute / traversal) paths.
    """
    ensure_dir(output_path.parent)
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in iter_files(staging_dir):
            arcname = normalize_arcname(staging_dir, file)
            if not is_safe_relative_path(arcname):
                raise ValueError(f"Unsafe path detected while archiving: {arcname}")
            zf.write(file, arcname)

    return output_path


def list_zip_entries(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.namelist()
