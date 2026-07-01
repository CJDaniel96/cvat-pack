import shutil
import zipfile
from pathlib import Path

from cvat_packer.core.config import PackConfig
from cvat_packer.formats.cvat_image import CvatImageAdapter
from cvat_packer.validators.zip_structure import validate_zip_structure

FIXTURE = Path(__file__).parent / "fixtures" / "cvat_image"


def _build(tmp_path: Path, add_junk: bool = False):
    dataset = tmp_path / "cvat_image"
    shutil.copytree(FIXTURE, dataset)
    if add_junk:
        (dataset / ".DS_Store").write_text("junk")
        (dataset / "images" / "Thumbs.db").write_text("junk")
        macosx = dataset / "__MACOSX"
        macosx.mkdir()
        (macosx / "leftover").write_text("junk")

    output = tmp_path / "out.zip"
    config = PackConfig(format="cvat-image", output=output, dataset=dataset)
    adapter = CvatImageAdapter()
    report = adapter.validate(config)
    adapter.build_package(config, report)
    return output


def test_zip_has_no_junk_files(tmp_path):
    output = _build(tmp_path, add_junk=True)
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert ".DS_Store" not in names
    assert not any(n.endswith("Thumbs.db") for n in names)
    assert not any("__MACOSX" in n for n in names)
    assert not validate_zip_structure(output)


def test_zip_uses_posix_paths(tmp_path):
    output = _build(tmp_path)
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert names, "zip should not be empty"
    assert all("\\" not in n for n in names)
    assert all(not n.startswith("/") for n in names)


def test_zip_contains_expected_entries(tmp_path):
    output = _build(tmp_path)
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
    assert "annotations.xml" in names
    assert "images/img1.jpg" in names
