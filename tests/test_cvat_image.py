import shutil
import zipfile
from pathlib import Path

from cvat_packer.core.config import PackConfig
from cvat_packer.core.models import Status
from cvat_packer.formats.cvat_image import CvatImageAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "cvat_image"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "cvat_image"
    shutil.copytree(FIXTURE, dest)
    return dest


def test_cvat_image_validate_success(tmp_path):
    dataset = _copy_fixture(tmp_path)
    config = PackConfig(format="cvat-image", output=tmp_path / "out.zip", dataset=dataset)
    report = CvatImageAdapter().validate(config)
    assert report.status == Status.SUCCESS
    assert report.images_count == 1
    assert report.annotations_count == 1
    assert not report.errors


def test_cvat_image_missing_annotations_xml_is_error(tmp_path):
    dataset = tmp_path / "empty"
    dataset.mkdir()
    config = PackConfig(format="cvat-image", output=tmp_path / "out.zip", dataset=dataset)
    report = CvatImageAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("annotations.xml" in e for e in report.errors)


def test_cvat_image_malformed_xml_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "annotations.xml").write_text("<annotations><unclosed>")
    config = PackConfig(format="cvat-image", output=tmp_path / "out.zip", dataset=dataset)
    report = CvatImageAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("Failed to parse" in e for e in report.errors)


def test_cvat_image_wrong_root_element_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "annotations.xml").write_text("<not_annotations></not_annotations>")
    config = PackConfig(format="cvat-image", output=tmp_path / "out.zip", dataset=dataset)
    report = CvatImageAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("<annotations>" in e for e in report.errors)


def test_cvat_image_shape_missing_label_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    xml_path = dataset / "annotations.xml"
    xml_path.write_text(xml_path.read_text().replace('label="cat" ', ""))
    config = PackConfig(format="cvat-image", output=tmp_path / "out.zip", dataset=dataset)
    report = CvatImageAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("without a 'label' attribute" in e for e in report.errors)


def test_cvat_image_missing_image_on_disk_is_warning(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "images" / "img1.jpg").unlink()
    config = PackConfig(format="cvat-image", output=tmp_path / "out.zip", dataset=dataset)
    report = CvatImageAdapter().validate(config)
    assert "img1.jpg" in report.missing_images
    assert report.status == Status.WARNING


def test_cvat_image_no_images_no_tracks_is_warning(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "annotations.xml").write_text(
        '<?xml version="1.0"?><annotations><version>1.1</version></annotations>'
    )
    config = PackConfig(format="cvat-image", output=tmp_path / "out.zip", dataset=dataset)
    report = CvatImageAdapter().validate(config)
    assert any("No <image> or <track>" in w for w in report.warnings)


def test_cvat_image_build_package_zip_structure(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    config = PackConfig(format="cvat-image", output=output, dataset=dataset)
    adapter = CvatImageAdapter()
    report = adapter.validate(config)
    result = adapter.build_package(config, report)

    assert result.success
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert "annotations.xml" in names
    assert "images/img1.jpg" in names


def test_cvat_image_dry_run(tmp_path):
    from cvat_packer.cli import main

    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    exit_code = main(
        ["--format", "cvat-image", "--dataset", str(dataset), "--output", str(output), "--dry-run"]
    )
    assert exit_code == 0
    assert not output.exists()


def test_cvat_image_validate_only(tmp_path):
    from cvat_packer.cli import main

    dataset = _copy_fixture(tmp_path)
    exit_code = main(["--format", "cvat-image", "--dataset", str(dataset), "--validate-only"])
    assert exit_code == 0
