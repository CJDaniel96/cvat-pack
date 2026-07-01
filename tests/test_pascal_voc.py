import shutil
from pathlib import Path

from cvat_packer.core.config import PackConfig
from cvat_packer.core.models import Status
from cvat_packer.formats.pascal_voc import PascalVocAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "pascal_voc"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "pascal_voc"
    shutil.copytree(FIXTURE, dest)
    return dest


def test_voc_validate_success(tmp_path):
    dataset = _copy_fixture(tmp_path)
    config = PackConfig(format="pascal-voc", output=tmp_path / "out.zip", dataset=dataset)
    report = PascalVocAdapter().validate(config)
    assert report.status == Status.SUCCESS
    assert report.images_count == 1
    assert report.annotations_count == 1


def test_voc_invalid_bndbox_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    xml_path = dataset / "Annotations" / "img1.xml"
    xml_path.write_text(xml_path.read_text().replace("<xmax>60</xmax>", "<xmax>5</xmax>"))

    config = PackConfig(format="pascal-voc", output=tmp_path / "out.zip", dataset=dataset)
    report = PascalVocAdapter().validate(config)
    assert report.status == Status.FAILED


def test_voc_missing_name_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    xml_path = dataset / "Annotations" / "img1.xml"
    xml_path.write_text(xml_path.read_text().replace("<name>cat</name>", ""))

    config = PackConfig(format="pascal-voc", output=tmp_path / "out.zip", dataset=dataset)
    report = PascalVocAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("missing <name>" in e for e in report.errors)


def test_voc_missing_image_is_warning(tmp_path):
    dataset = _copy_fixture(tmp_path)
    xml_path = dataset / "Annotations" / "img1.xml"
    xml_path.write_text(xml_path.read_text().replace("img1.jpg", "missing.jpg"))

    config = PackConfig(format="pascal-voc", output=tmp_path / "out.zip", dataset=dataset)
    report = PascalVocAdapter().validate(config)
    assert "missing.jpg" in report.missing_images


def test_voc_missing_annotations_dir_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    shutil.rmtree(dataset / "Annotations")

    config = PackConfig(format="pascal-voc", output=tmp_path / "out.zip", dataset=dataset)
    report = PascalVocAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("Annotations" in e for e in report.errors)


def test_voc_build_package(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    config = PackConfig(format="pascal-voc", output=output, dataset=dataset)
    adapter = PascalVocAdapter()
    report = adapter.validate(config)
    result = adapter.build_package(config, report)
    assert result.success
    assert output.exists()
