import shutil
from pathlib import Path

from cvat_packer.core.config import PackConfig
from cvat_packer.core.models import Status
from cvat_packer.formats.yolo import YoloAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "yolo"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "yolo"
    shutil.copytree(FIXTURE, dest)
    return dest


def test_yolo_validate_success(tmp_path):
    dataset = _copy_fixture(tmp_path)
    config = PackConfig(format="yolo", output=tmp_path / "out.zip", dataset=dataset)
    report = YoloAdapter().validate(config)
    assert report.status == Status.SUCCESS
    assert report.images_count == 2
    assert report.annotations_count == 2


def test_yolo_out_of_range_coordinate_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "obj_train_data" / "img1.txt").write_text("0 1.5 0.5 0.2 0.2\n")
    config = PackConfig(format="yolo", output=tmp_path / "out.zip", dataset=dataset)
    report = YoloAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("normalized" in e for e in report.errors)


def test_yolo_wrong_field_count_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "obj_train_data" / "img1.txt").write_text("0 0.5 0.5 0.2\n")
    config = PackConfig(format="yolo", output=tmp_path / "out.zip", dataset=dataset)
    report = YoloAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("5 fields" in e for e in report.errors)


def test_yolo_class_id_out_of_range_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "obj_train_data" / "img1.txt").write_text("9 0.5 0.5 0.2 0.2\n")
    config = PackConfig(format="yolo", output=tmp_path / "out.zip", dataset=dataset)
    report = YoloAdapter().validate(config)
    assert report.status == Status.FAILED


def test_yolo_orphan_label_file_is_warning(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "obj_train_data" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    config = PackConfig(format="yolo", output=tmp_path / "out.zip", dataset=dataset)
    report = YoloAdapter().validate(config)
    assert "orphan.txt" in report.orphan_annotations


def test_yolo_missing_classes_file_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "obj.names").unlink()
    config = PackConfig(format="yolo", output=tmp_path / "out.zip", dataset=dataset)
    report = YoloAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("class names" in e for e in report.errors)


def test_yolo_missing_labels_dir_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    shutil.rmtree(dataset / "obj_train_data")
    config = PackConfig(format="yolo", output=tmp_path / "out.zip", dataset=dataset)
    report = YoloAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("label .txt files" in e for e in report.errors)


def test_yolo_build_package(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    config = PackConfig(format="yolo", output=output, dataset=dataset)
    adapter = YoloAdapter()
    report = adapter.validate(config)
    result = adapter.build_package(config, report)
    assert result.success
    assert output.exists()
