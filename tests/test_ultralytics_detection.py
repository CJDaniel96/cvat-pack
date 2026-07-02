import shutil
import zipfile
from pathlib import Path

from cvat_packer.core.config import PackConfig
from cvat_packer.core.models import Status
from cvat_packer.formats.ultralytics_yolo import UltralyticsYoloDetectionAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "ultralytics_detection"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "ultralytics_detection"
    shutil.copytree(FIXTURE, dest)
    return dest


def test_detection_validate_success(tmp_path):
    dataset = _copy_fixture(tmp_path)
    config = PackConfig(format="ultralytics-yolo-detection", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloDetectionAdapter().validate(config)
    assert report.status == Status.SUCCESS
    assert report.images_count == 2
    assert report.annotations_count == 2
    assert not report.errors


def test_detection_missing_data_yaml_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "data.yaml").unlink()
    config = PackConfig(format="ultralytics-yolo-detection", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloDetectionAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("class names" in e for e in report.errors)


def test_detection_no_names_in_data_yaml_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "data.yaml").write_text("train: images/train\nval: images/val\n")
    config = PackConfig(format="ultralytics-yolo-detection", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloDetectionAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("class names" in e for e in report.errors)


def test_detection_wrong_field_count_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.2\n")
    config = PackConfig(format="ultralytics-yolo-detection", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloDetectionAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("5 fields" in e for e in report.errors)


def test_detection_out_of_range_coordinate_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "labels" / "train" / "img1.txt").write_text("0 1.5 0.5 0.2 0.2\n")
    config = PackConfig(format="ultralytics-yolo-detection", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloDetectionAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("normalized" in e for e in report.errors)


def test_detection_class_id_out_of_range_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "labels" / "train" / "img1.txt").write_text("9 0.5 0.5 0.2 0.2\n")
    config = PackConfig(format="ultralytics-yolo-detection", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloDetectionAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("out of range" in e for e in report.errors)


def test_detection_missing_image_is_warning(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "images" / "train" / "img1.jpg").unlink()
    config = PackConfig(format="ultralytics-yolo-detection", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloDetectionAdapter().validate(config)
    assert any("train/img1.txt" in o for o in report.orphan_annotations)


def test_detection_no_split_folders_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    shutil.rmtree(dataset / "images" / "train")
    shutil.rmtree(dataset / "images" / "val")
    config = PackConfig(format="ultralytics-yolo-detection", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloDetectionAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("No split folders" in e for e in report.errors)


def test_detection_build_package_zip_structure(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    config = PackConfig(format="ultralytics-yolo-detection", output=output, dataset=dataset)
    adapter = UltralyticsYoloDetectionAdapter()
    report = adapter.validate(config)
    result = adapter.build_package(config, report)

    assert result.success
    assert output.exists()
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert "data.yaml" in names
    assert "images/train/img1.jpg" in names
    assert "labels/train/img1.txt" in names


def test_detection_dry_run(tmp_path):
    from cvat_packer.cli import main

    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    exit_code = main(
        [
            "--format", "ultralytics-yolo-detection",
            "--dataset", str(dataset),
            "--output", str(output),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not output.exists()


def test_detection_validate_only(tmp_path):
    from cvat_packer.cli import main

    dataset = _copy_fixture(tmp_path)
    exit_code = main(
        ["--format", "ultralytics-yolo-detection", "--dataset", str(dataset), "--validate-only"]
    )
    assert exit_code == 0
