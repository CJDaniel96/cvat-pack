import shutil
import zipfile
from pathlib import Path

from cvat_packer.core.config import PackConfig
from cvat_packer.core.models import Status
from cvat_packer.formats.ultralytics_yolo import UltralyticsYoloSegmentationAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "ultralytics_segmentation"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "ultralytics_segmentation"
    shutil.copytree(FIXTURE, dest)
    return dest


def test_segmentation_validate_success(tmp_path):
    dataset = _copy_fixture(tmp_path)
    config = PackConfig(format="ultralytics-yolo-segmentation", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloSegmentationAdapter().validate(config)
    assert report.status == Status.SUCCESS
    assert report.images_count == 2
    assert report.annotations_count == 2
    assert not report.errors


def test_segmentation_missing_data_yaml_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "data.yaml").unlink()
    config = PackConfig(format="ultralytics-yolo-segmentation", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloSegmentationAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("class names" in e for e in report.errors)


def test_segmentation_odd_polygon_coordinates_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    # class + 7 coordinate values (odd count) -- above the 3-point minimum but not (x,y) paired
    (dataset / "labels" / "train" / "img1.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.9\n")
    config = PackConfig(format="ultralytics-yolo-segmentation", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloSegmentationAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("pairs" in e for e in report.errors)


def test_segmentation_too_few_points_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    # class + 2 (x,y) points = 5 fields, below the 3-point (7 field) minimum
    (dataset / "labels" / "train" / "img1.txt").write_text("0 0.1 0.1 0.5 0.5\n")
    config = PackConfig(format="ultralytics-yolo-segmentation", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloSegmentationAdapter().validate(config)
    assert report.status == Status.FAILED


def test_segmentation_out_of_range_coordinate_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "labels" / "train" / "img1.txt").write_text("0 0.1 0.1 1.5 0.1 0.5 0.5\n")
    config = PackConfig(format="ultralytics-yolo-segmentation", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloSegmentationAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("normalized" in e for e in report.errors)


def test_segmentation_class_id_out_of_range_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "labels" / "train" / "img1.txt").write_text("9 0.1 0.1 0.5 0.1 0.5 0.5\n")
    config = PackConfig(format="ultralytics-yolo-segmentation", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloSegmentationAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("out of range" in e for e in report.errors)


def test_segmentation_missing_image_is_warning(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "images" / "train" / "img1.jpg").unlink()
    config = PackConfig(format="ultralytics-yolo-segmentation", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloSegmentationAdapter().validate(config)
    assert any("train/img1.txt" in o for o in report.orphan_annotations)


def test_segmentation_build_package_zip_structure(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    config = PackConfig(format="ultralytics-yolo-segmentation", output=output, dataset=dataset)
    adapter = UltralyticsYoloSegmentationAdapter()
    report = adapter.validate(config)
    result = adapter.build_package(config, report)

    assert result.success
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert "data.yaml" in names
    assert "images/train/img1.jpg" in names
    assert "labels/train/img1.txt" in names


def test_segmentation_dry_run(tmp_path):
    from cvat_packer.cli import main

    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    exit_code = main(
        [
            "--format", "ultralytics-yolo-segmentation",
            "--dataset", str(dataset),
            "--output", str(output),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not output.exists()


def test_segmentation_validate_only(tmp_path):
    from cvat_packer.cli import main

    dataset = _copy_fixture(tmp_path)
    exit_code = main(
        ["--format", "ultralytics-yolo-segmentation", "--dataset", str(dataset), "--validate-only"]
    )
    assert exit_code == 0
