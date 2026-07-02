import shutil
import zipfile
from pathlib import Path

from cvat_packer.core.config import PackConfig
from cvat_packer.core.models import Status
from cvat_packer.formats.ultralytics_yolo import UltralyticsYoloPoseAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "ultralytics_pose"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "ultralytics_pose"
    shutil.copytree(FIXTURE, dest)
    return dest


def test_pose_validate_success(tmp_path):
    dataset = _copy_fixture(tmp_path)
    config = PackConfig(format="ultralytics-yolo-pose", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloPoseAdapter().validate(config)
    assert report.status == Status.SUCCESS
    assert report.images_count == 2
    assert report.annotations_count == 2
    assert not report.errors


def test_pose_missing_kpt_shape_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(data_yaml.read_text().replace("kpt_shape: [3, 3]\n", ""))
    config = PackConfig(format="ultralytics-yolo-pose", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloPoseAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("kpt_shape" in e for e in report.errors)


def test_pose_wrong_keypoint_count_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    # kpt_shape says 3 keypoints x 3 dims = 9 values; this line only has 6
    (dataset / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.3 0.3 0.4 0.4 2 0.5 0.5 1\n")
    config = PackConfig(format="ultralytics-yolo-pose", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloPoseAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("keypoint values" in e for e in report.errors)


def test_pose_invalid_visibility_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "labels" / "train" / "img1.txt").write_text(
        "0 0.5 0.5 0.3 0.3 0.4 0.4 9 0.5 0.5 1 0.6 0.6 0\n"
    )
    config = PackConfig(format="ultralytics-yolo-pose", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloPoseAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("visibility" in e for e in report.errors)


def test_pose_visibility_2_is_valid(tmp_path):
    """Regression test: visibility=2 must not be rejected as 'out of [0,1]'."""
    dataset = _copy_fixture(tmp_path)
    (dataset / "labels" / "train" / "img1.txt").write_text(
        "0 0.5 0.5 0.3 0.3 0.4 0.4 2 0.5 0.5 2 0.6 0.6 2\n"
    )
    config = PackConfig(format="ultralytics-yolo-pose", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloPoseAdapter().validate(config)
    assert report.status == Status.SUCCESS
    assert not report.errors


def test_pose_class_id_out_of_range_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "labels" / "train" / "img1.txt").write_text(
        "9 0.5 0.5 0.3 0.3 0.4 0.4 2 0.5 0.5 1 0.6 0.6 0\n"
    )
    config = PackConfig(format="ultralytics-yolo-pose", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloPoseAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("out of range" in e for e in report.errors)


def test_pose_missing_image_is_warning(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "images" / "train" / "img1.jpg").unlink()
    config = PackConfig(format="ultralytics-yolo-pose", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloPoseAdapter().validate(config)
    assert any("train/img1.txt" in o for o in report.orphan_annotations)


def test_pose_build_package_zip_structure(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    config = PackConfig(format="ultralytics-yolo-pose", output=output, dataset=dataset)
    adapter = UltralyticsYoloPoseAdapter()
    report = adapter.validate(config)
    result = adapter.build_package(config, report)

    assert result.success
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert "data.yaml" in names
    assert "labels/train/img1.txt" in names


def test_pose_dry_run(tmp_path):
    from cvat_packer.cli import main

    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    exit_code = main(
        ["--format", "ultralytics-yolo-pose", "--dataset", str(dataset), "--output", str(output), "--dry-run"]
    )
    assert exit_code == 0
    assert not output.exists()


def test_pose_validate_only(tmp_path):
    from cvat_packer.cli import main

    dataset = _copy_fixture(tmp_path)
    exit_code = main(["--format", "ultralytics-yolo-pose", "--dataset", str(dataset), "--validate-only"])
    assert exit_code == 0
