import json
import shutil
import zipfile
from pathlib import Path

from cvat_packer.core.config import PackConfig
from cvat_packer.core.models import Status
from cvat_packer.formats.coco_keypoints import CocoKeypointsAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "coco_keypoints"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "coco_keypoints"
    shutil.copytree(FIXTURE, dest)
    return dest


def _json_path(dataset: Path) -> Path:
    return dataset / "annotations" / "instances_default.json"


def test_coco_keypoints_validate_success(tmp_path):
    dataset = _copy_fixture(tmp_path)
    config = PackConfig(format="coco-keypoints", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoKeypointsAdapter().validate(config)
    assert report.status == Status.SUCCESS
    assert report.images_count == 2
    assert report.annotations_count == 2
    assert not report.errors


def test_coco_keypoints_missing_json_gives_clear_error(tmp_path):
    dataset = tmp_path / "empty"
    dataset.mkdir()
    config = PackConfig(format="coco-keypoints", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoKeypointsAdapter().validate(config)
    assert report.status == Status.FAILED
    assert "annotation JSON" in report.errors[0]


def test_coco_keypoints_missing_num_keypoints_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    json_path = _json_path(dataset)
    data = json.loads(json_path.read_text())
    del data["annotations"][0]["num_keypoints"]
    json_path.write_text(json.dumps(data))

    config = PackConfig(format="coco-keypoints", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoKeypointsAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("num_keypoints" in e for e in report.errors)


def test_coco_keypoints_length_mismatch_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    json_path = _json_path(dataset)
    data = json.loads(json_path.read_text())
    data["annotations"][0]["num_keypoints"] = 5  # keypoints array only has 3 points
    json_path.write_text(json.dumps(data))

    config = PackConfig(format="coco-keypoints", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoKeypointsAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("num_keypoints * 3" in e for e in report.errors)


def test_coco_keypoints_invalid_visibility_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    json_path = _json_path(dataset)
    data = json.loads(json_path.read_text())
    data["annotations"][0]["keypoints"][2] = 9  # visibility must be 0, 1, or 2
    json_path.write_text(json.dumps(data))

    config = PackConfig(format="coco-keypoints", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoKeypointsAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("visibility" in e for e in report.errors)


def test_coco_keypoints_missing_bbox_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    json_path = _json_path(dataset)
    data = json.loads(json_path.read_text())
    del data["annotations"][0]["bbox"]
    json_path.write_text(json.dumps(data))

    config = PackConfig(format="coco-keypoints", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoKeypointsAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("bbox" in e for e in report.errors)


def test_coco_keypoints_missing_image_is_warning(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "images" / "img1.jpg").unlink()

    config = PackConfig(format="coco-keypoints", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoKeypointsAdapter().validate(config)
    assert "img1.jpg" in report.missing_images


def test_coco_keypoints_no_keypoints_field_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    json_path = _json_path(dataset)
    data = json.loads(json_path.read_text())
    for ann in data["annotations"]:
        del ann["keypoints"]
    json_path.write_text(json.dumps(data))

    config = PackConfig(format="coco-keypoints", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoKeypointsAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("keypoints" in e for e in report.errors)


def test_coco_keypoints_build_package_zip_structure(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    config = PackConfig(format="coco-keypoints", output=output, dataset=dataset)
    adapter = CocoKeypointsAdapter()
    report = adapter.validate(config)
    result = adapter.build_package(config, report)

    assert result.success
    assert output.exists()
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert "annotations/instances_default.json" in names
    assert any(n.startswith("images/") for n in names)


def test_coco_keypoints_dry_run(tmp_path):
    from cvat_packer.cli import main

    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    exit_code = main(
        ["--format", "coco-keypoints", "--dataset", str(dataset), "--output", str(output), "--dry-run"]
    )
    assert exit_code == 0
    assert not output.exists()


def test_coco_keypoints_validate_only(tmp_path):
    from cvat_packer.cli import main

    dataset = _copy_fixture(tmp_path)
    exit_code = main(["--format", "coco-keypoints", "--dataset", str(dataset), "--validate-only"])
    assert exit_code == 0
