import json
import shutil
import zipfile
from pathlib import Path

from cvat_packer.core.config import PackConfig
from cvat_packer.core.models import Status
from cvat_packer.formats.coco import CocoAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "coco"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "coco"
    shutil.copytree(FIXTURE, dest)
    return dest


def test_coco_validate_success(tmp_path):
    dataset = _copy_fixture(tmp_path)
    config = PackConfig(format="coco", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoAdapter().validate(config)
    assert report.status == Status.SUCCESS
    assert report.images_count == 2
    assert report.annotations_count == 2
    assert not report.errors


def test_coco_missing_image_produces_warning(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "images" / "img1.jpg").unlink()
    config = PackConfig(format="coco", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoAdapter().validate(config)
    assert report.status == Status.WARNING
    assert "img1.jpg" in report.missing_images


def test_coco_orphan_annotation_detected(tmp_path):
    dataset = _copy_fixture(tmp_path)
    json_path = dataset / "annotations" / "instances_default.json"
    data = json.loads(json_path.read_text())
    data["annotations"].append(
        {"id": 99, "image_id": 999, "category_id": 1, "bbox": [0, 0, 1, 1]}
    )
    json_path.write_text(json.dumps(data))

    config = PackConfig(format="coco", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoAdapter().validate(config)
    assert "99" in report.orphan_annotations


def test_coco_invalid_bbox_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    json_path = dataset / "annotations" / "instances_default.json"
    data = json.loads(json_path.read_text())
    data["annotations"][0]["bbox"] = [0, 0, -5, 10]
    json_path.write_text(json.dumps(data))

    config = PackConfig(format="coco", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoAdapter().validate(config)
    assert report.status == Status.FAILED
    assert report.errors


def test_coco_malformed_segmentation_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    json_path = dataset / "annotations" / "instances_default.json"
    data = json.loads(json_path.read_text())
    data["annotations"][0]["segmentation"] = [[1, 2, 3]]  # odd length, too short
    json_path.write_text(json.dumps(data))

    config = PackConfig(format="coco", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoAdapter().validate(config)
    assert report.status == Status.FAILED


def test_coco_missing_top_level_key_is_error(tmp_path):
    dataset = _copy_fixture(tmp_path)
    json_path = dataset / "annotations" / "instances_default.json"
    data = json.loads(json_path.read_text())
    del data["categories"]
    json_path.write_text(json.dumps(data))

    config = PackConfig(format="coco", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("categories" in e for e in report.errors)


def test_coco_missing_json_gives_clear_error(tmp_path):
    dataset = tmp_path / "empty"
    dataset.mkdir()
    config = PackConfig(format="coco", output=tmp_path / "out.zip", dataset=dataset)
    report = CocoAdapter().validate(config)
    assert report.status == Status.FAILED
    assert "annotation JSON" in report.errors[0]


def test_coco_build_package(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    config = PackConfig(format="coco", output=output, dataset=dataset)
    adapter = CocoAdapter()
    report = adapter.validate(config)
    result = adapter.build_package(config, report)

    assert result.success
    assert output.exists()
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert "annotations/instances_default.json" in names
    assert any(n.startswith("images/") for n in names)


def test_coco_build_package_without_copy_images(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    config = PackConfig(format="coco", output=output, dataset=dataset, copy_images=False)
    adapter = CocoAdapter()
    report = adapter.validate(config)
    result = adapter.build_package(config, report)

    assert result.success
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert not any(n.startswith("images/") for n in names)
