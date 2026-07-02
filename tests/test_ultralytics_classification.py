import shutil
import zipfile
from pathlib import Path

from cvat_packer.core.config import PackConfig
from cvat_packer.core.models import Status
from cvat_packer.formats.ultralytics_yolo import UltralyticsYoloClassificationAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "ultralytics_classification"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "ultralytics_classification"
    shutil.copytree(FIXTURE, dest)
    return dest


def test_classification_validate_success(tmp_path):
    dataset = _copy_fixture(tmp_path)
    config = PackConfig(format="ultralytics-yolo-classification", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloClassificationAdapter().validate(config)
    assert report.status == Status.SUCCESS
    assert report.images_count == 4
    assert report.annotations_count == 4
    assert not report.errors


def test_classification_no_split_folders_is_error(tmp_path):
    dataset = tmp_path / "empty"
    dataset.mkdir()
    config = PackConfig(format="ultralytics-yolo-classification", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloClassificationAdapter().validate(config)
    assert report.status == Status.FAILED
    assert any("No split folders" in e for e in report.errors)


def test_classification_empty_class_folder_is_warning(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "train" / "cat" / "img1.jpg").unlink()
    config = PackConfig(format="ultralytics-yolo-classification", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloClassificationAdapter().validate(config)
    assert any("no images" in w for w in report.warnings)


def test_classification_invalid_extension_is_warning(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "train" / "cat" / "notes.txt").write_text("not an image")
    config = PackConfig(format="ultralytics-yolo-classification", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloClassificationAdapter().validate(config)
    assert report.status == Status.WARNING
    assert any("unrecognized image extensions" in w for w in report.warnings)


def test_classification_no_class_subfolders_is_error(tmp_path):
    dataset = tmp_path / "ds"
    (dataset / "train").mkdir(parents=True)
    (dataset / "train" / "loose_image.jpg").write_text("fake")
    config = PackConfig(format="ultralytics-yolo-classification", output=tmp_path / "out.zip", dataset=dataset)
    report = UltralyticsYoloClassificationAdapter().validate(config)
    assert report.status == Status.FAILED


def test_classification_build_package_zip_structure(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    config = PackConfig(format="ultralytics-yolo-classification", output=output, dataset=dataset)
    adapter = UltralyticsYoloClassificationAdapter()
    report = adapter.validate(config)
    result = adapter.build_package(config, report)

    assert result.success
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert "train/cat/img1.jpg" in names
    assert "val/dog/img4.jpg" in names


def test_classification_dry_run(tmp_path):
    from cvat_packer.cli import main

    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    exit_code = main(
        [
            "--format", "ultralytics-yolo-classification",
            "--dataset", str(dataset),
            "--output", str(output),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not output.exists()


def test_classification_validate_only(tmp_path):
    from cvat_packer.cli import main

    dataset = _copy_fixture(tmp_path)
    exit_code = main(
        ["--format", "ultralytics-yolo-classification", "--dataset", str(dataset), "--validate-only"]
    )
    assert exit_code == 0
