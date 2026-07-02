from pathlib import Path

from cvat_packer.core.config import PackConfig
from cvat_packer.core.models import PackageResult, Status, ValidationReport


def test_packconfig_coerces_str_paths_to_path():
    config = PackConfig(
        format="coco",
        output="out.zip",
        images="images",
        annotations="annotations",
        dataset="dataset",
        labels="labels.txt",
    )
    assert isinstance(config.output, Path)
    assert isinstance(config.images, Path)
    assert isinstance(config.annotations, Path)
    assert isinstance(config.dataset, Path)
    assert isinstance(config.labels, Path)


def test_packconfig_defaults():
    config = PackConfig(format="coco", output="out.zip")
    assert config.images is None
    assert config.annotations is None
    assert config.dataset is None
    assert config.labels is None
    assert config.dry_run is False
    assert config.validate_only is False
    assert config.copy_images is True
    assert config.strict is False
    assert config.force is False
    assert config.manifest is False
    assert config.verbose is False


def test_validation_report_starts_success_and_escalates_on_warning():
    report = ValidationReport(format="coco")
    assert report.status == Status.SUCCESS
    report.add_warning("careful")
    assert report.status == Status.WARNING
    assert report.ok() is True
    assert report.ok(strict=True) is False


def test_validation_report_error_escalates_and_never_downgrades():
    report = ValidationReport(format="coco")
    report.add_warning("careful")
    report.add_error("boom")
    assert report.status == Status.FAILED
    # a later warning must not downgrade status back to WARNING
    report.add_warning("careful again")
    assert report.status == Status.FAILED
    assert report.ok() is False
    assert report.ok(strict=True) is False


def test_validation_report_to_dict_serialization():
    report = ValidationReport(format="coco", images_count=2, annotations_count=3)
    report.missing_images.append("img1.jpg")
    report.orphan_annotations.append("42")
    report.add_warning("a warning")
    report.add_error("an error")
    report.output_zip = "out.zip"

    data = report.to_dict()
    assert data == {
        "format": "coco",
        "status": "failed",
        "images_count": 2,
        "annotations_count": 3,
        "missing_images": ["img1.jpg"],
        "orphan_annotations": ["42"],
        "warnings": ["a warning"],
        "errors": ["an error"],
        "output_zip": "out.zip",
    }


def test_package_result_defaults():
    report = ValidationReport(format="coco")
    result = PackageResult(success=True, report=report)
    assert result.staging_dir is None
    assert result.output_zip is None
    assert result.files_written == []
