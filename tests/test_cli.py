import json
import shutil
import zipfile
from pathlib import Path

import pytest

from cvat_packer.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "coco"


def _copy_fixture(tmp_path: Path) -> Path:
    dest = tmp_path / "coco"
    shutil.copytree(FIXTURE, dest)
    return dest


def test_cli_list_formats(capsys):
    exit_code = main(["--list-formats"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "coco" in out
    assert "pascal-voc" in out


def test_cli_happy_path(tmp_path, capsys):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    exit_code = main(["--format", "coco", "--dataset", str(dataset), "--output", str(output)])
    assert exit_code == 0
    assert output.exists()
    out = capsys.readouterr().out
    assert "Format: COCO 1.0" in out
    assert "Errors: 0" in out


def test_cli_dry_run_does_not_produce_zip(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    exit_code = main(
        ["--format", "coco", "--dataset", str(dataset), "--output", str(output), "--dry-run"]
    )
    assert exit_code == 0
    assert not output.exists()


def test_cli_validate_only(tmp_path):
    dataset = _copy_fixture(tmp_path)
    exit_code = main(["--format", "coco", "--dataset", str(dataset), "--validate-only"])
    assert exit_code == 0


def test_cli_strict_fails_on_warning(tmp_path):
    dataset = _copy_fixture(tmp_path)
    (dataset / "images" / "img1.jpg").unlink()
    exit_code = main(
        ["--format", "coco", "--dataset", str(dataset), "--validate-only", "--strict"]
    )
    assert exit_code == 1


def test_cli_unknown_format_returns_error_code(tmp_path):
    dataset = _copy_fixture(tmp_path)
    exit_code = main(["--format", "not-real", "--dataset", str(dataset), "--validate-only"])
    assert exit_code == 2


def test_cli_manifest_written(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    exit_code = main(
        ["--format", "coco", "--dataset", str(dataset), "--output", str(output), "--manifest"]
    )
    assert exit_code == 0
    manifest_path = output.with_name(output.stem + ".manifest.json")
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["format"] == "coco"
    assert data["status"] == "success"


def test_cli_no_copy_images(tmp_path):
    dataset = _copy_fixture(tmp_path)
    output = tmp_path / "out.zip"
    exit_code = main(
        [
            "--format", "coco",
            "--dataset", str(dataset),
            "--output", str(output),
            "--no-copy-images",
        ]
    )
    assert exit_code == 0
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert not any(n.startswith("images/") for n in names)


def test_cli_failed_validation_blocks_zip_without_force(tmp_path):
    dataset = _copy_fixture(tmp_path)
    json_path = dataset / "annotations" / "instances_default.json"
    data = json.loads(json_path.read_text())
    data["annotations"][0]["bbox"] = [0, 0, -1, -1]
    json_path.write_text(json.dumps(data))

    output = tmp_path / "out.zip"
    exit_code = main(["--format", "coco", "--dataset", str(dataset), "--output", str(output)])
    assert exit_code == 1
    assert not output.exists()


def test_cli_force_produces_zip_despite_errors(tmp_path):
    dataset = _copy_fixture(tmp_path)
    json_path = dataset / "annotations" / "instances_default.json"
    data = json.loads(json_path.read_text())
    data["annotations"][0]["bbox"] = [0, 0, -1, -1]
    json_path.write_text(json.dumps(data))

    output = tmp_path / "out.zip"
    exit_code = main(
        ["--format", "coco", "--dataset", str(dataset), "--output", str(output), "--force"]
    )
    assert exit_code == 0
    assert output.exists()


def test_cli_requires_input_source(tmp_path):
    output = tmp_path / "out.zip"
    with pytest.raises(SystemExit):
        main(["--format", "coco", "--output", str(output)])


def test_cli_requires_format(tmp_path):
    dataset = _copy_fixture(tmp_path)
    with pytest.raises(SystemExit):
        main(["--dataset", str(dataset), "--validate-only"])


def test_cli_requires_output_unless_dry_run(tmp_path):
    dataset = _copy_fixture(tmp_path)
    with pytest.raises(SystemExit):
        main(["--format", "coco", "--dataset", str(dataset)])
