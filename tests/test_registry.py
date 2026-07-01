import pytest

import cvat_packer.formats  # noqa: F401  (registers all format adapters)
from cvat_packer.core.errors import UnknownFormatError
from cvat_packer.core.registry import get_adapter, list_formats


def test_list_formats_includes_core_formats():
    names = {cls.format_name for cls in list_formats()}
    for expected in ["coco", "yolo", "ultralytics-yolo-detection", "pascal-voc", "cvat-image"]:
        assert expected in names


def test_list_formats_includes_phase3_stubs():
    names = {cls.format_name for cls in list_formats()}
    for expected in ["datumaro", "mot", "mots", "labelme", "cityscapes", "kitti", "lfw"]:
        assert expected in names


def test_get_adapter_by_canonical_name():
    adapter = get_adapter("coco")
    assert adapter.format_name == "coco"


def test_get_adapter_by_alias():
    adapter = get_adapter("voc")
    assert adapter.format_name == "pascal-voc"


def test_get_adapter_is_case_insensitive():
    adapter = get_adapter("COCO")
    assert adapter.format_name == "coco"


def test_unknown_format_raises_with_helpful_message():
    with pytest.raises(UnknownFormatError) as exc_info:
        get_adapter("not-a-real-format")
    assert "Unknown format" in str(exc_info.value)
    assert "coco" in str(exc_info.value)


def test_no_duplicate_format_names():
    names = [cls.format_name for cls in list_formats()]
    assert len(names) == len(set(names))
