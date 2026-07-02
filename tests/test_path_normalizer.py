from cvat_packer.utils.path_normalizer import basename, is_safe_relative_path


def test_rejects_absolute_posix_path():
    assert is_safe_relative_path("/etc/passwd") is False


def test_rejects_windows_drive_letter():
    assert is_safe_relative_path("C:/Windows/System32") is False


def test_rejects_parent_traversal():
    assert is_safe_relative_path("../../etc/passwd") is False
    assert is_safe_relative_path("images/../../etc/passwd") is False


def test_accepts_safe_relative_path():
    assert is_safe_relative_path("images/img1.jpg") is True
    assert is_safe_relative_path("annotations/instances_default.json") is True


def test_rejects_empty_path():
    assert is_safe_relative_path("") is False


def test_basename_handles_forward_slashes():
    assert basename("images/img1.jpg") == "img1.jpg"


def test_basename_handles_backslashes_regardless_of_host_os():
    assert basename("images\\img1.jpg") == "img1.jpg"


def test_basename_bare_filename():
    assert basename("img1.jpg") == "img1.jpg"
