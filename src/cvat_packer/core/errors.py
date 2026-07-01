"""Custom exceptions for cvat_packer."""

from __future__ import annotations


class CvatPackerError(Exception):
    """Base class for all cvat_packer errors."""


class UnknownFormatError(CvatPackerError):
    """Raised when --format does not match any registered adapter."""


class InputNotFoundError(CvatPackerError):
    """Raised when none of --images/--annotations/--dataset resolve to a usable path."""


class ValidationFailedError(CvatPackerError):
    """Raised when build_package() is called on data that cannot be packaged."""
