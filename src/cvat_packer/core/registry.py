"""Global registry mapping format names/aliases to FormatAdapter classes.

Format modules register themselves with the `@register` decorator at import
time (see cvat_packer/formats/__init__.py, which imports every format module
so this registry is fully populated as soon as it is imported).
"""

from __future__ import annotations

from typing import TypeVar

from cvat_packer.core.errors import UnknownFormatError
from cvat_packer.formats.base import FormatAdapter

_REGISTRY: dict[str, type[FormatAdapter]] = {}
_ALIAS_TO_NAME: dict[str, str] = {}

T = TypeVar("T", bound=type[FormatAdapter])


def register(adapter_cls: T) -> T:
    name = adapter_cls.format_name
    if not name:
        raise ValueError(f"{adapter_cls.__name__} must define a non-empty format_name")
    if name in _REGISTRY:
        raise ValueError(f"Format '{name}' is already registered by {_REGISTRY[name].__name__}")

    _REGISTRY[name] = adapter_cls
    _ALIAS_TO_NAME[name.lower()] = name
    for alias in adapter_cls.aliases:
        _ALIAS_TO_NAME[alias.lower()] = name
    return adapter_cls


def get_adapter(name: str) -> FormatAdapter:
    key = name.strip().lower()
    canonical = _ALIAS_TO_NAME.get(key)
    if canonical is None:
        available = ", ".join(sorted(_REGISTRY))
        raise UnknownFormatError(f"Unknown format '{name}'. Available formats: {available}")
    return _REGISTRY[canonical]()


def list_formats() -> list[type[FormatAdapter]]:
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]
