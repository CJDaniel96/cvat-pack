"""ImageNet 1.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout (torchvision ImageFolder-style)::

    <root>/
      <class_name_1>/
        img1.jpg
      <class_name_2>/
        img2.jpg

TODO (Phase 3): validate that every class subfolder contains at least one
image, and cross-check class names against an optional synsets.txt / labels
file if provided via --labels.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class ImageNetAdapter(SkeletonFormatAdapter):
    format_name = "imagenet"
    aliases = ["imagenet-1.0"]
    version = "1.0"
    supported_tasks = ["classification"]
    supported_shapes: list[str] = []
    supports_attributes = False
    supports_tracks = False
    display_name = "ImageNet 1.0"

    required_any = ["*/*.jpg", "*/*.jpeg", "*/*.png"]
    structure_hint = "one subfolder per class label containing images, e.g. <class_name>/*.jpg"
