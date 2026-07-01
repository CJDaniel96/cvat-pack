"""ICDAR Recognition / Detection / Segmentation 1.0 format adapters
(Phase 3: structural validation only).

Recognition::

    <root>/
      images/word_1.png
      gt.txt                       # "word_1.png, \"text\"" per line

Detection::

    <root>/
      images/img_1.jpg
      gt_img_1.txt                 # "x1,y1,x2,y2,x3,y3,x4,y4,text" per line

Segmentation::

    <root>/
      images/img_1.jpg
      img_1_GT.txt / img_1_GT.bmp  # per-character/word colour-coded GT

TODO (Phase 3): parse each GT file variant and cross-check referenced images.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class IcdarRecognitionAdapter(SkeletonFormatAdapter):
    format_name = "icdar-recognition"
    aliases = ["icdar_recognition", "icdar-recognition-1.0"]
    version = "1.0"
    supported_tasks = ["text_recognition"]
    supported_shapes: list[str] = []
    supports_attributes = False
    supports_tracks = False
    display_name = "ICDAR Recognition 1.0"

    required_any = ["gt.txt", "word.txt"]
    structure_hint = "images/ folder plus gt.txt with '<file>, \"text\"' lines"


@register
class IcdarDetectionAdapter(SkeletonFormatAdapter):
    format_name = "icdar-detection"
    aliases = ["icdar_detection", "icdar-detection-1.0"]
    version = "1.0"
    supported_tasks = ["detection"]
    supported_shapes = ["polygon"]
    supports_attributes = False
    supports_tracks = False
    display_name = "ICDAR Detection 1.0"

    required_any = ["gt_*.txt", "*/gt_*.txt"]
    structure_hint = "images/ folder plus per-image gt_<name>.txt files with polygon + text lines"


@register
class IcdarSegmentationAdapter(SkeletonFormatAdapter):
    format_name = "icdar-segmentation"
    aliases = ["icdar_segmentation", "icdar-segmentation-1.0"]
    version = "1.0"
    supported_tasks = ["segmentation"]
    supported_shapes = ["mask"]
    supports_attributes = False
    supports_tracks = False
    display_name = "ICDAR Segmentation 1.0"

    required_any = ["*_GT.txt", "*_GT.bmp"]
    structure_hint = "images/ folder plus per-image <name>_GT.txt/.bmp character-segmentation ground truth"
