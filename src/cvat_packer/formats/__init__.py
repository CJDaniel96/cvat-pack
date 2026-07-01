"""Importing this package registers every built-in format adapter.

Add new format modules to this list so `cvat_packer.core.registry` picks
them up. See README.md "How to add a new CVAT format adapter" for the
step-by-step guide.
"""

from cvat_packer.formats import (
    camvid,
    cityscapes,
    coco,
    coco_keypoints,
    cvat_image,
    cvat_video,
    datumaro,
    icdar,
    imagenet,
    kitti,
    labelme,
    lfw,
    market1501,
    mot,
    mots,
    open_images,
    pascal_voc,
    segmentation_mask,
    ultralytics_yolo,
    vggface2,
    wider_face,
    yolo,
)

__all__ = [
    "camvid",
    "cityscapes",
    "coco",
    "coco_keypoints",
    "cvat_image",
    "cvat_video",
    "datumaro",
    "icdar",
    "imagenet",
    "kitti",
    "labelme",
    "lfw",
    "market1501",
    "mot",
    "mots",
    "open_images",
    "pascal_voc",
    "segmentation_mask",
    "ultralytics_yolo",
    "vggface2",
    "wider_face",
    "yolo",
]
