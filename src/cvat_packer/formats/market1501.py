"""Market-1501 1.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout::

    <root>/
      bounding_box_train/
        0001_c1s1_000000_00.jpg   # <person_id>_c<camera_id>s<seq>_<frame>_<box>.jpg
      bounding_box_test/
      query/

TODO (Phase 3): parse the "<pid>_c<cam>s<seq>_<frame>_<box>" filename schema
and validate person_id / camera_id ranges against Market-1501 conventions.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class Market1501Adapter(SkeletonFormatAdapter):
    format_name = "market1501"
    aliases = ["market-1501", "market1501-1.0"]
    version = "1.0"
    supported_tasks = ["reidentification"]
    supported_shapes: list[str] = []
    supports_attributes = True
    supports_tracks = False
    display_name = "Market-1501 1.0"

    required_any = ["bounding_box_train/*.jpg", "bounding_box_test/*.jpg", "query/*.jpg"]
    structure_hint = "bounding_box_train/, bounding_box_test/, query/ folders with '<pid>_c<cam>s<seq>_...' filenames"
