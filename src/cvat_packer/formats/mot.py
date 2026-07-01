"""MOT 1.0 format adapter (Phase 3: structural validation only).

Expected CVAT upload layout::

    <root>/
      img1/000001.jpg
      gt/gt.txt          # "frame,id,x,y,w,h,conf,class,visibility" per line
      seqinfo.ini        # [Sequence] name/imDir/frameRate/seqLength/imWidth/imHeight/imExt

TODO (Phase 3): parse seqinfo.ini and gt.txt, cross-check frame numbers
against img1/ frame files, and validate track id continuity.
"""

from __future__ import annotations

from cvat_packer.core.registry import register
from cvat_packer.formats.base import SkeletonFormatAdapter


@register
class MotAdapter(SkeletonFormatAdapter):
    format_name = "mot"
    aliases = ["mot-1.0"]
    version = "1.0"
    supported_tasks = ["tracking"]
    supported_shapes = ["bbox"]
    supports_attributes = False
    supports_tracks = True
    display_name = "MOT 1.0"

    required_any = ["gt/gt.txt", "seqinfo.ini", "img1/*.jpg"]
    structure_hint = "sequence folder with img1/ frames, gt/gt.txt and seqinfo.ini"
