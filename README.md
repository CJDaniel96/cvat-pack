# CVAT Upload Annotation Maker (`cvat-pack`)

A Python CLI that packs local images + annotation files into a zip archive
matching the structure CVAT expects for **Upload Annotations** / **Import
Dataset**, for every dataset format listed in the
[CVAT dataset formats docs](https://docs.cvat.ai/docs/dataset_management/formats/).

Each format is implemented as an independent, pluggable **adapter**
(`detect` / `validate` / `build_package`) so formats can be completed one at
a time without touching a shared "God file". See
[Architecture](#architecture) and [Adding a new format adapter](#how-to-add-a-new-cvat-format-adapter).

## Status at a glance

| Tier | Formats | What works |
|---|---|---|
| **Full** | COCO 1.0, COCO Keypoints 1.0, YOLO 1.0, Ultralytics YOLO (Detection/Segmentation/Pose/OBB/Classification), Pascal VOC 1.0, CVAT for images 1.1, CVAT for video 1.1 | Full schema/field validation, missing-image & orphan-annotation detection, real zip packaging |
| **Structural (Phase 3 TODO)** | Datumaro, Segmentation Mask, ImageNet, WIDER Face, CamVid, VGGFace2, Market-1501, ICDAR (Recognition/Detection/Segmentation), Open Images, Cityscapes, KITTI, LFW, MOT, MOTS PNG, LabelMe | Folder/file existence checks + best-effort as-is packaging; content-level validation not yet implemented (always emits a warning saying so) |

## Installation

Requires Python 3.10+. Dependency management uses
[`uv`](https://docs.astral.sh/uv/) ([install instructions](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
git clone <this-repo>
cd CVAT-upload-annotation-maker
uv sync --extra dev
```

`uv sync` creates a `.venv`, installs the project in editable mode plus dev
dependencies (`pytest`), and writes/reuses `uv.lock` for reproducible
installs. Run the CLI without activating the venv via `uv run`:

```bash
uv run cvat-pack --list-formats
```

(Or activate it the normal way with `source .venv/bin/activate` and call
`cvat-pack` directly.)

Optional extras:

```bash
uv sync --extra dev --extra images   # adds Pillow, for image-dimension validation
```

### Why these third-party dependencies?

- **PyYAML** (required) — parses Ultralytics `data.yaml` files (class names,
  split paths). It's the standard library for YAML in Python and far lighter
  than pulling in an ML framework just to read a config file.
- **Pillow** (optional, `[images]` extra) — only used, when installed, to
  read actual pixel dimensions for stricter checks (e.g. matching a
  segmentation mask's size to its source image). All core validators fall
  back to filename/existence-only checks when Pillow isn't installed.
- **pytest / pytest-cov** (dev only) — test runner.

No other frameworks are used; all XML/JSON/CSV parsing uses the Python
standard library (`xml.etree.ElementTree`, `json`, `zipfile`, `argparse`).

## CLI usage

```bash
cvat-pack --format <format> [--images DIR] [--annotations DIR|FILE] [--dataset DIR] \
          --output OUT.zip [options]
```

| Flag | Meaning |
|---|---|
| `--format` | Target format name or alias, e.g. `coco`, `yolo`, `ultralytics-yolo-detection`, `pascal-voc`, `cvat-image`. See `--list-formats`. |
| `--images` | Folder (or file, for CVAT-video) containing images |
| `--annotations` | Folder or file containing annotations |
| `--dataset` | A dataset root already close to the target layout (most formats use this) |
| `--output` | Output zip path (required unless `--dry-run`/`--validate-only`) |
| `--labels` | Label map / classes.txt / obj.names override |
| `--dry-run` | Only validate, never write a zip |
| `--validate-only` | Same effect as `--dry-run` (validation-only, no zip) |
| `--copy-images` / `--no-copy-images` | Whether to include images in the zip (default: include) |
| `--strict` | Treat warnings as failures (affects exit code) |
| `--force` | Produce the zip even if validation failed |
| `--manifest` | Write a `<output>.manifest.json` validation report |
| `--verbose` | Debug-level logging |
| `--list-formats` | Print every registered format name + aliases and exit |

**Exit codes:** `0` success, `1` validation failed (or warnings in
`--strict` mode), `2` usage error (unknown format, bad arguments).

### Terminal output

```text
Format: COCO 1.0
Images: 123
Annotations: 456
Warnings: 2
Errors: 0
Output: ./output/cvat_coco.zip
```

### Manifest report (`--manifest`)

Written to `<output-stem>.manifest.json`:

```json
{
  "format": "coco",
  "status": "success",
  "images_count": 123,
  "annotations_count": 456,
  "missing_images": [],
  "orphan_annotations": [],
  "warnings": [],
  "errors": [],
  "output_zip": "output/cvat_coco.zip"
}
```

## Supported formats

Run `cvat-pack --list-formats` for the live list (name + every alias). As of
this writing, 28 formats are registered, matching the CVAT dataset formats
page:

CVAT for images 1.1 · CVAT for video 1.1 · Datumaro 1.0 · COCO 1.0 · COCO
Keypoints 1.0 · PASCAL VOC 1.0 · Segmentation Mask 1.0 · YOLO 1.0 ·
Ultralytics YOLO Detection/Segmentation/Pose/OBB/Classification 1.0 ·
ImageNet 1.0 · WIDER Face 1.0 · CamVid 1.0 · VGGFace2 1.0 · Market-1501 1.0 ·
ICDAR Recognition/Detection/Segmentation 1.0 · Open Images 1.0 · Cityscapes
1.0 · KITTI 1.0 · LFW 1.0 · MOT 1.0 · MOTS PNG 1.0 · LabelMe 3.0

## Input layout per format

### COCO 1.0 / COCO Keypoints 1.0 (`coco`, `coco-keypoints`)

```text
<dataset>/
  images/img1.jpg
  annotations/instances_default.json   # {"images":[...],"annotations":[...],"categories":[...]}
```

Checked: JSON parses; `images`/`annotations`/`categories` present; every
`image_id` in an annotation resolves to a real image entry (else it's
flagged as an **orphan annotation**); every `file_name` exists on disk (else
flagged as a **missing image**); `bbox` is `[x, y, w, h]` with positive
w/h; `segmentation` is a valid polygon list or RLE dict; `keypoints` length
is a multiple of 3.

### YOLO 1.0 (`yolo`)

```text
<dataset>/
  obj.names                # one class name per line
  obj_train_data/
    img1.jpg
    img1.txt                # "class x_center y_center width height", normalized [0,1]
```

### Ultralytics YOLO Detection/Segmentation/Pose/OBB (`ultralytics-yolo-*`)

```text
<dataset>/
  data.yaml                 # {"names": {0: "cat", ...}}
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt
```

Label line field count depends on the task: Detection = 5 fields;
Segmentation = class + (x,y) polygon pairs (variable length); Pose = 5 +
groups of 2/3 per keypoint; OBB = 9 fields (class + 4 corner points).

### Ultralytics YOLO Classification (`ultralytics-yolo-classification`)

```text
<dataset>/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg     # optional
```

### PASCAL VOC 1.0 (`pascal-voc`)

```text
<dataset>/
  JPEGImages/img1.jpg
  Annotations/img1.xml        # <annotation><filename>...<object><name>...<bndbox>
  ImageSets/Main/*.txt        # optional
  SegmentationClass/          # optional
  SegmentationObject/         # optional
```

### CVAT for images 1.1 (`cvat-image`) / CVAT for video 1.1 (`cvat-video`)

```text
<dataset>/
  annotations.xml    # <annotations><image name=...><box|polygon|polyline|points|cuboid|ellipse|mask|skeleton .../></image><track .../>
  images/img1.jpg    # (cvat-image) or a source video file (cvat-video)
```

### Phase-3 (structural-only) formats

Each stub adapter documents its own expected layout in its module docstring
(e.g. `src/cvat_packer/formats/mot.py`, `.../cityscapes.py`, ...) and in the
error message you get from `--validate-only` — the CLI always tells you
which folder/file it expected and didn't find.

## Example commands

```bash
# COCO
uv run cvat-pack --format coco --dataset ./my_coco_dataset --output ./output/cvat_coco.zip --manifest

# YOLO 1.0
uv run cvat-pack --format yolo --dataset ./my_yolo_dataset --output ./output/cvat_yolo.zip

# Ultralytics YOLO Detection
uv run cvat-pack --format ultralytics-yolo-detection --dataset ./dataset --output ./output/yolo_detection.zip

# Pascal VOC, validate only (CI-style check, no zip produced)
uv run cvat-pack --format pascal-voc --dataset ./voc_dataset --validate-only --strict

# CVAT for images, annotations only (no images in the zip)
uv run cvat-pack --format cvat-image --dataset ./cvat_task --output ./output/cvat_images.zip --no-copy-images
```

## Common errors & how to fix them

| Message | Fix |
|---|---|
| `[coco] Could not find a COCO annotation JSON file` | Point `--annotations` at the JSON file, or ensure `--dataset/annotations/*.json` exists |
| `[coco] Annotation N references unknown image_id` | Orphan annotation — fix/remove the annotation or add the missing image entry |
| `[yolo] Could not find class names` | Add `obj.names`/`classes.txt`, or pass `--labels` |
| `[yolo] ... coordinates must be normalized in [0,1]` | YOLO/Ultralytics coordinates must be fractions of image size, not pixels |
| `[pascal-voc] ... missing <bndbox>` | Every `<object>` needs a `<bndbox>` (or a `<polygon>` for segmentation) |
| `[cvat-image] Could not find annotations.xml` | Point `--annotations` at the XML file or its containing folder |
| `Validation did not pass; no zip was produced` | Fix the reported errors, or re-run with `--force` to package anyway |
| `Unknown format 'X'. Available formats: ...` | Run `cvat-pack --list-formats` for valid names/aliases |

## How to add a new CVAT format adapter

1. Create `src/cvat_packer/formats/<your_format>.py`.
2. Subclass `FormatAdapter` (full implementation) or `SkeletonFormatAdapter`
   (quick structural-only stub — set `required_any` glob patterns and
   `structure_hint`) from `cvat_packer.formats.base`.
3. Set the class attributes: `format_name`, `aliases`, `version`,
   `supported_tasks`, `supported_shapes`, `supports_attributes`,
   `supports_tracks`, `display_name`.
4. Implement `detect(input_path)`, `validate(config) -> ValidationReport`,
   `build_package(config, report) -> PackageResult`. Every error/warning
   should say **which file** is wrong and **why** (see existing adapters for
   the `"[format-name] ..."` message convention).
5. Decorate the class with `@register` (from `cvat_packer.core.registry`).
6. Add the module to the import list in `src/cvat_packer/formats/__init__.py`
   so it actually gets registered.
7. Add fixtures under `tests/fixtures/<format>/` and a `tests/test_<format>.py`
   covering: success case, at least one validation-error case, and a
   `build_package` zip test.

## Testing

```bash
uv sync --extra dev
uv run pytest                       # run everything
uv run pytest tests/test_coco.py -v # run one module
uv run pytest --cov=cvat_packer     # with coverage
```

45 tests currently cover: CLI argument handling (`--dry-run`,
`--validate-only`, `--strict`, `--copy-images`, `--manifest`, `--force`,
error exit codes), the format registry, COCO JSON validation (schema,
orphan annotations, missing images, malformed bbox/segmentation), YOLO label
validation (field count, class range, normalized coordinates, orphan
labels), Pascal VOC XML validation, and zip structure (no junk files, POSIX
paths, expected entries).

## Limitations & TODO

- 15 formats (see table above) currently only get **structural**
  validation — folder/file existence checks — not full content-level
  validation. Phase 3 work is to fill these in one at a time (parse each
  format's real annotation schema, cross-check image/label pairing, verify
  mask/image dimensions, etc.), following the pattern established by
  `coco.py` / `yolo.py` / `pascal_voc.py`.
- Image dimension checks (mask size == image size, etc.) require Pillow
  (`pip install -e ".[images]"`); without it, adapters fall back to
  filename/existence-only checks.
- `cvat-video` does not decode the actual video file (frame count, fps) —
  it only checks that a video file exists and that track/shape `frame`
  attributes parse as integers.
- No GUI/TUI; this is a CLI-only tool by design.
- Very large datasets are copied into a temporary staging directory before
  zipping (simplifies "no junk files / safe paths" guarantees); this uses
  extra disk space equal to the packaged dataset size.
