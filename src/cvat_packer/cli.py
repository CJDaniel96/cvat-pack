"""cvat-pack CLI entry point.

See README.md for full usage documentation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvat_packer.formats  # noqa: F401  (import registers every format adapter)
from cvat_packer.core.config import PackConfig
from cvat_packer.core.errors import CvatPackerError, UnknownFormatError
from cvat_packer.core.registry import get_adapter, list_formats
from cvat_packer.core.report import format_summary, write_manifest
from cvat_packer.utils.logging import get_logger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvat-pack",
        description="Pack local images/annotations into a CVAT Upload Annotations / Import Dataset zip.",
    )
    parser.add_argument(
        "--format",
        help="Target CVAT dataset format (e.g. coco, yolo, ultralytics-yolo-detection, "
        "pascal-voc, cvat-image). Use --list-formats to see every supported name/alias.",
    )
    parser.add_argument("--images", type=Path, help="Folder (or file) containing images")
    parser.add_argument("--annotations", type=Path, help="Folder or file containing annotations")
    parser.add_argument("--dataset", type=Path, help="Dataset root already close to the target layout")
    parser.add_argument("--output", type=Path, help="Output zip path")
    parser.add_argument("--labels", type=Path, help="Label map / classes.txt file")

    parser.add_argument("--dry-run", action="store_true", help="Only check inputs, do not produce a zip")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation, do not produce a zip")

    parser.add_argument(
        "--copy-images",
        dest="copy_images",
        action="store_true",
        default=True,
        help="Copy images into the output zip (default)",
    )
    parser.add_argument(
        "--no-copy-images",
        dest="copy_images",
        action="store_false",
        help="Do not copy images into the output zip (annotations only)",
    )

    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--force", action="store_true", help="Produce the zip even if validation failed")
    parser.add_argument("--manifest", action="store_true", help="Write a *.manifest.json validation report")
    parser.add_argument("--verbose", action="store_true", help="Verbose (debug-level) logging")
    parser.add_argument("--list-formats", action="store_true", help="List supported formats and exit")
    return parser


def _manifest_path_for(output: Path | None) -> Path:
    if output is None:
        return Path("manifest.json")
    return output.with_name(output.stem + ".manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logger = get_logger(args.verbose)

    if args.list_formats:
        for adapter_cls in list_formats():
            names = ", ".join([adapter_cls.format_name, *adapter_cls.aliases])
            print(f"{adapter_cls.format_name:32s} v{adapter_cls.version:<6s} ({names})")
        return 0

    if not args.format:
        parser.error("--format is required (use --list-formats to see available options)")

    if not (args.images or args.annotations or args.dataset):
        parser.error("At least one of --images, --annotations or --dataset is required")

    if not args.dry_run and not args.validate_only and not args.output:
        parser.error("--output is required unless --dry-run or --validate-only is set")

    output = args.output if args.output else Path("output.zip")

    config = PackConfig(
        format=args.format,
        output=output,
        images=args.images,
        annotations=args.annotations,
        dataset=args.dataset,
        labels=args.labels,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        copy_images=args.copy_images,
        strict=args.strict,
        force=args.force,
        manifest=args.manifest,
        verbose=args.verbose,
    )

    try:
        adapter = get_adapter(config.format)
    except UnknownFormatError as exc:
        logger.error(str(exc))
        return 2

    display_name = adapter.display_name or adapter.format_name
    logger.info(f"Validating input as {display_name}...")

    try:
        report = adapter.validate(config)
    except CvatPackerError as exc:
        logger.error(str(exc))
        return 2

    for warning in report.warnings:
        logger.warning(warning)
    for error in report.errors:
        logger.error(error)

    if args.manifest:
        manifest_path = write_manifest(report, _manifest_path_for(args.output))
        logger.info(f"Manifest written to {manifest_path}")

    if config.dry_run or config.validate_only:
        print(format_summary(report, display_name))
        return 0 if report.ok(strict=config.strict) else 1

    if not report.ok(strict=config.strict) and not config.force:
        print(format_summary(report, display_name))
        logger.error("Validation did not pass; no zip was produced. Use --force to package anyway.")
        return 1

    try:
        result = adapter.build_package(config, report)
    except CvatPackerError as exc:
        logger.error(str(exc))
        return 2

    if args.manifest:
        write_manifest(result.report, _manifest_path_for(args.output))

    print(format_summary(result.report, display_name))
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
