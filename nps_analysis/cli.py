from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean and analyze the NPS survey workbook."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("doc/指标汇总.xlsx"),
        help="Input Excel workbook",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data"), help="Output data directory"
    )
    parser.add_argument(
        "--batch-id", default="20260901", help="Source processing batch identifier"
    )
    parser.add_argument("--version", default="v1", help="Analysis output version")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_pipeline(args.input, args.output, args.batch_id, args.version)
    print(f"Completed. Manifest: {manifest}")


if __name__ == "__main__":
    main()
