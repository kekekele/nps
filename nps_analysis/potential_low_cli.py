from __future__ import annotations

import argparse
from pathlib import Path

from .potential_low import RuleConfig, run_potential_low


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Identify potential low-score customer groups with auditable rules.")
    parser.add_argument("--input", type=Path, required=True, help="Input Excel workbook")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--as-of", required=True, help="Scoring cutoff date, for example 2026-09-01")
    parser.add_argument("--rule-version", default="potential-low-v1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_potential_low(
        args.input,
        args.output,
        args.as_of,
        RuleConfig(version=args.rule_version),
    )
    print(f"Completed. Manifest: {manifest}")


if __name__ == "__main__":
    main()