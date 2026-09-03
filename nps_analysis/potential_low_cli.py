from __future__ import annotations

import argparse
from pathlib import Path

from .potential_low import RuleConfig, run_potential_low


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Identify potential low-score customer groups with auditable rules."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input Excel workbook"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--as-of", required=True, help="Scoring cutoff date, for example 2026-09-01"
    )
    parser.add_argument("--rule-version", default="potential-low-v3")
    parser.add_argument("--potential-low-min-types", type=int, default=3)
    parser.add_argument("--medium-priority-min-types", type=int, default=2)
    parser.add_argument(
        "--semantic-enabled",
        action="store_true",
        help="Enable local Chinese semantic matching for text evidence",
    )
    parser.add_argument("--semantic-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--semantic-threshold", type=float, default=0.82)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_potential_low(
        args.input,
        args.output,
        args.as_of,
        RuleConfig(
            version=args.rule_version,
            potential_low_min_types=args.potential_low_min_types,
            medium_priority_min_types=args.medium_priority_min_types,
            semantic_enabled=args.semantic_enabled,
            semantic_model=args.semantic_model,
            semantic_threshold=args.semantic_threshold,
        ),
    )
    print(f"Completed. Manifest: {manifest}")


if __name__ == "__main__":
    main()
