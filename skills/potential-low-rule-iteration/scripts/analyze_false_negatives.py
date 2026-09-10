"""Compare calibration false negatives with true negatives for rule discovery.

This script only produces candidate evidence. It never changes rules or
predictions and excludes raw_text from all features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from discover_ebm_candidates import build_feature_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find structured feature differences between calibration FN and TN users."
    )
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--prediction-detail", type=Path, required=True)
    parser.add_argument("--feature-end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-non-null", type=int, default=20)
    args = parser.parse_args()

    profiles = [
        json.loads(line)
        for line in args.profiles.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    outcomes = pd.read_csv(
        args.prediction_detail, encoding="utf-8-sig", dtype={"phone_id": str}
    ).set_index("phone_id")
    required = {"prediction_outcome", "is_low_score"}
    if not required.issubset(outcomes.columns):
        raise ValueError(f"Prediction detail must contain {sorted(required)}")
    fn_ids = outcomes.index[outcomes["prediction_outcome"].eq("FN")]
    tn_ids = outcomes.index[outcomes["prediction_outcome"].eq("TN")]
    if len(fn_ids) == 0 or len(tn_ids) == 0:
        raise ValueError("Prediction detail must contain both FN and TN users")

    features = build_feature_frame(profiles, args.feature_end)
    fn = features.loc[features.index.intersection(fn_ids)]
    tn = features.loc[features.index.intersection(tn_ids)]
    if len(fn) == 0 or len(tn) == 0:
        raise ValueError("No FN/TN users overlap the profiles")

    rows = []
    for feature in features.columns:
        fn_values = pd.to_numeric(fn[feature], errors="coerce").dropna()
        tn_values = pd.to_numeric(tn[feature], errors="coerce").dropna()
        if min(len(fn_values), len(tn_values)) < args.min_non_null:
            continue
        fn_nonzero_rate = float((fn_values != 0).mean())
        tn_nonzero_rate = float((tn_values != 0).mean())
        rows.append(
            {
                "feature": feature,
                "fn_non_null": len(fn_values),
                "tn_non_null": len(tn_values),
                "fn_mean": float(fn_values.mean()),
                "tn_mean": float(tn_values.mean()),
                "mean_difference": float(fn_values.mean() - tn_values.mean()),
                "fn_nonzero_rate": fn_nonzero_rate,
                "tn_nonzero_rate": tn_nonzero_rate,
                "nonzero_rate_difference": fn_nonzero_rate - tn_nonzero_rate,
                "review_guidance": "仅作为候选信号；确认业务语义、分段、TP/FP 和 Lift 后才能写入 YAML。",
            }
        )
    contrast = pd.DataFrame(rows)
    if contrast.empty:
        raise ValueError("No features satisfy the minimum non-null sample requirement")
    contrast["absolute_signal_difference"] = contrast[
        "nonzero_rate_difference"
    ].abs()
    contrast = contrast.sort_values("absolute_signal_difference", ascending=False)

    args.output.mkdir(parents=True, exist_ok=True)
    contrast.to_csv(
        args.output / "fn_tn_feature_contrast.csv", index=False, encoding="utf-8-sig"
    )
    (args.output / "fn_analysis_report.json").write_text(
        json.dumps(
            {
                "profile_count": len(profiles),
                "fn_count": len(fn),
                "tn_count": len(tn),
                "fn_rate_among_actual_low": float(
                    len(fn_ids) / (len(fn_ids) + int((outcomes["prediction_outcome"] == "TP").sum()) or 1)
                ),
                "feature_count_reviewed": len(contrast),
                "notice": "This report compares calibration FN with TN only. Validate candidates against all calibration labels before changing YAML.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote FN candidate evidence to {args.output}")


if __name__ == "__main__":
    main()