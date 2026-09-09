"""Discover explainable rule candidates from labeled user profiles with EBM.

This script is a candidate-discovery aid. It never modifies rules.yaml and does
not produce business predictions. Review every emitted candidate before adding a
rule to a versioned YAML file.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


WINDOWS = (30, 90, 183)
MAX_CATEGORIES = 20


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _event_age_days(event: dict[str, Any], feature_end: pd.Timestamp) -> int | None:
    event_time = pd.to_datetime(event.get("event_time"), errors="coerce")
    if pd.isna(event_time):
        return None
    age = (feature_end - event_time).days
    return age if age >= 0 else None


def _frequent_categories(profiles: list[dict[str, Any]]) -> dict[str, set[str]]:
    counts = {field: Counter() for field in ("action", "business", "intent")}
    for profile in profiles:
        for event in profile.get("user_journey", []):
            for field, counter in counts.items():
                value = str(event.get(field) or "").strip()
                if value:
                    counter[value] += 1
    return {
        field: {value for value, _ in counter.most_common(MAX_CATEGORIES)}
        for field, counter in counts.items()
    }


def build_feature_frame(
    profiles: list[dict[str, Any]], feature_end: str
) -> pd.DataFrame:
    """Create structured, reviewable aggregates; raw_text is intentionally excluded."""
    as_of = pd.Timestamp(feature_end)
    categories = _frequent_categories(profiles)
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        row: dict[str, Any] = {"phone_id": str(profile["phone_id"])}
        for field, value in profile.get("basic_info", {}).items():
            number = _number(value)
            if number is not None:
                row[f"basic__{field}"] = number

        for metric, values in profile.get("recent_metrics", {}).items():
            numeric = [number for value in values if (number := _number(value)) is not None]
            if numeric:
                row[f"monthly__{metric}__last"] = numeric[-1]
                row[f"monthly__{metric}__sum"] = sum(numeric)
                row[f"monthly__{metric}__min"] = min(numeric)
                row[f"monthly__{metric}__max"] = max(numeric)
                row[f"monthly__{metric}__positive_count"] = sum(value > 0 for value in numeric)

        events = profile.get("user_journey", [])
        for window in WINDOWS:
            window_events = [
                event
                for event in events
                if (age := _event_age_days(event, as_of)) is not None and age <= window
            ]
            row[f"journey__events__{window}d"] = len(window_events)
            for field, values in categories.items():
                for value in values:
                    row[f"journey__{field}={value}__{window}d"] = sum(
                        event.get(field) == value for event in window_events
                    )
            numeric_fields = {
                key
                for event in window_events
                for key, value in event.items()
                if _number(value) is not None and key not in {"event_id"}
            }
            for field in numeric_fields:
                values = [
                    number
                    for event in window_events
                    if (number := _number(event.get(field))) is not None
                ]
                if values:
                    row[f"journey__{field}__{window}d__min"] = min(values)
                    row[f"journey__{field}__{window}d__max"] = max(values)
        rows.append(row)
    return pd.DataFrame(rows).set_index("phone_id")


def load_labels(path: Path) -> pd.Series:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"phone_id": str})
    required = {"phone_id", "is_low_score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Label CSV must contain {sorted(required)}")
    labels = pd.to_numeric(frame["is_low_score"], errors="coerce")
    valid = labels.isin([0, 1]) & frame["phone_id"].notna()
    return pd.Series(labels[valid].astype(int).to_numpy(), index=frame.loc[valid, "phone_id"])


def _term_kind(name: str) -> str:
    if " & " in name:
        return "interaction"
    if name.startswith("journey__"):
        return "journey"
    if name.startswith("monthly__"):
        return "monthly"
    return "basic"


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover EBM rule candidates.")
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--feature-end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interactions", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    args = parser.parse_args()

    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError as error:
        raise SystemExit("Install dependencies first: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt") from error

    profiles = [json.loads(line) for line in args.profiles.read_text(encoding="utf-8").splitlines() if line]
    features = build_feature_frame(profiles, args.feature_end)
    labels = load_labels(args.labels)
    ids = features.index.intersection(labels.index)
    if len(ids) < 50 or labels.loc[ids].nunique() != 2:
        raise ValueError("At least 50 labeled profiles with both label classes are required")

    x = features.loc[ids].replace([float("inf"), float("-inf")], pd.NA)
    y = labels.loc[ids]
    model = ExplainableBoostingClassifier(
        interactions=args.interactions,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
    )
    model.fit(x, y)
    predicted = model.predict(x)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="binary", zero_division=0
    )

    args.output.mkdir(parents=True, exist_ok=True)
    feature_rows = [
        {
            "term": name,
            "term_type": _term_kind(name),
            "importance": float(importance),
            "review_guidance": "将连续变量分段或交互项转换为可解释的 YAML 候选；不得直接上线。",
        }
        for name, importance in sorted(
            zip(model.term_names_, model.term_importances()),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    with (args.output / "ebm_term_importance.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    (args.output / "ebm_discovery_report.json").write_text(
        json.dumps(
            {
                "profiles": len(profiles),
                "labeled_profiles": len(ids),
                "low_score_rate": float(y.mean()),
                "training_precision": float(precision),
                "training_recall": float(recall),
                "training_f1": float(f1),
                "feature_count": int(x.shape[1]),
                "interaction_count": args.interactions,
                "notice": "EBM results are calibration-only candidate evidence. Review and validate YAML rules independently.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    model.to_json(args.output / "ebm_model.json", detail="interpretable", indent=2)
    print(f"Wrote EBM candidate evidence to {args.output}")


if __name__ == "__main__":
    main()