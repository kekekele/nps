from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .config import load_config
from .engine import (
    build_hits,
    evaluate_rules,
    load_labels,
    load_profiles,
    optimize_rules,
    predict,
    rule_impacts,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "phone_id",
        "is_potential_low",
        "primary_type",
        "risk_reasons",
        "hit_rule_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_rows(
    path: Path, rows: list[dict[str, object]], columns: list[str]
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _labeled_prediction_details(
    predictions: list[dict[str, object]], labels: dict[str, int]
) -> list[dict[str, object]]:
    details = []
    for prediction in predictions:
        phone_id = str(prediction["phone_id"])
        if phone_id not in labels:
            continue
        actual = labels[phone_id]
        predicted = int(prediction["is_potential_low"])
        outcome = (
            "TP" if actual and predicted else "FP" if predicted else "FN" if actual else "TN"
        )
        details.append(
            {
                **prediction,
                "is_low_score": actual,
                "prediction_outcome": outcome,
            }
        )
    return details


def _rule_hit_details(
    hits: dict[str, dict[str, bool]],
    predictions: list[dict[str, object]],
    rules: list[object],
    selected: object,
    labels: dict[str, int],
    evaluation_labels: dict[str, int] | None,
) -> list[dict[str, object]]:
    prediction_by_id = {str(row["phone_id"]): row for row in predictions}
    rows = []
    for rule in rules:
        rule_id = getattr(rule, "rule_id")
        for phone_id, user_hits in hits.items():
            if not user_hits.get(rule_id, False):
                continue
            rows.append(
                {
                    "phone_id": phone_id,
                    "rule_id": rule_id,
                    "risk_type": getattr(rule, "risk_type"),
                    "rule_reason": getattr(rule, "reason"),
                    "selected_weight": getattr(selected, "weights").get(rule_id, 0),
                    "contributes_to_decision": int(
                        getattr(selected, "weights").get(rule_id, 0) > 0
                    ),
                    "is_potential_low": prediction_by_id[phone_id]["is_potential_low"],
                    "calibration_is_low_score": labels.get(phone_id, ""),
                    "evaluation_is_low_score": (
                        evaluation_labels.get(phone_id, "") if evaluation_labels else ""
                    ),
                }
            )
    return rows


def _show_rule_progress(index: int, total: int, rule: object) -> None:
    rule_id = getattr(rule, "rule_id")
    print(f"\r      [{index:>2}/{total}] {rule_id}", end="", flush=True)
    if index == total:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize configurable potential low-score rules."
    )
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--evaluation-labels",
        type=Path,
        help="Optional independent CSV used only to calculate final metrics",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=["precision", "recall", "balanced"])
    args = parser.parse_args()

    print("[1/6] Loading rule configuration...")
    windows, optimization, rules = load_config(args.config)
    if args.mode:
        optimization = type(optimization)(
            **{**optimization.__dict__, "selection_mode": args.mode}
        )
    print(
        f"      feature_end={windows['feature_end']}, mode={optimization.selection_mode}, "
        f"enabled_rules={sum(rule.enabled for rule in rules)}"
    )

    print("[2/6] Loading user profiles and optimization labels...")
    profiles = load_profiles(args.profiles)
    labels = load_labels(args.labels)
    print(f"      profiles={len(profiles)}, optimization_labels={len(labels)}")

    print("[3/6] Calculating rule hits...")
    hits = build_hits(
        profiles,
        rules,
        windows["feature_end"],
        progress=_show_rule_progress,
    )
    print(f"      profiles_with_hits={len(hits)}")

    print("[4/6] Optimizing rule weights and decision threshold...")
    selected = optimize_rules(hits, labels, rules, optimization)
    calibration_rule_impacts = rule_impacts(hits, labels, rules, selected)
    print(
        f"      threshold={selected.threshold}, "
        f"calibration_precision={selected.calibration_metrics['precision']:.3f}, "
        f"calibration_recall={selected.calibration_metrics['recall']:.3f}"
    )

    print("[5/6] Generating potential low-score decisions...")
    predictions = predict(hits, rules, selected)
    predicted_count = sum(row["is_potential_low"] for row in predictions)
    print(
        f"      predicted_low={predicted_count}, total_predictions={len(predictions)}"
    )
    evaluation_metrics = None
    evaluation_rule_impacts = []
    evaluation_labels = None
    if args.evaluation_labels:
        print("      Calculating independent evaluation metrics...")
        evaluation_labels = load_labels(args.evaluation_labels)
        evaluation_metrics = evaluate_rules(hits, evaluation_labels, rules, selected)
        evaluation_rule_impacts = rule_impacts(
            hits, evaluation_labels, rules, selected
        )
        print(
            f"      evaluation_users={evaluation_metrics['tp'] + evaluation_metrics['fp'] + evaluation_metrics['fn'] + evaluation_metrics['tn']}, "
            f"precision={evaluation_metrics['precision']:.3f}, "
            f"recall={evaluation_metrics['recall']:.3f}, f1={evaluation_metrics['f1']:.3f}"
        )

    print("[6/6] Writing output files...")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "potential_low_user.csv", predictions)
    calibration_details = _labeled_prediction_details(predictions, labels)
    detail_columns = [
        "phone_id",
        "is_potential_low",
        "primary_type",
        "risk_reasons",
        "hit_rule_count",
        "is_low_score",
        "prediction_outcome",
    ]
    _write_rows(
        args.output / "calibration_prediction_detail.csv",
        calibration_details,
        detail_columns,
    )
    _write_rows(
        args.output / "calibration_recalled_low_users.csv",
        [row for row in calibration_details if row["prediction_outcome"] == "TP"],
        detail_columns,
    )
    if evaluation_labels is not None:
        evaluation_details = _labeled_prediction_details(predictions, evaluation_labels)
        _write_rows(
            args.output / "evaluation_prediction_detail.csv",
            evaluation_details,
            detail_columns,
        )
        _write_rows(
            args.output / "evaluation_recalled_low_users.csv",
            [row for row in evaluation_details if row["prediction_outcome"] == "TP"],
            detail_columns,
        )
    _write_rows(
        args.output / "user_rule_hit_detail.csv",
        _rule_hit_details(
            hits, predictions, rules, selected, labels, evaluation_labels
        ),
        [
            "phone_id",
            "rule_id",
            "risk_type",
            "rule_reason",
            "selected_weight",
            "contributes_to_decision",
            "is_potential_low",
            "calibration_is_low_score",
            "evaluation_is_low_score",
        ],
    )
    (args.output / "rule_optimization.json").write_text(
        json.dumps(
            {
                "weights": selected.weights,
                "threshold": selected.threshold,
                "calibration_metrics": selected.calibration_metrics,
                "evaluation_metrics": evaluation_metrics,
                "calibration_rule_impacts": calibration_rule_impacts,
                "evaluation_rule_impacts": evaluation_rule_impacts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Completed. Output directory: {args.output}")


if __name__ == "__main__":
    main()
