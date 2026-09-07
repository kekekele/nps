import csv
import json
import sys

from nps_analysis.protential_low_score import cli
from nps_analysis.protential_low_score.engine import (
    OptimizationConfig,
    Rule,
    build_hits,
    evaluate_rules,
    optimize_rules,
    predict,
    rule_impacts,
)


def _profile(phone_id, events=(), overage=()):
    return {
        "phone_id": phone_id,
        "basic_info": {},
        "recent_metrics": {"over_gprs_fee": list(overage)},
        "user_journey": list(events),
    }


def test_rule_optimization_predicts_without_exposing_scores():
    complaint = {
        "event_time": "2026-08-10",
        "action": "投诉",
        "business": "套餐资费",
        "intent": "费用争议",
    }
    profiles = (
        [_profile(f"low-{index}", [complaint, complaint]) for index in range(5)]
        + [_profile(f"high-{index}") for index in range(5)]
        + [_profile("unmeasured", [complaint, complaint])]
    )
    rule = Rule(
        rule_id="COMPLAINT_REPEAT",
        risk_type="投诉未解决疑似型",
        condition={
            "kind": "journey_count",
            "within_days": 30,
            "actions": ["投诉"],
            "min_count": 2,
        },
        weight_candidates=(1, 2),
        reason="重复投诉",
    )
    hits = build_hits(profiles, [rule], "2026-08-14")
    labels = {
        **{f"low-{index}": 1 for index in range(5)},
        **{f"high-{index}": 0 for index in range(5)},
    }
    selected = optimize_rules(
        hits,
        labels,
        [rule],
        OptimizationConfig(threshold_candidates=(1, 2)),
    )
    evaluation = evaluate_rules(hits, labels, [rule], selected)

    results = {row["phone_id"]: row for row in predict(hits, [rule], selected)}

    assert evaluation["precision"] == 1.0
    assert evaluation["recall"] == 1.0
    assert results["low-0"]["is_potential_low"] == 1
    assert results["unmeasured"]["is_potential_low"] == 1
    assert "risk_score" not in results["low-0"]
    impact = rule_impacts(hits, labels, [rule], selected)[0]
    assert impact["rule_id"] == "COMPLAINT_REPEAT"
    assert impact["alone_precision"] == 1.0


def test_configurable_composite_and_sequence_conditions():
    profiles = [
        _profile(
            "u1",
            [
                {"event_time": "2026-08-01", "action": "咨询", "intent": "费用问题"},
                {"event_time": "2026-08-05", "action": "退订", "intent": "退订"},
            ],
            overage=[0, 25],
        ),
        _profile(
            "u2", [{"event_time": "2026-08-01", "action": "投诉"}], overage=[0, 25]
        ),
    ]
    rules = [
        Rule(
            "consult_then_cancel",
            "办理变更退订受阻疑似型",
            {
                "kind": "journey_sequence",
                "within_days": 30,
                "first": {"actions": ["咨询"]},
                "second": {"actions": ["退订"]},
            },
            (1,),
            "咨询后退订",
        ),
        Rule(
            "overage_without_complaint",
            "提醒告知不足疑似型",
            {
                "kind": "all_of",
                "conditions": [
                    {
                        "kind": "any_of",
                        "conditions": [
                            {
                                "kind": "monthly_sum_gt",
                                "metric": "over_gprs_fee",
                                "recent_months": 2,
                                "value": 20,
                            }
                        ],
                    },
                    {
                        "kind": "none_of",
                        "conditions": [
                            {
                                "kind": "journey_count",
                                "within_days": 30,
                                "actions": ["投诉"],
                                "min_count": 1,
                            }
                        ],
                    },
                ],
            },
            (1,),
            "超套且无投诉",
        ),
    ]

    hits = build_hits(profiles, rules, "2026-08-14")

    assert hits["u1"] == {
        "consult_then_cancel": True,
        "overage_without_complaint": True,
    }
    assert hits["u2"] == {
        "consult_then_cancel": False,
        "overage_without_complaint": False,
    }


def test_build_hits_reports_progress_per_enabled_rule():
    rules = [
        Rule(
            "enabled",
            "测试",
            {
                "kind": "journey_count",
                "within_days": 1,
                "actions": ["投诉"],
                "min_count": 1,
            },
            (1,),
            "启用",
        ),
        Rule(
            "disabled",
            "测试",
            {
                "kind": "journey_count",
                "within_days": 1,
                "actions": ["投诉"],
                "min_count": 1,
            },
            (1,),
            "禁用",
            enabled=False,
        ),
    ]
    progress = []

    build_hits(
        [_profile("u1")],
        rules,
        "2026-08-14",
        progress=lambda index, total, rule: progress.append(
            (index, total, rule.rule_id)
        ),
    )

    assert progress == [(1, 1, "enabled")]


def test_cli_writes_predictions_and_optimization_report(tmp_path, monkeypatch):
    profiles_path = tmp_path / "profiles.jsonl"
    labels_path = tmp_path / "labels.csv"
    evaluation_path = tmp_path / "evaluation.csv"
    output_path = tmp_path / "output"
    profiles = [
        _profile(f"low-{index}", [{"event_time": "2026-08-10", "action": "投诉"}] * 2)
        for index in range(5)
    ] + [_profile(f"high-{index}") for index in range(5)]
    profiles_path.write_text(
        "\n".join(json.dumps(profile) for profile in profiles), encoding="utf-8"
    )
    with labels_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["phone_id", "is_low_score"])
        writer.writeheader()
        writer.writerows(
            [{"phone_id": f"low-{index}", "is_low_score": 1} for index in range(5)]
            + [{"phone_id": f"high-{index}", "is_low_score": 0} for index in range(5)]
        )
    with evaluation_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["phone_id", "is_low_score"])
        writer.writeheader()
        writer.writerows(
            [
                {"phone_id": "low-0", "is_low_score": 1},
                {"phone_id": "high-0", "is_low_score": 0},
            ]
        )
    config_path = tmp_path / "rules.yaml"
    config_path.write_text(
        "windows:\n  feature_end: '2026-08-14'\n"
        "optimization:\n  threshold_candidates: [1]\n"
        "rules:\n  - rule_id: complaint\n    risk_type: 投诉\n"
        "    condition: {kind: journey_count, within_days: 30, actions: [投诉], min_count: 2}\n"
        "    weight_candidates: [1]\n    reason: 重复投诉\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rule-score",
            "--profiles",
            str(profiles_path),
            "--labels",
            str(labels_path),
            "--evaluation-labels",
            str(evaluation_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
        ],
    )

    cli.main()

    assert (output_path / "potential_low_user.csv").exists()
    assert (output_path / "rule_optimization.json").exists()
    assert (output_path / "calibration_recalled_low_users.csv").exists()
    assert (output_path / "evaluation_recalled_low_users.csv").exists()
    assert (output_path / "calibration_prediction_detail.csv").exists()
    assert (output_path / "evaluation_prediction_detail.csv").exists()
    assert (output_path / "user_rule_hit_detail.csv").exists()
    with (output_path / "evaluation_recalled_low_users.csv").open(
        encoding="utf-8-sig", newline=""
    ) as output:
        recalled = list(csv.DictReader(output))
    assert recalled[0]["phone_id"] == "low-0"
    assert recalled[0]["prediction_outcome"] == "TP"
    report = json.loads(
        (output_path / "rule_optimization.json").read_text(encoding="utf-8")
    )
    assert report["evaluation_metrics"]["tp"] == 1
    assert report["evaluation_metrics"]["tn"] == 1
    assert report["calibration_rule_impacts"][0]["rule_id"] == "complaint"
