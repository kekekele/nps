import pandas as pd

from nps_analysis.potential_low import (
    EVIDENCE_COLUMNS,
    RuleConfig,
    build_validation,
    extract_monthly_evidence,
    extract_text_evidence,
    score_at,
)


def test_text_rules_allow_multiple_types_and_require_overage_for_reminder():
    complaints = pd.DataFrame(
        [
            {
                "手机号码": "masked-user-1",
                "时间": "20260801",
                "投诉节点": "资费套餐",
                "明细": "多次反映仍未解决，套餐贵，超套扣费未提醒",
            }
        ]
    )
    evidence, events = extract_text_evidence("投诉明细", complaints)
    users, user_type, selected = score_at(
        pd.DataFrame(evidence, columns=EVIDENCE_COLUMNS),
        events,
        ["masked-user-1"],
        "2026-09-01",
        RuleConfig(),
    )

    hits = set(user_type.loc[user_type["is_hit"].eq(1), "risk_type"])
    assert users.loc[0, "is_potential_low"] == 1
    assert "投诉高风险型" in hits
    assert "投诉未解决疑似型" in hits
    assert "套餐资费不满疑似型" in hits
    assert "提醒告知不足疑似型" not in hits
    assert not selected["rule_code"].eq("TEXT_REMINDER_MISSING").any()


def test_monthly_structured_evidence_enables_reminder_rule():
    columns = [
        "月份", "手机号码标识", "手机号码", "客户类型", "宽带号码", "是否有宽带",
        "是否我号他宽", "卡槽情况", "是否携入", "是否查询携转码", "入网时间", "ARPU 值",
        "语音使用量", "流量使用量", "语言超套费用", "流量超套费用", "关停类型",
        "宽带资费费用", "套内流量资源", "流量使用量", "套内语音资源", "语音使用量",
        "是否驻留省外",
    ]
    monthly = pd.DataFrame(
        [["202608", None, "masked-user-1", None, None, None, None, None, None, None,
          None, None, None, None, None, 12.5, None, None, 100, 120, None, None, None]],
        columns=columns,
    )
    complaints = pd.DataFrame(
        [{"手机号码": "masked-user-1", "时间": "20260810", "投诉节点": "提醒", "明细": "超套收费未提醒"}]
    )
    text_evidence, events = extract_text_evidence("投诉明细", complaints)
    evidence = extract_monthly_evidence(monthly) + text_evidence

    _, user_type, selected = score_at(
        pd.DataFrame(evidence, columns=EVIDENCE_COLUMNS),
        events,
        ["masked-user-1"],
        "2026-09-01",
        RuleConfig(),
    )

    hits = set(user_type.loc[user_type["is_hit"].eq(1), "risk_type"])
    assert "流量超套风险型" in hits
    assert "提醒告知不足疑似型" in hits
    assert selected["rule_code"].eq("DATA_USAGE_OVER_RESOURCE").any()


def test_text_rules_do_not_treat_resolved_language_as_unresolved():
    complaints = pd.DataFrame(
        [{"手机号码": "masked-user-1", "时间": "20260801", "投诉节点": "宽带", "明细": "宽带故障已解决"}]
    )
    evidence, _ = extract_text_evidence("投诉明细", complaints)
    codes = {item["rule_code"] for item in evidence}

    assert "COMPLAINT_EVENT" in codes
    assert "TEXT_UNRESOLVED" not in codes
    assert "TEXT_BROADBAND_PROBLEM" not in codes


def test_validation_outputs_survey_matches_and_confusion_metrics():
    surveys = pd.DataFrame(
        [
            {"survey_id": "s1", "customer_key": "u1", "survey_month": pd.Timestamp("2026-08-01"), "segment_label": "low"},
            {"survey_id": "s2", "customer_key": "u2", "survey_month": pd.Timestamp("2026-08-01"), "segment_label": "non_low"},
        ]
    )
    evidence = pd.DataFrame(
        [{
            "customer_key": "u1", "risk_type": "投诉高风险型", "rule_code": "COMPLAINT_EVENT",
            "evidence_level": "strong", "points": 3, "source_sheet": "投诉明细",
            "source_column": "投诉节点", "event_time": pd.Timestamp("2026-07-01"),
            "evidence_value": "资费", "evidence_excerpt": None, "source_row_no": 2,
        }],
        columns=EVIDENCE_COLUMNS,
    )
    events = pd.DataFrame(
        columns=["customer_key", "event_time", "event_kind", "event_group", "source_row_no", "negative"]
    )

    metrics, detail = build_validation(surveys, evidence, events, RuleConfig())
    total = metrics.loc[(metrics["survey_month"] == "ALL") & (metrics["risk_type"] == "ALL")].iloc[0]

    assert set(detail["match_result"]) == {"TP", "TN"}
    assert (total["tp"], total["fp"], total["fn"], total["tn"]) == (1, 0, 0, 1)
    assert total["f1"] == 1.0
    assert total["jaccard"] == 1.0