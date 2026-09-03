import pandas as pd
import pytest

from nps_analysis.potential_low import (
    EVIDENCE_COLUMNS,
    RuleConfig,
    SemanticMatcher,
    build_validation,
    extract_household_candidates,
    extract_monthly_evidence,
    extract_text_evidence,
    score_at,
)


class _FakeSemanticEncoder:
    def encode(self, texts, normalize_embeddings=True):
        return [
            [1.0, 0.0]
            if "套餐" in text or "资费" in text or "收费" in text
            else [0.0, 1.0]
            for text in texts
        ]


def test_semantic_matching_adds_auditable_medium_evidence():
    complaints = pd.DataFrame(
        [
            {
                "手机号码": "masked-user-1",
                "时间": "20260801",
                "投诉节点": "资费套餐",
                "明细": "客户觉得套餐费用太高，收费不合理",
            }
        ]
    )
    matcher = SemanticMatcher(
        "unused-in-test", 0.8, encoder=_FakeSemanticEncoder()
    )

    evidence, _ = extract_text_evidence("投诉明细", complaints, matcher)

    semantic = [item for item in evidence if item["rule_code"].startswith("SEMANTIC_")]
    assert len(semantic) == 1
    assert semantic[0]["risk_type"] == "套餐资费不满疑似型"
    assert semantic[0]["evidence_level"] == "medium"
    assert "similarity=1.000" in semantic[0]["evidence_value"]


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
        RuleConfig(potential_low_min_types=1, medium_priority_min_types=1),
    )

    hits = set(user_type.loc[user_type["is_hit"].eq(1), "risk_type"])
    assert users.loc[0, "is_potential_low"] == 1
    assert "投诉未解决疑似型" in hits
    assert "套餐资费不满疑似型" in hits
    assert "提醒告知不足疑似型" not in hits
    assert not selected["rule_code"].eq("TEXT_REMINDER_MISSING").any()


def test_monthly_structured_evidence_enables_reminder_rule():
    columns = [
        "月份",
        "手机号码标识",
        "手机号码",
        "客户类型",
        "宽带号码",
        "是否有宽带",
        "是否我号他宽",
        "卡槽情况",
        "是否携入",
        "是否查询携转码",
        "入网时间",
        "ARPU 值",
        "语音使用量",
        "流量使用量",
        "语言超套费用",
        "流量超套费用",
        "关停类型",
        "宽带资费费用",
        "套内流量资源",
        "流量使用量",
        "套内语音资源",
        "语音使用量",
        "是否驻留省外",
    ]
    monthly = pd.DataFrame(
        [
            [
                "202608",
                None,
                "masked-user-1",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                25,
                None,
                None,
                100,
                120,
                None,
                None,
                None,
            ]
        ],
        columns=columns,
    )
    complaints = pd.DataFrame(
        [
            {
                "手机号码": "masked-user-1",
                "时间": "20260810",
                "投诉节点": "提醒",
                "明细": "超套收费未提醒",
            }
        ]
    )
    text_evidence, events = extract_text_evidence("投诉明细", complaints)
    evidence = extract_monthly_evidence(monthly) + text_evidence

    _, user_type, selected = score_at(
        pd.DataFrame(evidence, columns=EVIDENCE_COLUMNS),
        events,
        ["masked-user-1"],
        "2026-09-01",
        RuleConfig(potential_low_min_types=1, medium_priority_min_types=1),
    )

    hits = set(user_type.loc[user_type["is_hit"].eq(1), "risk_type"])
    assert "流量超套风险型" in hits
    assert "提醒告知不足疑似型" in hits
    assert selected["rule_code"].eq("DATA_OVERAGE_TOTAL_FEE_GT_20").any()


def test_text_rules_do_not_treat_resolved_language_as_unresolved():
    complaints = pd.DataFrame(
        [
            {
                "手机号码": "masked-user-1",
                "时间": "20260801",
                "投诉节点": "宽带",
                "明细": "宽带故障已解决",
            }
        ]
    )
    evidence, _ = extract_text_evidence("投诉明细", complaints)
    codes = {item["rule_code"] for item in evidence}

    assert "COMPLAINT_EVENT" in codes
    assert "TEXT_UNRESOLVED" not in codes
    assert "TEXT_BROADBAND_PROBLEM" not in codes


def test_validation_outputs_survey_matches_and_confusion_metrics():
    surveys = pd.DataFrame(
        [
            {
                "survey_id": "s1",
                "customer_key": "u1",
                "survey_month": pd.Timestamp("2026-08-01"),
                "segment_label": "low",
            },
            {
                "survey_id": "s2",
                "customer_key": "u2",
                "survey_month": pd.Timestamp("2026-08-01"),
                "segment_label": "non_low",
            },
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "customer_key": "u1",
                "risk_type": "投诉高风险型",
                "rule_code": "COMPLAINT_EVENT",
                "evidence_level": "strong",
                "points": 3,
                "source_sheet": "投诉明细",
                "source_column": "投诉节点",
                "event_time": pd.Timestamp("2026-07-01"),
                "evidence_value": "资费",
                "evidence_excerpt": None,
                "source_row_no": 2,
            }
        ],
        columns=EVIDENCE_COLUMNS,
    )
    events = pd.DataFrame(
        columns=[
            "customer_key",
            "event_time",
            "event_kind",
            "event_group",
            "source_row_no",
            "negative",
        ]
    )

    metrics, detail = build_validation(
        surveys,
        evidence,
        events,
        RuleConfig(potential_low_min_types=1, medium_priority_min_types=1),
    )
    total = metrics.loc[
        (metrics["survey_month"] == "ALL") & (metrics["risk_type"] == "ALL")
    ].iloc[0]

    assert set(detail["match_result"]) == {"TP", "TN"}
    assert (total["tp"], total["fp"], total["fn"], total["tn"]) == (1, 0, 0, 1)
    assert total["f1"] == 1.0
    assert total["jaccard"] == 1.0


def test_repeated_rows_do_not_inflate_rule_score_and_priority_uses_type_count():
    evidence = pd.DataFrame(
        [
            {
                "customer_key": "u1",
                "risk_type": "流量超套风险型",
                "rule_code": "DATA_OVERAGE_FEE",
                "evidence_level": "strong",
                "points": 3,
                "source_sheet": "近半年指标",
                "source_column": "流量超套费用",
                "event_time": pd.Timestamp(f"2026-0{month}-01"),
                "evidence_value": 10,
                "evidence_excerpt": None,
                "source_row_no": month,
            }
            for month in (5, 6, 7)
        ],
        columns=EVIDENCE_COLUMNS,
    )
    events = pd.DataFrame(
        columns=[
            "customer_key",
            "event_time",
            "event_kind",
            "event_group",
            "source_row_no",
            "negative",
        ]
    )

    users, user_type, _ = score_at(evidence, events, ["u1"], "2026-08-01", RuleConfig())
    traffic = user_type.loc[user_type["risk_type"].eq("流量超套风险型")].iloc[0]

    assert traffic["type_score"] == 3
    assert traffic["evidence_count"] == 1
    assert users.loc[0, "is_potential_low"] == 0
    assert users.loc[0, "priority_level"] == "watch"


def test_priority_thresholds_must_be_ordered():
    with pytest.raises(ValueError, match="medium_priority_min_types"):
        RuleConfig(potential_low_min_types=2, medium_priority_min_types=3)


def test_overage_requires_two_months_or_fee_strictly_above_twenty():
    columns = [
        "月份",
        "手机号码标识",
        "手机号码",
        "客户类型",
        "宽带号码",
        "是否有宽带",
        "是否我号他宽",
        "卡槽情况",
        "是否携入",
        "是否查询携转码",
        "入网时间",
        "ARPU 值",
        "语音使用量",
        "流量使用量",
        "语言超套费用",
        "流量超套费用",
        "关停类型",
        "宽带资费费用",
        "套内流量资源",
        "流量使用量",
        "套内语音资源",
        "语音使用量",
        "是否驻留省外",
    ]

    def monthly_row(month, customer, fee):
        return [
            month,
            None,
            customer,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            fee,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

    monthly = pd.DataFrame(
        [
            monthly_row("202606", "two-months", 1),
            monthly_row("202607", "two-months", 2),
            monthly_row("202607", "twenty", 20),
            monthly_row("202607", "above-twenty", 20.01),
        ],
        columns=columns,
    )
    evidence = pd.DataFrame(extract_monthly_evidence(monthly), columns=EVIDENCE_COLUMNS)
    events = pd.DataFrame(
        columns=[
            "customer_key",
            "event_time",
            "event_kind",
            "event_group",
            "source_row_no",
            "negative",
        ]
    )
    _, types, selected = score_at(
        evidence,
        events,
        ["two-months", "twenty", "above-twenty"],
        "2026-08-01",
        RuleConfig(potential_low_min_types=1, medium_priority_min_types=1),
    )
    hits = set(types.loc[types["is_hit"].eq(1), "customer_key"])

    assert hits == {"two-months", "above-twenty"}
    assert set(selected["rule_code"]) == {
        "DATA_OVERAGE_MONTHS_GE_2",
        "DATA_OVERAGE_TOTAL_FEE_GT_20",
    }


def test_low_bandwidth_home_excludes_complaint_and_overage_customers():
    household = pd.DataFrame(
        [
            ["home-only", None, "b1", 300, None, None, None, None, None],
            ["with-complaint", None, "b2", 100, None, None, None, None, None],
            ["with-overage", None, "b3", 499, None, None, None, None, None],
            ["fast-home", None, "b4", 500, None, None, None, None, None],
        ]
    )
    evidence = extract_household_candidates(household, "2026-08-01")
    evidence.append(
        {
            "customer_key": "with-overage",
            "risk_type": "流量超套风险型",
            "rule_code": "DATA_OVERAGE_MONTH_CANDIDATE",
            "evidence_level": "weak",
            "points": 1,
            "source_sheet": "近半年指标",
            "source_column": "流量超套费用",
            "event_time": pd.Timestamp("2026-07-01"),
            "evidence_value": 21,
            "evidence_excerpt": None,
            "source_row_no": 3,
        }
    )
    events = pd.DataFrame(
        [
            {
                "customer_key": "with-complaint",
                "event_time": pd.Timestamp("2026-07-01"),
                "event_kind": "complaint",
                "event_group": "x",
                "source_row_no": 2,
                "negative": False,
            }
        ]
    )
    _, types, _ = score_at(
        pd.DataFrame(evidence, columns=EVIDENCE_COLUMNS),
        events,
        ["home-only", "with-complaint", "with-overage", "fast-home"],
        "2026-08-01",
        RuleConfig(potential_low_min_types=1, medium_priority_min_types=1),
    )
    home_hits = set(
        types.loc[
            (types["risk_type"] == "家宽体验风险型") & types["is_hit"].eq(1),
            "customer_key",
        ]
    )

    assert home_hits == {"home-only"}


def test_complaint_fallback_excludes_overage_and_special_contact_customers():
    complaints = pd.DataFrame(
        [
            {
                "手机号码": "complaint-only",
                "时间": "20260701",
                "投诉节点": "其他",
                "明细": "普通工单",
            },
            {
                "手机号码": "with-overage",
                "时间": "20260701",
                "投诉节点": "其他",
                "明细": "普通工单",
            },
            {
                "手机号码": "with-special",
                "时间": "20260701",
                "投诉节点": "其他",
                "明细": "普通工单",
            },
        ]
    )
    evidence, complaint_events = extract_text_evidence("投诉明细", complaints)
    evidence.append(
        {
            "customer_key": "with-overage",
            "risk_type": "流量超套风险型",
            "rule_code": "DATA_OVERAGE_MONTH_CANDIDATE",
            "evidence_level": "weak",
            "points": 1,
            "source_sheet": "近半年指标",
            "source_column": "流量超套费用",
            "event_time": pd.Timestamp("2026-06-01"),
            "evidence_value": 21,
            "evidence_excerpt": None,
            "source_row_no": 3,
        }
    )
    special_event = pd.DataFrame(
        [
            {
                "customer_key": "with-special",
                "event_time": pd.Timestamp("2026-07-02"),
                "event_kind": "狼号",
                "event_group": "测评号码",
                "source_row_no": 2,
                "negative": False,
            }
        ]
    )
    events = pd.concat([complaint_events, special_event], ignore_index=True)

    _, types, selected = score_at(
        pd.DataFrame(evidence, columns=EVIDENCE_COLUMNS),
        events,
        ["complaint-only", "with-overage", "with-special"],
        "2026-08-01",
        RuleConfig(potential_low_min_types=1, medium_priority_min_types=1),
    )
    complaint_hits = set(
        types.loc[
            (types["risk_type"] == "投诉未解决疑似型") & types["is_hit"].eq(1),
            "customer_key",
        ]
    )

    assert complaint_hits == {"complaint-only"}
    assert set(
        selected.loc[
            selected["rule_code"].eq("COMPLAINT_NO_OVERAGE_SPECIAL"), "customer_key"
        ]
    ) == {"complaint-only"}
