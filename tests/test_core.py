import pandas as pd

from nps_analysis.core import (
    benjamini_hochberg,
    is_subquestion_hit,
    month_start,
    parse_business_datetime,
    parse_score,
    parse_subquestion_header,
    segment_scores,
)
from nps_analysis.pipeline import (
    PipelineContext,
    _feature_result,
    build_analysis_sample,
    build_static_feature_profile,
    static_profile_feature_names,
)


def test_confirmed_segment_rule_uses_any_low_score():
    assert segment_scores([10, "6", "\\N", -1]) == ("low", 2, 1)
    assert segment_scores([7, 8, 9, 10, "\\N"]) == ("non_low", 4, 0)
    assert segment_scores(["\\N", -1, None]) == ("unlabeled", 0, 0)


def test_score_validation_distinguishes_invalid_values():
    assert parse_score("10") == (10, None)
    assert parse_score(1) == (1, None)
    assert parse_score(0) == (None, "out_of_range")
    assert parse_score("6.5") == (None, "not_integer")
    assert parse_score(11) == (None, "out_of_range")
    assert parse_score("bad") == (None, "not_numeric")


def test_subquestion_header_and_hit_are_column_driven():
    header = "Q1-1. 您遇到过哪些问题？（可多选） 2 关键限制条件隐瞒或不突出"
    parsed = parse_subquestion_header(header)
    assert parsed == {
        "parent_question_code": "Q1",
        "subquestion_code": "Q1-1",
        "option_code": "2",
        "option_text": "关键限制条件隐瞒或不突出",
    }
    assert is_subquestion_hit("2", "2") == (True, None)
    assert is_subquestion_hit("-1", "2") == (False, None)
    assert is_subquestion_hit("3", "2") == (False, "subquestion_value_mismatch")


def test_month_and_fdr_helpers():
    assert month_start(202608) == pd.Timestamp("2026-08-01")
    adjusted = benjamini_hochberg(pd.Series([0.01, 0.04, 0.03]))
    assert adjusted.round(2).tolist() == [0.03, 0.04, 0.04]


def test_business_datetime_parser_handles_source_formats():
    assert parse_business_datetime(20260706) == pd.Timestamp("2026-07-06")
    assert parse_business_datetime("20260806194448") == pd.Timestamp(
        "2026-08-06 19:44:48"
    )
    assert parse_business_datetime(45567) == pd.Timestamp("2024-10-02")
    assert pd.isna(parse_business_datetime(123456789))


def test_analysis_sample_resolves_geography_and_month_duplicates(tmp_path):
    survey = pd.DataFrame(
        {
            "survey_id": ["s1"],
            "customer_key": ["c1"],
            "survey_month": [pd.Timestamp("2026-08-01")],
            "segment_label": ["low"],
            "city": ["问卷城市"],
            "district": ["问卷区县"],
            "service_center": ["问卷中心"],
            "source_row_no": [3],
            "source_batch_id": ["b1"],
        }
    )
    customer = pd.DataFrame(
        {
            "customer_key": ["c1"],
            "city": ["客户城市"],
            "district": ["客户区县"],
            "service_center": ["客户中心"],
        }
    )
    members = pd.DataFrame(
        {
            "customer_key": ["c1"],
            "household_key": ["h1"],
            "is_primary_payer": [1],
        }
    )
    metrics = [
        "arpu",
        "voice_total",
        "data_total",
        "voice_overage_fee",
        "data_overage_fee",
        "broadband_fee",
        "included_data",
        "used_data",
        "included_voice",
        "used_voice",
    ]
    monthly = pd.DataFrame(
        {
            "customer_key": ["c1", "c1"],
            "month": [pd.Timestamp("2026-07-01")] * 2,
            **{metric: [1.0, 3.0] for metric in metrics},
        }
    )
    context = PipelineContext(tmp_path / "input.xlsx", tmp_path, "b1", "v1")
    context.prepare()

    events = pd.DataFrame(
        columns=["customer_key", "event_time", "event_id", "event_type"]
    )
    events["event_time"] = pd.to_datetime(events["event_time"])
    sample, trajectory = build_analysis_sample(
        context, survey, customer, members, monthly, events
    )

    assert sample.loc[0, "city"] == "问卷城市"
    assert "city_survey" not in sample.columns
    assert len(trajectory) == 1
    assert sample.loc[0, "arpu_valid_months"] == 1
    assert sample.loc[0, "arpu_last"] == 2.0


def test_numeric_feature_lift_uses_quantile_groups():
    feature = pd.Series(range(100))
    labels = pd.Series(["low"] * 10 + ["non_low"] * 90)

    result = _feature_result(feature, labels, "numeric_feature", "static")

    assert result is not None
    assert result["feature_type"] == "numeric"
    assert result["max_low_rate_lift"] <= 10.0


def test_static_profile_contains_all_three_cohorts():
    sample = pd.DataFrame(
        {
            "segment_label": ["low", "non_low", "non_low"],
            "age": [20, 30, 40],
            "city": ["A", "A", "B"],
        }
    )

    profile = build_static_feature_profile(sample, ["age", "city"])

    assert set(profile["cohort"]) == {"all", "low", "non_low"}
    assert (
        profile.loc[
            (profile["cohort"] == "all") & (profile["feature_name"] == "city"),
            "value_count",
        ].sum()
        == 3
    )


def test_static_profile_candidates_include_constant_features():
    sample = pd.DataFrame(
        {
            "survey_id": ["s1", "s2"],
            "customer_key": ["c1", "c2"],
            "segment_label": ["low", "non_low"],
            "constant_static_feature": ["same", "same"],
            "arpu_mean_6m": [1.0, 2.0],
        }
    )

    names = static_profile_feature_names(sample, {"arpu_mean_6m"})

    assert names == ["constant_static_feature"]
