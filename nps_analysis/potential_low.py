from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .core import (
    benjamini_hochberg,
    clean_text,
    normalize_identifier,
    parse_business_datetime,
    segment_scores,
    standardized_mean_difference,
)
from .pipeline import (
    BASIC_COLUMNS,
    HOUSEHOLD_COLUMNS,
    MONTHLY_COLUMNS,
    SCORE_CODES,
    _clean_frame,
    _read_sheet,
    _rename_by_position,
    _score_column,
)

EVIDENCE_COLUMNS = [
    "customer_key",
    "risk_type",
    "rule_code",
    "evidence_level",
    "points",
    "source_sheet",
    "source_column",
    "event_time",
    "evidence_value",
    "evidence_excerpt",
    "source_row_no",
]

VALIDATION_COLUMNS = [
    "rule_version",
    "survey_month",
    "risk_type",
    "survey_count",
    "actual_low_count",
    "actual_non_low_count",
    "predicted_count",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "f1",
    "jaccard",
    "actual_low_rate",
    "predicted_low_rate",
    "lift",
]

VALIDATION_DETAIL_COLUMNS = [
    "survey_id",
    "customer_key",
    "survey_month",
    "actual_segment",
    "actual_low",
    "predicted_low",
    "match_result",
    "total_risk_score",
    "primary_type",
    "secondary_types",
    "hit_type_count",
    "rule_version",
]

RISK_TYPES = (
    "投诉未解决疑似型",
    "投诉高风险型",
    "家宽体验风险型",
    "流量超套风险型",
    "套餐资费不满疑似型",
    "办理变更退订受阻疑似型",
    "营销宣传争议型",
    "提醒告知不足疑似型",
    "高频触点负面型",
)

TEXT_COLUMNS = {
    "投诉明细": ["投诉节点", "明细"],
    "触点轨迹": [
        "触点类型",
        "触点小类",
        "投诉类型",
        "节点",
        "明细",
        "投诉明细",
        "小热线工单明细",
        "小热线来电明细",
        "语音转文本-客户",
        "AI提炼",
        "问题分类",
        "情绪分类",
    ],
}

PATTERNS = {
    "unresolved": re.compile(
        r"未解决|仍未.{0,8}(处理|解决)|再次投诉|重复投诉|多次反映.{0,8}(未|没有).{0,8}(处理|解决)"
    ),
    "broadband_entity": re.compile(r"宽带|家宽|光猫|路由器|WiFi", re.I),
    "broadband_problem": re.compile(
        r"断网|掉线|无法上网|网速慢|卡顿|故障|维修.{0,8}(未恢复|没修好)"
    ),
    "traffic_dissatisfaction": re.compile(
        r"流量不够|流量用完|超套.{0,8}(扣费|收费)|限速|加速包"
    ),
    "price_dissatisfaction": re.compile(
        r"套餐贵|资费贵|性价比低|收费不合理|违约金|优惠到期"
    ),
    "service_blocked": re.compile(
        r"无法办理|办不了|无法退订|强制绑定|限制条件|流程复杂|未告知"
    ),
    "marketing_dispute": re.compile(
        r"夸大宣传|与实际不符|隐瞒.{0,8}(条件|限制)|诱导|未经同意|不知情.{0,8}(办理|开通)"
    ),
    "reminder_missing": re.compile(r"未提醒|没有提醒|提醒不及时|未告知.{0,8}(收费|费用)"),
    "negative": re.compile(r"不满|生气|愤怒|差评|负向|负面|投诉|离网|销户"),
}

NEGATED = re.compile(r"已解决|处理完成|已经处理|没有投诉|无投诉|未出现故障|没有故障")


@dataclass(frozen=True)
class RuleConfig:
    version: str = "potential-low-v1"
    lookback_days: int = 183
    repeat_days: int = 90
    repeat_complaint_count: int = 2
    repeat_plan_change_count: int = 2
    repeat_subscription_count: int = 2
    negative_contact_count: int = 3
    type_threshold: int = 3


def _event_time(value: Any) -> pd.Timestamp:
    parsed = parse_business_datetime(value)
    if pd.isna(parsed) or not 2000 <= parsed.year <= datetime.now().year + 2:
        return pd.NaT
    return parsed


def _excerpt(text: str, match: re.Match[str] | None = None, radius: int = 28) -> str:
    if match is None:
        return text[: radius * 2]
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return text[start:end]


def _is_negated(text: str, match: re.Match[str]) -> bool:
    start = max(0, match.start() - 12)
    end = min(len(text), match.end() + 12)
    return bool(NEGATED.search(text[start:end]))


def _evidence(
    customer_key: str,
    risk_type: str,
    rule_code: str,
    level: str,
    source_sheet: str,
    source_column: str,
    event_time: Any,
    value: Any,
    source_row_no: int,
    excerpt: str | None = None,
) -> dict[str, Any]:
    points = {"strong": 3, "medium": 2, "weak": 1}[level]
    return {
        "customer_key": customer_key,
        "risk_type": risk_type,
        "rule_code": rule_code,
        "evidence_level": level,
        "points": points,
        "source_sheet": source_sheet,
        "source_column": source_column,
        "event_time": event_time,
        "evidence_value": value,
        "evidence_excerpt": excerpt,
        "source_row_no": source_row_no,
    }


def _prepare(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return _clean_frame(_rename_by_position(frame, names))


def extract_monthly_evidence(frame: pd.DataFrame) -> list[dict[str, Any]]:
    data = _prepare(frame, MONTHLY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for index, row in data.iterrows():
        customer = normalize_identifier(row["phone"])
        when = _event_time(row["month"])
        if customer is None or pd.isna(when):
            continue
        overage = pd.to_numeric(row["data_overage_fee"], errors="coerce")
        included = pd.to_numeric(row["included_data"], errors="coerce")
        used = pd.to_numeric(row["used_data"], errors="coerce")
        if pd.notna(overage) and overage > 0:
            rows.append(
                _evidence(
                    customer,
                    "流量超套风险型",
                    "DATA_OVERAGE_FEE",
                    "strong",
                    "近半年指标",
                    "流量超套费用",
                    when,
                    float(overage),
                    index + 3,
                )
            )
        if pd.notna(included) and pd.notna(used) and used > included:
            rows.append(
                _evidence(
                    customer,
                    "流量超套风险型",
                    "DATA_USAGE_OVER_RESOURCE",
                    "strong",
                    "近半年指标",
                    "gprs_used_v/gprs_total",
                    when,
                    float(used - included),
                    index + 3,
                )
            )
    return rows


def extract_structured_event_evidence(
    sheet_name: str, frame: pd.DataFrame
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    specs = {
        "业务订购": (0, 5, 3, 4, 2),
        "资费变更": (1, 10, 9, 6, 3),
        "限速与加包": (2, 6, 0, 8, 3),
        "狼号": (0, 2, 3, 1, 2),
    }
    phone_col, time_col, subtype_col, detail_col, start_row = specs[sheet_name]
    events: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for offset, values in enumerate(frame.itertuples(index=False, name=None), start=start_row):
        customer = normalize_identifier(values[phone_col])
        when = _event_time(values[time_col])
        if customer is None or pd.isna(when):
            continue
        subtype = clean_text(values[subtype_col])
        detail = clean_text(values[detail_col])
        event_group = (
            f"{subtype or '<MISSING>'}|{detail or '<MISSING>'}"
            if sheet_name == "业务订购"
            else subtype or detail or "<MISSING>"
        )
        events.append(
            {
                "customer_key": customer,
                "event_time": when,
                "event_kind": sheet_name,
                "event_group": event_group,
                "source_row_no": offset,
            }
        )
        if sheet_name == "限速与加包":
            evidence.append(
                _evidence(
                    customer,
                    "流量超套风险型",
                    "SPEED_OR_PACKAGE_EVENT",
                    "medium",
                    sheet_name,
                    "业务类型/加速包资费名称",
                    when,
                    subtype or detail,
                    offset,
                )
            )
        elif sheet_name == "资费变更":
            direction = pd.to_numeric(values[subtype_col], errors="coerce")
            before = pd.to_numeric(values[5], errors="coerce")
            after = pd.to_numeric(values[6], errors="coerce")
            if direction == -1 or (pd.notna(before) and pd.notna(after) and after < before):
                evidence.append(
                    _evidence(
                        customer,
                        "套餐资费不满疑似型",
                        "PLAN_DOWNGRADE",
                        "medium",
                        sheet_name,
                        "变更标识/变更前后资费费用",
                        when,
                        f"{before}->{after}",
                        offset,
                    )
                )
        elif sheet_name == "业务订购" and subtype and "退订" in subtype:
            evidence.append(
                _evidence(
                    customer,
                    "套餐资费不满疑似型",
                    "SUBSCRIPTION_CANCEL",
                    "medium",
                    sheet_name,
                    "订购类型",
                    when,
                    subtype,
                    offset,
                )
            )
    return evidence, pd.DataFrame(events)


def extract_text_evidence(sheet_name: str, frame: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    phone_column = "手机号码"
    time_column = "时间" if sheet_name == "投诉明细" else "触点时间"
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    start_row = 2 if sheet_name == "投诉明细" else 3
    for index, row in frame.iterrows():
        customer = normalize_identifier(row.get(phone_column))
        when = _event_time(row.get(time_column))
        if customer is None or pd.isna(when):
            continue
        source_row = index + start_row
        texts = {
            column: clean_text(row.get(column))
            for column in TEXT_COLUMNS[sheet_name]
            if column in frame.columns and clean_text(row.get(column))
        }
        combined = " | ".join(texts.values())
        category = clean_text(
            row.get("投诉节点" if sheet_name == "投诉明细" else "投诉类型")
        ) or "<MISSING>"
        events.append(
            {
                "customer_key": customer,
                "event_time": when,
                "event_kind": "complaint" if sheet_name == "投诉明细" else "contact",
                "event_group": category,
                "source_row_no": source_row,
                "negative": bool(PATTERNS["negative"].search(combined)),
            }
        )
        if sheet_name == "投诉明细":
            rows.append(
                _evidence(
                    customer,
                    "投诉高风险型",
                    "COMPLAINT_EVENT",
                    "strong",
                    sheet_name,
                    "投诉节点",
                    when,
                    category,
                    source_row,
                )
            )
        if not combined:
            continue
        rules = [
            ("unresolved", "投诉未解决疑似型", "TEXT_UNRESOLVED"),
            ("traffic_dissatisfaction", "流量超套风险型", "TEXT_TRAFFIC_DISSATISFACTION"),
            ("price_dissatisfaction", "套餐资费不满疑似型", "TEXT_PRICE_DISSATISFACTION"),
            ("service_blocked", "办理变更退订受阻疑似型", "TEXT_SERVICE_BLOCKED"),
            ("marketing_dispute", "营销宣传争议型", "TEXT_MARKETING_DISPUTE"),
            ("reminder_missing", "提醒告知不足疑似型", "TEXT_REMINDER_MISSING"),
        ]
        if PATTERNS["broadband_entity"].search(combined) and PATTERNS[
            "broadband_problem"
        ].search(combined):
            rules.append(("broadband_problem", "家宽体验风险型", "TEXT_BROADBAND_PROBLEM"))
        for pattern_name, risk_type, rule_code in rules:
            match = PATTERNS[pattern_name].search(combined)
            if not match or _is_negated(combined, match):
                continue
            source_column = next(
                (column for column, text in texts.items() if PATTERNS[pattern_name].search(text)),
                "+".join(texts),
            )
            rows.append(
                _evidence(
                    customer,
                    risk_type,
                    rule_code,
                    "strong",
                    sheet_name,
                    source_column,
                    when,
                    "regex_hit",
                    source_row,
                    _excerpt(combined, match),
                )
            )
    return rows, pd.DataFrame(events)


def _window_derived_evidence(events: pd.DataFrame, as_of: pd.Timestamp, config: RuleConfig) -> list[dict[str, Any]]:
    if events.empty:
        return []
    start = as_of - pd.Timedelta(days=config.repeat_days)
    current = events.loc[events["event_time"].between(start, as_of)].copy()
    rows: list[dict[str, Any]] = []
    definitions = [
        ("complaint", config.repeat_complaint_count, "投诉未解决疑似型", "REPEAT_COMPLAINT", "strong"),
        ("资费变更", config.repeat_plan_change_count, "套餐资费不满疑似型", "REPEAT_PLAN_CHANGE", "medium"),
        ("业务订购", config.repeat_subscription_count, "套餐资费不满疑似型", "REPEAT_SUBSCRIPTION", "medium"),
    ]
    for kind, threshold, risk_type, rule_code, level in definitions:
        subset = current.loc[current["event_kind"].eq(kind)]
        for (customer, group), part in subset.groupby(["customer_key", "event_group"], dropna=False):
            if len(part) >= threshold:
                rows.append(
                    _evidence(
                        customer,
                        risk_type,
                        rule_code,
                        level,
                        "窗口聚合",
                        "event_group",
                        part["event_time"].max(),
                        len(part),
                        int(part["source_row_no"].iloc[-1]),
                    )
                )
    negative = (
        current["negative"].fillna(False)
        if "negative" in current
        else pd.Series(False, index=current.index)
    )
    contacts = current.loc[current["event_kind"].eq("contact") & negative]
    for customer, part in contacts.groupby("customer_key"):
        if len(part) >= config.negative_contact_count:
            rows.append(
                _evidence(
                    customer,
                    "高频触点负面型",
                    "FREQUENT_NEGATIVE_CONTACT",
                    "strong",
                    "触点轨迹",
                    "情绪分类/负面文本",
                    part["event_time"].max(),
                    len(part),
                    int(part["source_row_no"].iloc[-1]),
                )
            )
    return rows


def score_at(
    base_evidence: pd.DataFrame,
    events: pd.DataFrame,
    customers: Iterable[str],
    as_of: Any,
    config: RuleConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cutoff = _event_time(as_of)
    if pd.isna(cutoff):
        raise ValueError("as_of must be a valid business date")
    start = cutoff - pd.Timedelta(days=config.lookback_days)
    selected = base_evidence.loc[base_evidence["event_time"].between(start, cutoff)].copy()
    derived = pd.DataFrame(_window_derived_evidence(events, cutoff, config), columns=EVIDENCE_COLUMNS)
    selected = pd.concat([selected, derived], ignore_index=True)

    overage_customers = set(selected.loc[selected["rule_code"].isin({"DATA_OVERAGE_FEE", "DATA_USAGE_OVER_RESOURCE"}), "customer_key"])
    selected = selected.loc[
        ~selected["rule_code"].eq("TEXT_REMINDER_MISSING")
        | selected["customer_key"].isin(overage_customers)
    ]
    selected = selected.drop_duplicates(
        ["customer_key", "risk_type", "rule_code", "source_sheet", "source_row_no"]
    )
    grouped = (
        selected.groupby(["customer_key", "risk_type"], as_index=False)
        .agg(
            type_score=("points", "sum"),
            evidence_count=("rule_code", "nunique"),
            strong_count=("evidence_level", lambda value: int((value == "strong").sum())),
            latest_evidence_time=("event_time", "max"),
        )
    )
    grid = pd.MultiIndex.from_product(
        [sorted(set(customers)), RISK_TYPES], names=["customer_key", "risk_type"]
    ).to_frame(index=False)
    user_type = grid.merge(grouped, how="left", on=["customer_key", "risk_type"])
    for column in ("type_score", "evidence_count", "strong_count"):
        user_type[column] = user_type[column].fillna(0).astype(int)
    user_type["is_hit"] = user_type["type_score"].ge(config.type_threshold).astype(int)
    user_type["rule_version"] = config.version
    user_type["as_of"] = cutoff

    hit = user_type.loc[user_type["is_hit"].eq(1)].sort_values(
        ["customer_key", "type_score", "strong_count", "latest_evidence_time"],
        ascending=[True, False, False, False],
    )
    summaries: list[dict[str, Any]] = []
    for customer in sorted(set(customers)):
        part = hit.loc[hit["customer_key"].eq(customer)]
        summaries.append(
            {
                "customer_key": customer,
                "as_of": cutoff,
                "total_risk_score": int(part["type_score"].sum()),
                "is_potential_low": int(not part.empty),
                "primary_type": part.iloc[0]["risk_type"] if not part.empty else None,
                "secondary_types": "|".join(part.iloc[1:]["risk_type"].tolist()),
                "hit_type_count": len(part),
                "rule_version": config.version,
            }
        )
    return pd.DataFrame(summaries), user_type, selected


def _customer_features(basic: pd.DataFrame, household: pd.DataFrame) -> pd.DataFrame:
    base = _prepare(basic, BASIC_COLUMNS)
    base["customer_key"] = base["phone"].map(normalize_identifier)
    base = base.dropna(subset=["customer_key"]).drop_duplicates("customer_key")
    keep = [
        "customer_key", "is_household", "city", "district", "gender", "age",
        "tenure_months", "device_vendor", "device_type", "main_plan_fee",
    ]
    result = base[keep].copy()
    home = _prepare(household, HOUSEHOLD_COLUMNS)
    home["customer_key"] = home["phone"].map(normalize_identifier)
    for column in ("bandwidth", "broadband_fee", "broadband_usage"):
        home[column] = pd.to_numeric(home[column], errors="coerce")
    home = home.dropna(subset=["customer_key"]).drop_duplicates("customer_key")
    return result.merge(
        home[["customer_key", "is_primary_payer", "bandwidth", "broadband_fee", "broadband_usage"]],
        how="left",
        on="customer_key",
    )


def _behavior_features(
    monthly: pd.DataFrame,
    events: pd.DataFrame,
    as_of: Any,
    config: RuleConfig,
) -> pd.DataFrame:
    cutoff = _event_time(as_of)
    start = cutoff - pd.Timedelta(days=config.lookback_days)
    data = _prepare(monthly, MONTHLY_COLUMNS)
    data["customer_key"] = data["phone"].map(normalize_identifier)
    data["month"] = data["month"].map(_event_time)
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
    for column in metrics:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.loc[data["month"].between(start, cutoff)]
    monthly_features = data.groupby("customer_key")[metrics].mean().add_suffix("_mean_6m")

    if events.empty:
        return monthly_features.reset_index()
    current_events = events.loc[events["event_time"].between(start, cutoff)].copy()
    event_features = (
        current_events.groupby(["customer_key", "event_kind"])
        .size()
        .unstack(fill_value=0)
        .add_suffix("_count_183d")
    )
    return monthly_features.join(event_features, how="outer").reset_index()


def build_type_profile(features: pd.DataFrame, user_type: pd.DataFrame) -> pd.DataFrame:
    joined = features.merge(
        user_type.loc[user_type["is_hit"].eq(1), ["customer_key", "risk_type"]],
        how="left",
        on="customer_key",
    )
    rows: list[dict[str, Any]] = []
    feature_columns = [column for column in features if column != "customer_key"]
    for risk_type in RISK_TYPES:
        hit_customers = set(user_type.loc[(user_type["risk_type"] == risk_type) & user_type["is_hit"].eq(1), "customer_key"])
        type_mask = features["customer_key"].isin(hit_customers)
        for cohort, mask in {
            "type_users": type_mask,
            "other_potential_low": features["customer_key"].isin(set(joined.dropna(subset=["risk_type"])["customer_key"]) - hit_customers),
            "all_users": pd.Series(True, index=features.index),
        }.items():
            part = features.loc[mask]
            for column in feature_columns:
                numeric = pd.to_numeric(part[column], errors="coerce")
                is_numeric = numeric.notna().mean() >= 0.8 if len(part) else False
                if is_numeric:
                    comparison = pd.to_numeric(features.loc[~type_mask, column], errors="coerce")
                    valid_numeric = numeric.dropna()
                    valid_comparison = comparison.dropna()
                    p_value = (
                        stats.mannwhitneyu(valid_numeric, valid_comparison).pvalue
                        if len(valid_numeric) and len(valid_comparison)
                        else np.nan
                    )
                    rows.append({
                        "risk_type": risk_type, "cohort": cohort, "feature_name": column,
                        "feature_type": "numeric", "feature_value": None,
                        "user_count": len(part), "missing_rate": float(numeric.isna().mean()) if len(part) else np.nan,
                        "value_count": int(numeric.notna().sum()), "value_rate": None,
                        "mean": numeric.mean(), "median": numeric.median(),
                        "effect_size": standardized_mean_difference(valid_numeric, valid_comparison),
                        "lift": None, "p_value": p_value,
                    })
                else:
                    values = part[column].fillna("<MISSING>").astype(str)
                    counts = values.value_counts().head(20)
                    for value, count in counts.items():
                        type_rate = (features.loc[type_mask, column].fillna("<MISSING>").astype(str) == value).mean() if type_mask.any() else np.nan
                        all_rate = (features[column].fillna("<MISSING>").astype(str) == value).mean() if len(features) else np.nan
                        table = pd.crosstab(type_mask, features[column].fillna("<MISSING>").astype(str) == value)
                        p_value = stats.chi2_contingency(table, correction=False)[1] if table.shape == (2, 2) else np.nan
                        rows.append({
                            "risk_type": risk_type, "cohort": cohort, "feature_name": column,
                            "feature_type": "categorical", "feature_value": value,
                            "user_count": len(part), "missing_rate": float(part[column].isna().mean()) if len(part) else np.nan,
                            "value_count": int(count), "value_rate": count / len(part) if len(part) else np.nan,
                            "mean": None, "median": None,
                            "effect_size": None,
                            "lift": type_rate / all_rate if all_rate else np.nan,
                            "p_value": p_value,
                        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr"] = benjamini_hochberg(result["p_value"])
    return result


def _survey_labels(frame: pd.DataFrame) -> pd.DataFrame:
    headers = [clean_text(column) or f"unnamed_{index}" for index, column in enumerate(frame.columns)]
    data = frame.copy()
    data.columns = headers
    score_columns = [_score_column(headers, code) for code in SCORE_CODES]
    rows = []
    for _, row in data.iterrows():
        customer = normalize_identifier(row.get("手机号码"))
        survey_month = parse_business_datetime(row.get("调研时间"))
        if customer is None or pd.isna(survey_month):
            continue
        rows.append({
            "survey_id": normalize_identifier(row.get("调研流水号")),
            "customer_key": customer,
            "survey_month": survey_month.to_period("M").to_timestamp(),
            "segment_label": segment_scores(row[column] for column in score_columns)[0],
        })
    return pd.DataFrame(rows)


def build_validation_detail(
    surveys: pd.DataFrame, base_evidence: pd.DataFrame, events: pd.DataFrame, config: RuleConfig
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for survey_month, part in surveys.loc[surveys["segment_label"].isin(["low", "non_low"])].groupby("survey_month"):
        users, _, _ = score_at(base_evidence, events, part["customer_key"], survey_month, config)
        evaluated = part.merge(users, on="customer_key", how="left", validate="many_to_one")
        evaluated["actual_segment"] = evaluated.pop("segment_label")
        evaluated["actual_low"] = evaluated["actual_segment"].eq("low").astype(int)
        evaluated["predicted_low"] = evaluated["is_potential_low"].fillna(0).astype(int)
        evaluated["match_result"] = np.select(
            [
                evaluated["actual_low"].eq(1) & evaluated["predicted_low"].eq(1),
                evaluated["actual_low"].eq(0) & evaluated["predicted_low"].eq(1),
                evaluated["actual_low"].eq(1) & evaluated["predicted_low"].eq(0),
            ],
            ["TP", "FP", "FN"],
            default="TN",
        )
        rows.append(evaluated[VALIDATION_DETAIL_COLUMNS])
    if not rows:
        return pd.DataFrame(columns=VALIDATION_DETAIL_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def _validation_metric_row(
    detail: pd.DataFrame,
    survey_month: Any,
    risk_type: str,
    predicted: pd.Series,
    config: RuleConfig,
) -> dict[str, Any]:
    actual = detail["actual_low"].eq(1)
    predicted = predicted.astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    accuracy = (tp + tn) / len(detail) if len(detail) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    jaccard = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    actual_rate = actual.mean() if len(detail) else np.nan
    predicted_rate = predicted.mean() if len(detail) else np.nan
    return {
        "rule_version": config.version,
        "survey_month": survey_month,
        "risk_type": risk_type,
        "survey_count": len(detail),
        "actual_low_count": int(actual.sum()),
        "actual_non_low_count": int((~actual).sum()),
        "predicted_count": int(predicted.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "jaccard": jaccard,
        "actual_low_rate": actual_rate,
        "predicted_low_rate": predicted_rate,
        "lift": precision / actual_rate if actual_rate and pd.notna(precision) else np.nan,
    }


def build_validation(
    surveys: pd.DataFrame,
    base_evidence: pd.DataFrame,
    events: pd.DataFrame,
    config: RuleConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = build_validation_detail(surveys, base_evidence, events, config)
    if detail.empty:
        return pd.DataFrame(columns=VALIDATION_COLUMNS), detail

    metric_rows: list[dict[str, Any]] = []
    periods: list[tuple[Any, pd.DataFrame]] = [("ALL", detail)]
    periods.extend(detail.groupby("survey_month"))
    for survey_month, part in periods:
        metric_rows.append(
            _validation_metric_row(
                part, survey_month, "ALL", part["predicted_low"].eq(1), config
            )
        )
        for risk_type in RISK_TYPES:
            predicted = part["primary_type"].eq(risk_type) | part["secondary_types"].fillna("").str.split("|").map(lambda values: risk_type in values)
            metric_rows.append(
                _validation_metric_row(part, survey_month, risk_type, predicted, config)
            )
    return pd.DataFrame(metric_rows, columns=VALIDATION_COLUMNS), detail


def _rule_dictionary(config: RuleConfig) -> pd.DataFrame:
    records = []
    for name, pattern in PATTERNS.items():
        records.append({
            "rule_code": name, "source": "投诉明细/触点轨迹", "condition": pattern.pattern,
            "evidence_level": "strong", "threshold": None, "rule_version": config.version,
        })
    for code, source, condition, level in [
        ("COMPLAINT_EVENT", "投诉明细", "窗口内存在投诉", "strong"),
        ("DATA_OVERAGE_FEE", "近半年指标.流量超套费用", "> 0", "strong"),
        ("DATA_USAGE_OVER_RESOURCE", "近半年指标.gprs_used_v/gprs_total", "gprs_used_v > gprs_total", "strong"),
        ("PLAN_DOWNGRADE", "资费变更", "变更标识=-1或变更后费用下降", "medium"),
        ("SUBSCRIPTION_CANCEL", "业务订购.订购类型", "包含退订", "medium"),
        ("SPEED_OR_PACKAGE_EVENT", "限速与加包", "窗口内存在记录", "medium"),
    ]:
        records.append({
            "rule_code": code, "source": source, "condition": condition,
            "evidence_level": level, "threshold": None, "rule_version": config.version,
        })
    for code, source, threshold, level in [
        ("REPEAT_COMPLAINT", "投诉明细", config.repeat_complaint_count, "strong"),
        ("REPEAT_PLAN_CHANGE", "资费变更", config.repeat_plan_change_count, "medium"),
        ("REPEAT_SUBSCRIPTION", "业务订购", config.repeat_subscription_count, "medium"),
        ("FREQUENT_NEGATIVE_CONTACT", "触点轨迹", config.negative_contact_count, "strong"),
    ]:
        records.append({
            "rule_code": code, "source": source,
            "condition": f"{config.repeat_days}天窗口内次数达到阈值",
            "evidence_level": level, "threshold": threshold,
            "rule_version": config.version,
        })
    return pd.DataFrame(records)


def run_potential_low(
    workbook: Path,
    output_dir: Path,
    as_of: Any,
    config: RuleConfig | None = None,
) -> Path:
    config = config or RuleConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    quality: list[dict[str, Any]] = []

    required = ["基础指标", "家庭指标", "近半年指标", "业务订购", "投诉明细", "资费变更", "限速与加包", "狼号", "触点轨迹"]
    frames: dict[str, pd.DataFrame] = {}
    for sheet in required:
        try:
            frames[sheet] = _read_sheet(workbook, sheet)
            quality.append({"sheet": sheet, "status": "available", "row_count": len(frames[sheet]), "message": None})
        except Exception as error:
            quality.append({"sheet": sheet, "status": "unavailable", "row_count": 0, "message": str(error)})
    if "基础指标" not in frames or "近半年指标" not in frames:
        raise ValueError("基础指标 and 近半年指标 are required")

    base_rows.extend(extract_monthly_evidence(frames["近半年指标"]))
    for sheet in ("业务订购", "资费变更", "限速与加包", "狼号"):
        if sheet in frames:
            evidence, events = extract_structured_event_evidence(sheet, frames[sheet])
            base_rows.extend(evidence)
            event_frames.append(events)
    for sheet in ("投诉明细", "触点轨迹"):
        if sheet in frames:
            evidence, events = extract_text_evidence(sheet, frames[sheet])
            base_rows.extend(evidence)
            event_frames.append(events)

    base = pd.DataFrame(base_rows, columns=EVIDENCE_COLUMNS)
    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(
        columns=["customer_key", "event_time", "event_kind", "event_group", "source_row_no", "negative"]
    )
    features = _customer_features(frames["基础指标"], frames.get("家庭指标", pd.DataFrame(columns=range(len(HOUSEHOLD_COLUMNS)))))
    behavior = _behavior_features(frames["近半年指标"], events, as_of, config)
    features = features.merge(behavior, how="left", on="customer_key")
    users, user_type, evidence = score_at(base, events, features["customer_key"], as_of, config)
    available_count = sum(sheet in frames for sheet in required)
    users["data_coverage_rate"] = available_count / len(required)
    profile = build_type_profile(features, user_type)

    validation = pd.DataFrame(columns=VALIDATION_COLUMNS)
    validation_detail = pd.DataFrame(columns=VALIDATION_DETAIL_COLUMNS)
    try:
        survey = _read_sheet(workbook, "测评明细")
        validation, validation_detail = build_validation(
            _survey_labels(survey), base, events, config
        )
        quality.append({"sheet": "测评明细", "status": "validation_available", "row_count": len(survey), "message": None})
    except Exception as error:
        quality.append({"sheet": "测评明细", "status": "validation_unavailable", "row_count": 0, "message": str(error)})

    outputs = {
        "potential_low_user.csv": users,
        "potential_low_user_type.csv": user_type,
        "potential_low_evidence.csv": evidence,
        "potential_low_type_profile.csv": profile,
        "rule_validation.csv": validation,
        "rule_validation_detail.csv": validation_detail,
        "rule_dictionary.csv": _rule_dictionary(config),
        "batch_quality.csv": pd.DataFrame(quality),
    }
    files = []
    for name, frame in outputs.items():
        path = output_dir / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        files.append({"file": name, "rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(), "input": str(workbook),
        "as_of": str(_event_time(as_of)), "config": asdict(config), "files": files,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest