from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score

NULL_STRINGS = {"", "\\N", "NONE", "NULL", "NAN", "NAT"}
SCORE_CODES = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q98", "N1")


def clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if text.upper() in NULL_STRINGS:
        return None
    return re.sub(r"\s+", " ", text)


def normalize_identifier(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return re.sub(r"\.0$", "", text).replace(" ", "")


def parse_score(value: Any) -> tuple[int | None, str | None]:
    text = clean_text(value)
    if text is None or text == "-1":
        return None, "missing_or_sentinel"
    try:
        number = float(text)
    except ValueError:
        return None, "not_numeric"
    if not number.is_integer():
        return None, "not_integer"
    score = int(number)
    if not 0 <= score <= 10:
        return None, "out_of_range"
    return score, None


def segment_scores(values: Iterable[Any]) -> tuple[str, int, int]:
    scores = [score for score, _ in map(parse_score, values) if score is not None]
    low_count = sum(score <= 6 for score in scores)
    if not scores:
        return "unlabeled", 0, 0
    return ("low" if low_count else "non_low"), len(scores), low_count


def parse_subquestion_header(header: str) -> dict[str, str] | None:
    text = clean_text(header)
    if text is None or "填写内容" in text:
        return None
    match = re.match(r"^(Q\d+-\d+)\..*?[（(]可多选[）)]\s*(\d+)\s+(.+)$", text)
    if not match:
        return None
    subquestion_code, option_code, option_text = match.groups()
    return {
        "parent_question_code": subquestion_code.split("-")[0],
        "subquestion_code": subquestion_code,
        "option_code": option_code,
        "option_text": option_text.strip(),
    }


def is_subquestion_hit(value: Any, option_code: str) -> tuple[bool, str | None]:
    text = clean_text(value)
    if text is None:
        return False, None
    if text == "-1":
        return False, None
    if text == option_code or text == f"{option_code}.0":
        return True, None
    return False, "subquestion_value_mismatch"


def month_start(value: Any) -> pd.Timestamp:
    text = normalize_identifier(value)
    if text is None:
        return pd.NaT
    if re.fullmatch(r"\d{6}", text):
        return pd.to_datetime(text + "01", format="%Y%m%d", errors="coerce")
    if re.fullmatch(r"\d{8}", text):
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        return parsed.to_period("M").to_timestamp() if not pd.isna(parsed) else pd.NaT
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.to_period("M").to_timestamp() if not pd.isna(parsed) else pd.NaT


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().clip(0, 1)
    if valid.empty:
        return result
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy() * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    result.loc[order] = adjusted
    return result


def cramers_v(contingency: pd.DataFrame) -> float:
    if contingency.empty or min(contingency.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(contingency, correction=False)[0]
    total = contingency.to_numpy().sum()
    denominator = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    return math.sqrt((chi2 / total) / denominator) if total and denominator else 0.0


def standardized_mean_difference(low: pd.Series, non_low: pd.Series) -> float:
    low = pd.to_numeric(low, errors="coerce").dropna()
    non_low = pd.to_numeric(non_low, errors="coerce").dropna()
    if len(low) < 2 or len(non_low) < 2:
        return np.nan
    pooled = math.sqrt((low.var(ddof=1) + non_low.var(ddof=1)) / 2)
    return (low.mean() - non_low.mean()) / pooled if pooled else 0.0


def information_value(feature: pd.Series, label: pd.Series, bins: int = 10) -> float:
    valid = label.isin(["low", "non_low"])
    x = feature.loc[valid]
    y = label.loc[valid]
    numeric = pd.to_numeric(x, errors="coerce")
    if numeric.notna().mean() >= 0.8 and numeric.nunique() > bins:
        try:
            grouped = pd.qcut(numeric, q=bins, duplicates="drop").astype(str)
        except ValueError:
            grouped = numeric.astype(str)
    else:
        grouped = x.fillna("<MISSING>").astype(str)
    grouped = grouped.fillna("<MISSING>")
    table = pd.crosstab(grouped, y).reindex(columns=["low", "non_low"], fill_value=0)
    if table.empty:
        return 0.0
    low_dist = (table["low"] + 0.5) / (table["low"].sum() + 0.5 * len(table))
    high_dist = (table["non_low"] + 0.5) / (table["non_low"].sum() + 0.5 * len(table))
    return float(((low_dist - high_dist) * np.log(low_dist / high_dist)).sum())


def mutual_information(feature: pd.Series, label: pd.Series, bins: int = 10) -> float:
    valid = label.isin(["low", "non_low"])
    x = feature.loc[valid]
    y = label.loc[valid]
    numeric = pd.to_numeric(x, errors="coerce")
    if numeric.notna().mean() >= 0.8 and numeric.nunique() > bins:
        try:
            x = pd.qcut(numeric, q=bins, duplicates="drop").astype(str)
        except ValueError:
            x = numeric.astype(str)
    return float(mutual_info_score(x.fillna("<MISSING>").astype(str), y))
