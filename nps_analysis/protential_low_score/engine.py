from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from ..core import clean_text, parse_business_datetime


@dataclass(frozen=True)
class Rule:
    rule_id: str
    risk_type: str
    condition: dict[str, Any]
    weight_candidates: tuple[int, ...]
    reason: str
    enabled: bool = True


@dataclass(frozen=True)
class OptimizationConfig:
    selection_mode: str = "balanced"
    beta: float = 1.0
    threshold_candidates: tuple[int, ...] = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class SelectedRuleSet:
    weights: dict[str, int]
    threshold: int
    calibration_metrics: dict[str, float]


def load_profiles(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_labels(path: str | Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            phone_id = clean_text(row.get("phone_id"))
            value = clean_text(row.get("is_low_score"))
            if phone_id and value in {"0", "1"}:
                labels[phone_id] = int(value)
    return labels


def _as_of(value: str | datetime) -> datetime:
    parsed = parse_business_datetime(value)
    if getattr(parsed, "to_pydatetime", None) is None:
        raise ValueError(f"Invalid feature_end: {value!r}")
    return parsed.to_pydatetime()


def _events(
    profile: dict[str, Any], as_of: datetime, within_days: int
) -> list[dict[str, Any]]:
    result = []
    for event in profile.get("user_journey", []):
        moment = parse_business_datetime(event.get("event_time"))
        if getattr(moment, "to_pydatetime", None) is None:
            continue
        event_time = moment.to_pydatetime()
        age_days = (as_of - event_time).days
        if 0 <= age_days <= within_days:
            result.append(event)
    return result


def _matches_text(value: Any, patterns: list[str]) -> bool:
    text = clean_text(value) or ""
    return any(pattern in text for pattern in patterns)


def _rule_hit(profile: dict[str, Any], rule: Rule, as_of: datetime) -> bool:
    condition = rule.condition
    kind = condition["kind"]
    if kind == "journey_count":
        events = _events(profile, as_of, int(condition["within_days"]))
        actions = set(condition.get("actions", []))
        return sum(event.get("action") in actions for event in events) >= int(
            condition["min_count"]
        )
    if kind == "journey_category_count":
        events = _events(profile, as_of, int(condition["within_days"]))
        actions = set(condition.get("actions", []))
        field = condition.get("field", "intent")
        patterns = list(condition.get("patterns", []))
        count = sum(
            event.get("action") in actions and _matches_text(event.get(field), patterns)
            for event in events
        )
        return count >= int(condition["min_count"])
    if kind == "monthly_positive_count":
        values = list(profile.get("recent_metrics", {}).get(condition["metric"], []))
        valid = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and value >= 0
        ]
        return sum(
            value > float(condition.get("minimum", 0))
            for value in valid[-int(condition["recent_months"]) :]
        ) >= int(condition["min_count"])
    if kind == "monthly_sum_gt":
        values = list(profile.get("recent_metrics", {}).get(condition["metric"], []))
        valid = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and value >= 0
        ]
        return sum(valid[-int(condition["recent_months"]) :]) > float(
            condition["value"]
        )
    if kind == "static_numeric_lt":
        value = profile.get("basic_info", {}).get(condition["field"])
        return isinstance(value, (int, float)) and value < float(condition["value"])
    raise ValueError(f"Unsupported rule condition kind: {kind!r}")


def build_hits(
    profiles: list[dict[str, Any]], rules: list[Rule], feature_end: str
) -> dict[str, dict[str, bool]]:
    as_of = _as_of(feature_end)
    return {
        profile["phone_id"]: {
            rule.rule_id: _rule_hit(profile, rule, as_of)
            for rule in rules
            if rule.enabled
        }
        for profile in profiles
        if clean_text(profile.get("phone_id"))
    }


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
    }


def _objective(
    metrics: dict[str, float], config: OptimizationConfig
) -> tuple[float, float, float]:
    if config.selection_mode == "precision":
        return metrics["precision"], metrics["recall"], -metrics["fp"]
    if config.selection_mode == "recall":
        return metrics["recall"], metrics["precision"], -metrics["fp"]
    beta_squared = config.beta**2
    precision, recall = metrics["precision"], metrics["recall"]
    f_beta = (
        (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)
        if precision + recall
        else 0.0
    )
    return f_beta, precision, recall


def optimize_rules(
    hits: dict[str, dict[str, bool]],
    labels: dict[str, int],
    rules: list[Rule],
    config: OptimizationConfig,
) -> SelectedRuleSet:
    ids = sorted(set(hits) & set(labels))
    if len(ids) < 4 or len(set(labels[phone_id] for phone_id in ids)) < 2:
        raise ValueError(
            "At least four measured users with both label classes are required"
        )
    y = np.array([labels[phone_id] for phone_id in ids])
    enabled_rules = [rule for rule in rules if rule.enabled]
    hit_matrix = np.array(
        [
            [hits[phone_id].get(rule.rule_id, False) for rule in enabled_rules]
            for phone_id in ids
        ],
        dtype=int,
    )
    best: (
        tuple[tuple[float, float, float], dict[str, int], int, dict[str, float]] | None
    ) = None
    for candidate_weights in product(
        *(rule.weight_candidates for rule in enabled_rules)
    ):
        scores = hit_matrix @ np.array(candidate_weights)
        for threshold in config.threshold_candidates:
            metrics = _metrics(y, (scores >= threshold).astype(int))
            candidate = (
                _objective(metrics, config),
                dict(zip((rule.rule_id for rule in enabled_rules), candidate_weights)),
                threshold,
                metrics,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    weights, threshold, calibration_metrics = best[1:]
    return SelectedRuleSet(weights, threshold, calibration_metrics)


def evaluate_rules(
    hits: dict[str, dict[str, bool]],
    labels: dict[str, int],
    rules: list[Rule],
    selected: SelectedRuleSet,
) -> dict[str, float]:
    ids = sorted(set(hits) & set(labels))
    if not ids:
        raise ValueError("No evaluation-label users were found in the user profiles")
    actual = np.array([labels[phone_id] for phone_id in ids])
    scores = np.array(
        [
            sum(
                selected.weights.get(rule.rule_id, 0)
                for rule in rules
                if rule.enabled and hits[phone_id].get(rule.rule_id, False)
            )
            for phone_id in ids
        ]
    )
    return _metrics(actual, (scores >= selected.threshold).astype(int))


def predict(
    hits: dict[str, dict[str, bool]], rules: list[Rule], selected: SelectedRuleSet
) -> list[dict[str, Any]]:
    rule_by_id = {rule.rule_id: rule for rule in rules}
    results = []
    for phone_id, user_hits in hits.items():
        matched = [
            rule_by_id[rule_id]
            for rule_id, hit in user_hits.items()
            if hit and selected.weights.get(rule_id, 0) > 0
        ]
        is_low = (
            sum(selected.weights[rule.rule_id] for rule in matched)
            >= selected.threshold
        )
        results.append(
            {
                "phone_id": phone_id,
                "is_potential_low": int(is_low),
                "primary_type": matched[0].risk_type if matched else "",
                "risk_reasons": "|".join(rule.reason for rule in matched),
                "hit_rule_count": len(matched),
            }
        )
    return results
