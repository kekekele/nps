from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .profile_config import SheetSpec


@dataclass(frozen=True)
class JourneySheetSpec:
    sheet: SheetSpec
    phone_column: str
    time_column: str


@dataclass(frozen=True)
class JourneyFieldTemplate:
    business: str = "其他"
    intent: str = "其他"


@dataclass(frozen=True)
class MatchValueRule:
    pattern: str
    value: str
    match_type: str = "regex"


@dataclass(frozen=True)
class OutputFieldBinding:
    target: str
    column: str
    transform: str = "text"


@dataclass(frozen=True)
class RowFilter:
    column: str
    pattern: str
    match_type: str = "contains"
    negate: bool = False


@dataclass(frozen=True)
class TextValueRule:
    rule_id: str
    pattern: str
    value: str
    negation_pattern: str = ""
    apply_to: tuple[str, ...] = ()


@dataclass(frozen=True)
class BusinessRule:
    business: str
    pattern: str


@dataclass(frozen=True)
class EventSourceConfig:
    source_name: str
    sheet: JourneySheetSpec
    action: str
    input_fields: tuple[str, ...]
    row_filters: tuple[RowFilter, ...] = ()
    month_range_name: str = ""
    default_intent: str = "其他"
    intent_column: str = ""
    intent_rules: tuple[MatchValueRule, ...] = ()
    field_template: JourneyFieldTemplate = JourneyFieldTemplate()
    output_bindings: tuple[OutputFieldBinding, ...] = ()
    business_text_rule_fields: tuple[str, ...] = ()
    intent_text_rule_fields: tuple[str, ...] = ()
    business_key_fields: tuple[str, ...] = ()
    dedupe_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class JourneyConfig:
    default_business: str = "其他"
    default_intent: str = "其他"
    consultation_month_range: tuple[str, str] = ()
    complaint_month_range: tuple[str, str] = ()
    event_sources: tuple[EventSourceConfig, ...] = ()
    business_rules: tuple[BusinessRule, ...] = ()
    business_text_rules: tuple[TextValueRule, ...] = ()
    intent_text_rules: tuple[TextValueRule, ...] = ()

    @property
    def subscription(self) -> JourneySheetSpec:
        return self._find_sheet("业务订购", "办理")

    @property
    def plan_change(self) -> JourneySheetSpec:
        return self._find_sheet("资费变更", "资费变更")

    @property
    def speed_package(self) -> JourneySheetSpec:
        return self._find_sheet("限速与加包", "办理")

    @property
    def wolf_number(self) -> JourneySheetSpec:
        return self._find_sheet("狼号", "狼号感染")

    @property
    def complaint(self) -> JourneySheetSpec:
        return self._find_sheet("投诉明细", "投诉")

    @property
    def touchpoint(self) -> JourneySheetSpec:
        return self._find_sheet("触点轨迹", "咨询")

    def _find_sheet(self, source_name: str, action: str) -> JourneySheetSpec:
        for source in self.event_sources:
            if source.source_name == source_name and source.action == action:
                return source.sheet
        raise ValueError(
            f"Journey source {source_name!r} for action {action!r} was not configured"
        )


def _sheet_spec(data: dict[str, Any]) -> JourneySheetSpec:
    return JourneySheetSpec(
        sheet=SheetSpec(
            data["sheet_name"],
            int(data["header_row"]),
            int(data["data_start_row"]),
        ),
        phone_column=data["phone_column"],
        time_column=data["time_column"],
    )


def _field_template(data: dict[str, Any] | None) -> JourneyFieldTemplate:
    data = data or {}
    return JourneyFieldTemplate(
        business=data.get("business", "其他"),
        intent=data.get("intent", "其他"),
    )


def _match_value_rules(
    items: list[dict[str, Any]] | None,
) -> tuple[MatchValueRule, ...]:
    return tuple(
        MatchValueRule(
            pattern=item["pattern"],
            value=item["value"],
            match_type=item.get("match_type", "regex"),
        )
        for item in (items or [])
    )


def _row_filters(items: list[dict[str, Any]] | None) -> tuple[RowFilter, ...]:
    return tuple(
        RowFilter(
            column=item["column"],
            pattern=item["pattern"],
            match_type=item.get("match_type", "contains"),
            negate=bool(item.get("negate", False)),
        )
        for item in (items or [])
    )


def _output_bindings(
    items: list[dict[str, Any]] | None,
) -> tuple[OutputFieldBinding, ...]:
    return tuple(
        OutputFieldBinding(
            target=item["target"],
            column=item["column"],
            transform=item.get("transform", "text"),
        )
        for item in (items or [])
    )


def _event_sources(items: list[dict[str, Any]]) -> tuple[EventSourceConfig, ...]:
    sources: list[EventSourceConfig] = []
    for item in items:
        shared_text_rule_fields = tuple(item.get("text_rule_fields", []))
        sources.append(
            EventSourceConfig(
                source_name=item["source_name"],
                sheet=_sheet_spec(item["sheet"]),
                action=item["action"],
                input_fields=tuple(item.get("input_fields", [])),
                row_filters=_row_filters(item.get("row_filters")),
                month_range_name=item.get("month_range_name", ""),
                default_intent=item.get("default_intent", "其他"),
                intent_column=item.get("intent_column", ""),
                intent_rules=_match_value_rules(item.get("intent_rules")),
                field_template=_field_template(item.get("field_template")),
                output_bindings=_output_bindings(item.get("output_bindings")),
                business_text_rule_fields=tuple(
                    item.get("business_text_rule_fields", shared_text_rule_fields)
                ),
                intent_text_rule_fields=tuple(
                    item.get("intent_text_rule_fields", shared_text_rule_fields)
                ),
                business_key_fields=tuple(item.get("business_key_fields", [])),
                dedupe_fields=tuple(item.get("dedupe_fields", [])),
            )
        )
    return tuple(sources)


def _business_rules(items: list[dict[str, Any]]) -> tuple[BusinessRule, ...]:
    return tuple(BusinessRule(item["business"], item["pattern"]) for item in items)


def _text_value_rules(groups: list[dict[str, Any]] | None) -> tuple[TextValueRule, ...]:
    rules: list[TextValueRule] = []
    for group in groups or []:
        action = group["action"]
        for item in group.get("rules", []):
            rules.append(
                TextValueRule(
                    rule_id=item["rule_id"],
                    pattern=item["pattern"],
                    value=item["value"],
                    negation_pattern=item.get("negation_pattern", ""),
                    apply_to=tuple(item.get("apply_to", [action])),
                )
            )
    return tuple(rules)


def load_journey_config(path: str | Path | None = None) -> JourneyConfig:
    config_path = (
        Path(path) if path else Path(__file__).with_name("journey_config.yaml")
    )
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return JourneyConfig(
        default_business=data.get("default_business", "其他"),
        default_intent=data.get("default_intent", "其他"),
        consultation_month_range=tuple(data.get("consultation_month_range", [])),
        complaint_month_range=tuple(data.get("complaint_month_range", [])),
        event_sources=_event_sources(data.get("event_sources", [])),
        business_rules=_business_rules(data.get("business_rules", [])),
        business_text_rules=_text_value_rules(data.get("business_text_rules", [])),
        intent_text_rules=_text_value_rules(data.get("intent_text_rules", [])),
    )


DEFAULT_JOURNEY_CONFIG = load_journey_config()
