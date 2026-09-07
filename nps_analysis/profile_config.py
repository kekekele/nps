from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldSpec:
    key: str
    source: str
    occurrence: int = 1
    value_type: str = "text"


@dataclass(frozen=True)
class SheetSpec:
    name: str
    header_row: int
    data_start_row: int


@dataclass(frozen=True)
class UserProfileConfig:
    basic_sheet: SheetSpec = SheetSpec("基础指标", 1, 3)
    survey_sheet: SheetSpec = SheetSpec("测评明细", 2, 3)
    household_sheet: SheetSpec = SheetSpec("家庭指标", 1, 2)
    monthly_sheet: SheetSpec = SheetSpec("近半年指标", 1, 3)
    basic_phone_column: str = "手机号码"
    basic_household_column: str = "209号码"
    survey_phone_column: str = "手机号码"
    household_broadband_column: str = "宽带号码"
    household_primary_payer_column: str = "是否主付费人"
    monthly_phone_column: str = "手机号码"
    monthly_month_column: str = "月份"
    basic_fields: tuple[FieldSpec, ...] = (
        FieldSpec("customer_type", "类型"),
        FieldSpec("phone_209", "209号码"),
        FieldSpec("is_household", "是否家庭", value_type="bool"),
        FieldSpec("city", "地市名称"),
        FieldSpec("district", "区县名称"),
        FieldSpec("service_center", "服务中心名称"),
        FieldSpec("gender", "性别"),
        FieldSpec("age", "年龄", value_type="number"),
        FieldSpec("open_date", "入网时间"),
        FieldSpec("tenure_months", "在网时长（月）", value_type="number"),
        FieldSpec("global_level", "全球通等级"),
        FieldSpec("term_factory", "终端厂商"),
        FieldSpec("term_model", "终端型号"),
        FieldSpec("term_type", "终端类型"),
        FieldSpec("main_plan_fee", "主套餐费用", value_type="number"),
        FieldSpec("scene", "场景"),
        FieldSpec("enterprise", "政企单位"),
        FieldSpec("phone_280", "280号码"),
        FieldSpec("is_13n", "是否13N", value_type="bool"),
    )
    survey_fields: tuple[FieldSpec, ...] = (
        FieldSpec("is_fttr", "是否FTTR", value_type="bool"),
        FieldSpec("is_bandwidth_ge_500m", "是否带宽大于等于500M", value_type="bool"),
        FieldSpec("is_data_operation_customer", "是否流量运营客户", value_type="bool"),
    )
    household_fields: tuple[FieldSpec, ...] = (
        FieldSpec("broadband_bandwidth", "带宽", value_type="number"),
        FieldSpec("broadband_area", "宽带小区"),
        FieldSpec("broadband_open_date", "开户日期"),
        FieldSpec("broadband_fee", "宽带资费费用", value_type="number"),
        FieldSpec("broadband_usage", "宽带流量", value_type="number"),
    )
    monthly_fields: tuple[FieldSpec, ...] = (
        FieldSpec("arpu", "ARPU 值", value_type="number"),
        FieldSpec("dou", "语音使用量", 1, "number"),
        FieldSpec("mou", "流量使用量", 1, "number"),
        FieldSpec("over_voc_fee", "语言超套费用", value_type="number"),
        FieldSpec("over_gprs_fee", "流量超套费用", value_type="number"),
        FieldSpec("run_code", "关停类型"),
        FieldSpec("rh_month_fee", "宽带资费费用", value_type="number"),
        FieldSpec("gprs_total", "套内流量资源", value_type="number"),
        FieldSpec("gprs_used_v", "流量使用量", 2, "number"),
        FieldSpec("voc_total", "套内语音资源", value_type="number"),
        FieldSpec("voc_used_v", "语音使用量", 2, "number"),
        FieldSpec("roaming_outside_province", "是否驻留省外", value_type="bool"),
    )
    common_vendor_aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "苹果": ("苹果", "APPLE"),
            "华为": ("华为", "HUAWEI"),
            "荣耀": ("荣耀", "HONOR"),
            "一加": ("一加", "ONEPLUS"),
            "小米": ("小米", "XIAOMI", "红米", "REDMI"),
            "OPPO": ("OPPO",),
            "vivo": ("VIVO",),
            "三星": ("三星", "SAMSUNG"),
            "中兴": ("中兴", "ZTE"),
        }
    )


DEFAULT_PROFILE_CONFIG = UserProfileConfig()
