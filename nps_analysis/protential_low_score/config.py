from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .engine import OptimizationConfig, Rule


def load_config(
    path: str | Path | None = None,
) -> tuple[dict[str, Any], OptimizationConfig, list[Rule]]:
    config_path = Path(path) if path else Path(__file__).with_name("rules.yaml")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    optimization_data = data.get("optimization", {})
    optimization = OptimizationConfig(
        random_seed=int(optimization_data.get("random_seed", 20260907)),
        selection_mode=optimization_data.get("selection_mode", "balanced"),
        beta=float(optimization_data.get("beta", 1.0)),
        threshold_candidates=tuple(
            optimization_data.get("threshold_candidates", [1, 2, 3, 4, 5])
        ),
    )
    rules = [
        Rule(
            rule_id=item["rule_id"],
            risk_type=item["risk_type"],
            condition=item["condition"],
            weight_candidates=tuple(item.get("weight_candidates", [1])),
            reason=item["reason"],
            enabled=bool(item.get("enabled", True)),
        )
        for item in data.get("rules", [])
    ]
    return data.get("windows", {}), optimization, rules
