from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

try:  # optional dependency
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class TimePreset:
    dt_ms: float = 1.0
    tau_m_ms: float = 50.0
    refractory_ms: float = 3.0


@dataclass(frozen=True)
class TopologyPreset:
    K_in: int = 128
    EI_ratio: float = 1.0
    sigma_dist: float = 1.0
    alpha_layer: float = 2.0
    long_range_ratio: float = 0.01


@dataclass(frozen=True)
class ThresholdPreset:
    beta_sigma: float = 2.0
    est_pre_rate_hz: float = 5.0


@dataclass(frozen=True)
class PlasticityPreset:
    prune_interval: int = 500
    prune_quota: float = 0.10
    stdp_window: int = 20
    reward_modulated: bool = False


@dataclass(frozen=True)
class StabilityRules:
    merge_parallel_same_delay: bool = True
    forbid_short_EE_loops: bool = True


@dataclass(frozen=True)
class SystemConfig:
    time: TimePreset
    topology: TopologyPreset
    threshold: ThresholdPreset
    plasticity: PlasticityPreset
    stability: StabilityRules


def default_config() -> SystemConfig:
    return SystemConfig(
        time=TimePreset(),
        topology=TopologyPreset(),
        threshold=ThresholdPreset(),
        plasticity=PlasticityPreset(),
        stability=StabilityRules(),
    )


def load_config(path: str | Path) -> SystemConfig:
    text = Path(path).read_text(encoding="utf-8")
    data = _parse_text(text)
    return config_from_dict(data)


def config_from_dict(data: Mapping[str, Any]) -> SystemConfig:
    defaults = default_config()
    return SystemConfig(
        time=_merge(TimePreset, defaults.time, data.get("time", {})),
        topology=_merge(TopologyPreset, defaults.topology, data.get("topology", {})),
        threshold=_merge(ThresholdPreset, defaults.threshold, data.get("threshold", {})),
        plasticity=_merge(PlasticityPreset, defaults.plasticity, data.get("plasticity", {})),
        stability=_merge(StabilityRules, defaults.stability, data.get("stability_rules", data.get("stability", {}))),
    )


def _merge(cls, defaults, overrides):
    values = {field: getattr(defaults, field) for field in defaults.__dataclass_fields__}
    for key, value in overrides.items():
        if key in values:
            values[key] = value
    return cls(**values)


def _parse_text(text: str) -> Dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _mini_yaml_parse(text)


def _mini_yaml_parse(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[tuple[Dict[str, Any], int]] = [(root, -1)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"Invalid line: {raw_line}")
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][1]:
            stack.pop()
        parent = stack[-1][0]
        if not value:
            new_dict: Dict[str, Any] = {}
            parent[key] = new_dict
            stack.append((new_dict, indent))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"\'')
