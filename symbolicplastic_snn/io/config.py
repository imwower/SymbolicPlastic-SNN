from __future__ import annotations

import json
from typing import Any, Dict, List


def _parse_scalar(val: str) -> Any:
    s = val.strip()
    if s == "":
        return ""
    low = s.lower()
    if low in ("true", "yes"):  # booleans
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none"):
        return None
    # Quoted string
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    # Try int
    try:
        if s.startswith("0x") or s.startswith("0X"):
            return int(s, 16)
        return int(s)
    except ValueError:
        pass
    # Try float
    try:
        return float(s)
    except ValueError:
        pass
    # Fallback to raw string
    return s


def load_yaml(path: str) -> Dict[str, Any]:
    """Load config from file, supporting JSON or a minimal YAML subset.

    YAML subset rules:
    - Mappings only (no sequences), using indentation (spaces) for nesting.
    - Each mapping line is `key: value` or `key:` to start a nested mapping.
    - Values can be numbers, booleans, null, or quoted/unquoted strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    # Try JSON first
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        pass

    # Minimal YAML mapping parser
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(0, root)]

    for raw in data.splitlines():
        line = raw.split("#", 1)[0].rstrip()  # strip comments
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError("Indentation must be multiples of 2 spaces")
        key_val = line.strip()

        # Adjust stack to current indent
        while stack and stack[-1][0] > indent:
            stack.pop()
        if not stack:
            raise ValueError("Invalid indentation structure")
        curr = stack[-1][1]

        if ":" not in key_val:
            raise ValueError("Expected 'key: value' line in YAML")

        key, val = key_val.split(":", 1)
        key = key.strip()
        if val.strip() == "":
            # Start nested mapping
            new_map: Dict[str, Any] = {}
            curr[key] = new_map
            stack.append((indent + 2, new_map))
        else:
            curr[key] = _parse_scalar(val)

    return root


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate minimal required sections and value ranges.

    Required sections/fields:
    - time: {dt_ms > 0, tau_m_ms > 0, refractory_ms >= 0}
    - topology: {K_in >= 1, EI_ratio in [0, 2], long_range_ratio in [0, 1]}
    - readout: {window_steps >= 1}
    Optional:
    - fixed_point: if present, refractory_steps >= 0, lambda_q15 is bool
    """
    def need(section: str, key: str) -> Any:
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Missing section: {section}")
        if key not in cfg[section]:
            raise ValueError(f"Missing field: {section}.{key}")
        return cfg[section][key]

    # time
    dt = need("time", "dt_ms")
    tau = need("time", "tau_m_ms")
    ref = need("time", "refractory_ms")
    if not (isinstance(dt, (int, float)) and dt > 0):
        raise ValueError("time.dt_ms must be > 0")
    if not (isinstance(tau, (int, float)) and tau > 0):
        raise ValueError("time.tau_m_ms must be > 0")
    if not (isinstance(ref, (int, float)) and ref >= 0):
        raise ValueError("time.refractory_ms must be >= 0")

    # topology
    kin = need("topology", "K_in")
    ei = need("topology", "EI_ratio")
    lr = need("topology", "long_range_ratio")
    if not (isinstance(kin, (int, float)) and kin >= 1):
        raise ValueError("topology.K_in must be >= 1")
    if not (isinstance(ei, (int, float)) and 0.0 <= float(ei) <= 2.0):
        raise ValueError("topology.EI_ratio must be in [0, 2]")
    if not (isinstance(lr, (int, float)) and 0.0 <= float(lr) <= 1.0):
        raise ValueError("topology.long_range_ratio must be in [0, 1]")

    # readout
    ws = need("readout", "window_steps")
    if not (isinstance(ws, (int, float)) and ws >= 1):
        raise ValueError("readout.window_steps must be >= 1")

    # fixed_point (optional)
    if "fixed_point" in cfg and isinstance(cfg["fixed_point"], dict):
        fp = cfg["fixed_point"]
        if "refractory_steps" in fp and not (isinstance(fp["refractory_steps"], (int, float)) and fp["refractory_steps"] >= 0):
            raise ValueError("fixed_point.refractory_steps must be >= 0")
        if "lambda_q15" in fp and not isinstance(fp["lambda_q15"], (bool, int)):
            raise ValueError("fixed_point.lambda_q15 must be boolean")


__all__ = ["load_yaml", "validate_config"]

