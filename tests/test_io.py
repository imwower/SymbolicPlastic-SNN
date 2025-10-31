import os
import json
import tempfile

import numpy as np

from symbolicplastic_snn.io.config import load_yaml, validate_config
from symbolicplastic_snn.io.snapshot import save_snapshot, load_snapshot


def test_snapshot_roundtrip():
    n = 64
    # Deterministic sequences (no external RNG)
    v = (((np.arange(n, dtype=np.int32) * 12345) % 60001) - 30000).astype(np.int16)
    ref = (np.arange(n, dtype=np.int32) % 10).astype(np.uint8)
    seeds = (np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
    alias_prob = (np.arange(16, dtype=np.uint16) * np.uint16(4096)).astype(np.uint16)
    alias_alias = (np.arange(16, dtype=np.int32) % 16).astype(np.int32)
    alias = {"prob": alias_prob, "alias": alias_alias}
    rng_state = {"algo": "splitmix64", "seed": 12345, "step": 678}

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "snap.bin")
        save_snapshot(path, v, ref, seeds, alias, rng_state)
        v2, ref2, seeds2, alias2, rng2 = load_snapshot(path)

    assert v2.dtype == np.int16 and ref2.dtype == np.uint8 and seeds2.dtype == np.uint64
    assert alias2["prob"].dtype == np.uint16 and alias2["alias"].dtype == np.int32
    assert np.array_equal(v2, v)
    assert np.array_equal(ref2, ref)
    assert np.array_equal(seeds2, seeds)
    assert np.array_equal(alias2["prob"], alias_prob)
    assert np.array_equal(alias2["alias"], alias_alias)
    assert rng2 == rng_state


def test_config_validate(tmp_path):
    # Minimal valid config in YAML subset
    yaml_text = (
        "time:\n"
        "  dt_ms: 1\n"
        "  tau_m_ms: 50\n"
        "  refractory_ms: 3\n"
        "topology:\n"
        "  K_in: 128\n"
        "  EI_ratio: 1.0\n"
        "  long_range_ratio: 0.1\n"
        "readout:\n"
        "  window_steps: 200\n"
    )
    p = tmp_path / "cfg.yml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_yaml(str(p))
    validate_config(cfg)  # should not raise

    # Missing sections/fields
    bad_yaml = (
        "time:\n"
        "  dt_ms: 0\n"  # invalid
        "topology:\n"
        "  K_in: 0\n"  # invalid
        "  EI_ratio: -1\n"
        "  long_range_ratio: 1.2\n"
        "readout:\n"
        "  window_steps: 0\n"
    )
    q = tmp_path / "bad.yml"
    q.write_text(bad_yaml, encoding="utf-8")
    bad = load_yaml(str(q))
    raised = False
    try:
        validate_config(bad)
    except ValueError:
        raised = True
    assert raised

    # Also accept JSON
    j = tmp_path / "cfg.json"
    j.write_text(
        json.dumps(
            {
                "time": {"dt_ms": 1, "tau_m_ms": 30, "refractory_ms": 2},
                "topology": {"K_in": 64, "EI_ratio": 1.0, "long_range_ratio": 0.0},
                "readout": {"window_steps": 50},
            }
        ),
        encoding="utf-8",
    )
    cfg2 = load_yaml(str(j))
    validate_config(cfg2)
