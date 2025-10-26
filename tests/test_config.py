from __future__ import annotations

import os
import tempfile
import unittest

from config import config_from_dict, load_config


class ConfigTests(unittest.TestCase):
    def test_loads_yaml_sample(self):
        sample = """
time:
  dt_ms: 2
  tau_m_ms: 40
  refractory_ms: 4
topology:
  K_in: 64
  EI_ratio: 1.0
  sigma_dist: 1.0
  alpha_layer: 2.0
  long_range_ratio: 0.01
threshold:
  beta_sigma: 2.5
  est_pre_rate_hz: 7
plasticity:
  prune_interval: 600
  prune_quota: 0.12
  stdp_window: 25
  reward_modulated: true
stability_rules:
  merge_parallel_same_delay: false
  forbid_short_EE_loops: true
"""
        with tempfile.NamedTemporaryFile("w+", delete=False) as handle:
            handle.write(sample)
            handle.flush()
            path = handle.name

        config = load_config(path)
        os.unlink(path)
        self.assertEqual(config.time.dt_ms, 2)
        self.assertEqual(config.topology.K_in, 64)
        self.assertTrue(config.plasticity.reward_modulated)

    def test_config_from_dict_overrides(self):
        data = {"time": {"dt_ms": 2}, "topology": {"K_in": 32}}
        config = config_from_dict(data)
        self.assertEqual(config.time.dt_ms, 2)
        self.assertEqual(config.topology.K_in, 32)


if __name__ == "__main__":
    unittest.main()
