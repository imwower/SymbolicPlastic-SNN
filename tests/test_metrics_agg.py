from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np

from monitor.metrics import Metrics
from symbolicplastic_snn.io.snapshot import save_runner_snapshot, load_runner_snapshot


class TestMetricsAggregator(unittest.TestCase):
    def test_basic_merge_and_reset(self):
        a = Metrics()
        b = Metrics()
        a.steps = 10
        a.events_generated = 100
        a.add_winner("C0")
        a.add_winner("C0")
        b.steps = 5
        b.events_generated = 50
        b.add_winner("C1")
        a.merge(b)
        d = a.to_dict()
        self.assertEqual(d["steps"], 15)
        self.assertEqual(d["events_generated"], 150)
        self.assertEqual(d["winners"], {"C0": 2, "C1": 1})
        a.reset()
        self.assertEqual(a.to_dict()["steps"], 0)

    def test_snapshot_roundtrip_meta(self):
        m = Metrics()
        m.steps = 7
        m.events_processed = 77
        m.add_winner("C0")
        meta = {"monitor_metrics": m.to_dict()}
        arrays = {"arr": np.arange(4, dtype=np.int16)}
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "snap.bin")
            save_runner_snapshot(p, arrays, meta)
            _, meta2 = load_runner_snapshot(p)
            self.assertIn("monitor_metrics", meta2)
            m2 = Metrics.from_dict(meta2["monitor_metrics"])  # type: ignore
            self.assertEqual(m2.to_dict()["winners"], {"C0": 1})


if __name__ == "__main__":
    unittest.main()

