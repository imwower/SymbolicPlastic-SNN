from __future__ import annotations

import unittest

from monitor import StabilityMonitor


class MonitorTests(unittest.TestCase):
    def test_metrics_tracking(self):
        monitor = StabilityMonitor(num_neurons=4, window=3, dt_ms=1.0)
        monitor.record_step([1, 0, 0, 0], [1, 2])
        monitor.record_step([0, 1, 0, 0], [2])
        monitor.record_step([1, 1, 1, 1], [])

        metrics = monitor.metrics()
        self.assertAlmostEqual(metrics.branching_factor, 1.0, places=2)
        self.assertGreater(metrics.average_rate_hz, 0)
        self.assertGreater(metrics.synchrony_index, 0)


if __name__ == "__main__":
    unittest.main()
