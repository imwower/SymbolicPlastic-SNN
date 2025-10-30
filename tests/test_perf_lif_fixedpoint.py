import time

import numpy as np

from symbolicplastic_snn.core.lif_fixedpoint import ConfigFp, lif_step


def test_lif_step_performance_1e6_under_50ms():
    """Micro-benchmark: 1e6 elements should run < 50 ms per step.

    Notes
    - This benchmark uses vectorized NumPy on CPU. It warms up the kernel,
      then measures average time over several repeats to reduce noise.
    - Threshold is aligned with the acceptance criteria given for local CPU.
    """

    n = 1_000_000
    v = np.zeros(n, dtype=np.int16)
    ref = np.zeros(n, dtype=np.uint8)
    I = np.zeros(n, dtype=np.int32)
    lam = np.uint16(32000)  # ~0.98 in Q1.15
    theta = np.int16(30000)
    cfg = ConfigFp(refractory_steps=3)

    # Warm-up
    for _ in range(3):
        lif_step(v, ref, I, theta, lam, cfg)

    repeats = 10
    t0 = time.perf_counter()
    for _ in range(repeats):
        lif_step(v, ref, I, theta, lam, cfg)
    t1 = time.perf_counter()
    per_step_ms = (t1 - t0) * 1000.0 / repeats

    assert per_step_ms < 50.0, f"Avg per step {per_step_ms:.3f} ms >= 50 ms"

