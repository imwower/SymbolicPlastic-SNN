from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.topology.metrics import (
    degree_hist,
    inout_invariants,
    small_worldness,
    avg_path_len_approx,
    clustering_coeff_approx,
)
from symbolicplastic_snn.core.prng import SeedSpace


def ring_lattice(n: int, k: int) -> np.ndarray:
    """Return NxN directed ring lattice: each node connects to k/2 neighbors on each side.
    Ensures constant out-degree k.
    """
    A = np.zeros((n, n), dtype=np.uint8)
    r = k // 2
    for i in range(n):
        for d in range(1, r + 1):
            A[i, (i + d) % n] = 1
            A[i, (i - d) % n] = 1
    np.fill_diagonal(A, 0)
    return A


def rewire(A: np.ndarray, p: float, seed: int) -> np.ndarray:
    """Deterministic Watts-Strogatz style rewiring using SeedSpace.Stream.
    For each edge i->j, with probability p, rewire to i->k uniformly (k != i).
    """
    n = A.shape[0]
    ss = SeedSpace(seed)
    st = ss.derive("module=rewire", f"n={n}")
    B = A.copy()
    for i in range(n):
        js = np.nonzero(A[i])[0]
        for j in js:
            if st.uniform() < p:
                B[i, j] = 0
                # pick a new target != i
                k = int(st.randbelow(n - 1))
                if k >= i:
                    k += 1
                B[i, k] = 1
    np.fill_diagonal(B, 0)
    return B


class TestTopologyMetrics(unittest.TestCase):
    def test_small_world_invariants(self):
        n = 64
        k = 8
        A0 = ring_lattice(n, k)
        # Invariants on degrees
        in_hist, out_hist = degree_hist(A0)
        inv = inout_invariants(A0)
        self.assertAlmostEqual(inv["mean_out"], k, delta=1e-6)
        # Rewire to get small-world characteristics
        A = rewire(A0, p=0.1, seed=2025)
        sigma = small_worldness(A, samples=32)
        self.assertGreater(sigma, 1.0)  # weak but robust assertion
        # Ranges (very loose due to approximation)
        L = avg_path_len_approx(A, k=16)
        C = clustering_coeff_approx(A, k=16)
        self.assertTrue(1.0 <= L <= 10.0)
        self.assertTrue(0.0 <= C <= 1.0)


if __name__ == "__main__":
    unittest.main()

