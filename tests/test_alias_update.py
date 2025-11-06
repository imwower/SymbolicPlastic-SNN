from __future__ import annotations

import unittest
import numpy as np

from symbolicplastic_snn.conn.alias import build_alias, rebuild_or_update
from symbolicplastic_snn.core.prng import SeedSpace


class TestAliasIncremental(unittest.TestCase):
    def test_incremental_matches_full(self):
        n = 512
        # Construct a deterministic weight vector in Q0.16
        w = np.array([(i * 17) % 65535 for i in range(n)], dtype=np.uint16)
        p_full, a_full = build_alias(w)

        # Modify 1% indices
        changed = np.arange(0, n, 100, dtype=np.int64)
        w2 = w.copy()
        w2[changed] = (w2[changed].astype(np.uint32) ^ 12345).astype(np.uint16)

        p_ref, a_ref = build_alias(w2)
        # Provide the updated weights (w2) to incrementally rebuild
        p_upd, a_upd = rebuild_or_update(w2, a_full, changed)
        # Our conservative implementation rebuilds fully on changes
        # so p_upd,a_upd should equal p_ref,a_ref
        self.assertTrue(np.array_equal(p_upd, p_ref))
        self.assertTrue(np.array_equal(a_upd, a_ref))

        # Deterministic sampling agreement for first 128 draws using a fixed SeedSpace
        ss = SeedSpace(0xBADC0FFEE)
        s = ss.derive("module=alias", "test=1")
        draws_ref = []
        draws_upd = []
        for _ in range(128):
            # emulate sampling: choose column via randbelow(n) then decide by threshold
            col = int(s.randbelow(n))
            u = int(s.randbelow(65536))
            draws_ref.append(col if u < int(p_ref[col]) else int(a_ref[col]))
            draws_upd.append(col if u < int(p_upd[col]) else int(a_upd[col]))
        self.assertEqual(draws_ref, draws_upd)


if __name__ == "__main__":
    unittest.main()
