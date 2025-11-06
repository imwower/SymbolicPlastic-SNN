from __future__ import annotations

import json
import unittest
from pathlib import Path

from symbolicplastic_snn.core.prng import SeedSpace


class TestPrngGoldenJson(unittest.TestCase):
    def test_golden_vectors_match(self):
        p = Path("docs/prng_golden.json")
        self.assertTrue(p.exists(), "missing docs/prng_golden.json; run scripts/gen_prng_golden.py")
        data = json.loads(p.read_text())
        for ent in data.get("entries", []):
            seed = int(ent["run_seed"])
            keys = [str(k) for k in ent["keys"]]
            ss = SeedSpace(seed)
            st = ss.derive(*keys)
            got_u64 = [int(st.u64()) for _ in range(8)]
            self.assertEqual(got_u64, [int(x) for x in ent["u64"]])
            st2 = ss.derive(*keys)
            got_uni = [float(st2.uniform()) for _ in range(4)]
            self.assertEqual([round(x, 16) for x in got_uni], [round(float(y), 16) for y in ent["uniform"]])
            st3 = ss.derive(*keys)
            got_perm = [int(x) for x in st3.permutation(16).tolist()]
            self.assertEqual(got_perm, [int(z) for z in ent["perm16"]])


if __name__ == "__main__":
    unittest.main()

