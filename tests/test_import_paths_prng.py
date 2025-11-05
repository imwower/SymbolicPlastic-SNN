from __future__ import annotations

import unittest


class TestPrngImportPaths(unittest.TestCase):
    def test_reexport_identity_and_golden(self):
        # Import from both paths
        from symbolicplastic_snn.core.prng import SeedSpace as SeedSpaceCore, Stream as StreamCore
        from symbolicplastic_snn.utils.prng import SeedSpace as SeedSpaceUtils, Stream as StreamUtils

        # Identity of symbols (re-export)
        self.assertIs(SeedSpaceCore, SeedSpaceUtils)
        self.assertIs(StreamCore, StreamUtils)

        # Golden vector check: fixed run_seed + keys -> first 5 u64 and uniform values
        ss = SeedSpaceCore(0x0123456789ABCDEF)
        st = ss.derive("module=noise", "layer=3", "conn=pre42->post7")
        got_u64 = [st.u64() for _ in range(5)]
        st2 = SeedSpaceCore(0x0123456789ABCDEF).derive("module=noise", "layer=3", "conn=pre42->post7")
        got_uniform = [st2.uniform() for _ in range(5)]

        exp_hex = [
            0x89C7E3D87EA26747,
            0x57ACC3A9513BBF4C,
            0x3A662726F3CB96F5,
            0xAF521F1A3D20D4A0,
            0x91F2E9C8664BBE49,
        ]
        exp_uniform = [
            0.53820632968439375,
            0.34247992404674121,
            0.22812123013481611,
            0.68484682455630552,
            0.57011281149452386,
        ]
        self.assertEqual([int(x) for x in got_u64], exp_hex)
        # Compare floats with exact repr match via formatting to 17 digits
        self.assertEqual([f"{x:.17g}" for x in got_uniform], [f"{x:.17g}" for x in exp_uniform])


if __name__ == "__main__":
    unittest.main()
