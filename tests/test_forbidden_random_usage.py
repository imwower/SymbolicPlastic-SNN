from __future__ import annotations

import os
import unittest


class TestForbiddenRandomUsage(unittest.TestCase):
    def test_forbidden_random_usage_in_package(self):
        base = os.path.join(os.getcwd(), "symbolicplastic_snn")
        bad = []
        for root, _, files in os.walk(base):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(root, fn)
                # prng implementation is allowed
                if path.endswith(os.path.join("utils", "prng.py")):
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        text = f.read()
                    except Exception:
                        continue
                if "numpy.random" in text or "import random" in text:
                    bad.append(path)
        self.assertFalse(bad, f"Forbidden random usage found in: {bad}")


if __name__ == "__main__":
    unittest.main()
