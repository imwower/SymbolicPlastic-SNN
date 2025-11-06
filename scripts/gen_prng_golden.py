from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from symbolicplastic_snn.core.prng import SeedSpace


def gen_vectors() -> Dict[str, Any]:
    """Generate cross-language golden vectors for PRNG components.

    Contents:
    - entries: list of dicts with (run_seed, keys[]) and outputs:
      * u64[:8]
      * uniform[:4]
      * permutation(16)
    """
    seeds = [0x0123456789ABCDEF, 0xDEADBEEFCAFEBABE, 0x1234]
    key_sets = [
        ["module=noise", "layer=0"],
        ["module=topology", "pre=42", "post=7"],
        ["module=encode", "trial=3"],
    ]
    entries: List[Dict[str, Any]] = []
    for run_seed in seeds:
        for keys in key_sets:
            ss = SeedSpace(run_seed)
            st = ss.derive(*keys)
            u64 = [int(st.u64()) for _ in range(8)]
            st2 = ss.derive(*keys)
            uni = [float(st2.uniform()) for _ in range(4)]
            st3 = ss.derive(*keys)
            perm = [int(x) for x in st3.permutation(16).tolist()]
            entries.append({
                "run_seed": int(run_seed),
                "keys": [str(k) for k in keys],
                "u64": u64,
                "uniform": uni,
                "perm16": perm,
            })
    meta = {
        "algorithm": "xorshift64* + SplitMix64 seeding; FNV-1a hashing",
        "references": [
            "Vigna (2014): Further scramblings of Marsaglia's xorshift generators",
            "SplitMix64 (Steele et al.)",
            "FNV-1a 64-bit"
        ],
        "endianness": "host",  # independent for integer sequences
        "dtype": {
            "u64": "uint64",
            "uniform": "float64",
            "perm16": "int64",
        },
    }
    return {"meta": meta, "entries": entries}


def main() -> None:
    data = gen_vectors()
    out = Path("docs/prng_golden.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(str(out))


if __name__ == "__main__":
    main()

