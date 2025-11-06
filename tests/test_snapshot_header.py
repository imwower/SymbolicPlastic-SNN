from __future__ import annotations

import io
import json
import os
import struct
import tempfile
import unittest
import numpy as np

from symbolicplastic_snn.io.snapshot import (
    MAGIC,
    save_runner_snapshot,
    load_runner_snapshot,
)


class TestSnapshotHeader(unittest.TestCase):
    def test_header_version_endianness_and_strides(self):
        arrays = {
            "a": np.arange(12, dtype=np.int16).reshape(3, 4),
            "b": (np.arange(8, dtype=np.uint8)),
        }
        meta = {"msg": "hello"}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "snap.bin")
            save_runner_snapshot(path, arrays, meta)

            # Read raw header fields
            with open(path, "rb") as f:
                magic = f.read(len(MAGIC))
                self.assertEqual(magic, MAGIC)
                # version(u16), endian(u8), reserved(5), meta_len(u64)
                ver, endian, _res, mlen = struct.unpack("<H B 5s Q", f.read(2 + 1 + 5 + 8))
                self.assertGreaterEqual(ver, 2)
                self.assertIn(endian, (0, 1))
                meta_json = f.read(mlen)
            hdr = json.loads(meta_json.decode("utf-8"))
            self.assertIn("segments", hdr)
            # Strides are present and match numpy's
            segs = {s["name"]: s for s in hdr["segments"]}
            self.assertEqual(segs["a"]["strides"], list(arrays["a"].strides))
            self.assertEqual(segs["b"]["strides"], list(arrays["b"].strides))

            # Normal load should succeed and restore arrays
            arr2, meta2 = load_runner_snapshot(path)
            self.assertTrue(np.array_equal(arr2["a"], arrays["a"]))
            self.assertTrue(np.array_equal(arr2["b"], arrays["b"]))
            self.assertEqual(meta2.get("msg"), "hello")

    def test_incompatible_version_rejected(self):
        arrays = {"x": np.arange(4, dtype=np.int16)}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "snap.bin")
            save_runner_snapshot(path, arrays, {"v": 1})
            # Corrupt the version to a large unsupported value
            with open(path, "rb") as f:
                hdr = f.read()
            # MAGIC + (u16,u8,5B,u64) + meta json
            # Overwrite u16 version at offset len(MAGIC)
            data = bytearray(hdr)
            off = len(MAGIC)
            struct.pack_into("<H", data, off, 0xFFFF)
            with open(path, "wb") as f:
                f.write(data)
            with self.assertRaisesRegex(ValueError, "endianness mismatch|Invalid|Corrupt"):
                load_runner_snapshot(path)

    def test_endianness_mismatch(self):
        arrays = {"x": np.arange(4, dtype=np.int16)}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "snap.bin")
            save_runner_snapshot(path, arrays, {"v": 1})
            # Toggle endian flag byte
            with open(path, "rb") as f:
                hdr = f.read()
            data = bytearray(hdr)
            off = len(MAGIC) + 2  # after version u16
            data[off] = 1 if data[off] == 0 else 0
            with open(path, "wb") as f:
                f.write(data)
            with self.assertRaisesRegex(ValueError, "endianness mismatch"):
                load_runner_snapshot(path)


if __name__ == "__main__":
    unittest.main()

