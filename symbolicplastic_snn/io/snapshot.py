from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import numpy as np


MAGIC = b"SPSNNSNP"


def _dtype_str(dt: np.dtype) -> str:
    # Preserve endianness and itemsize, e.g., '<i2', '|u1'
    return np.dtype(dt).str


def _nbytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


def save_snapshot(
    path: str,
    v: np.ndarray,
    ref: np.ndarray,
    seeds: np.ndarray,
    alias: Dict[str, np.ndarray],
    rng_state: Dict[str, Any],
) -> None:
    """Save simulation state to a single binary file using numpy.memmap.

    Arrays are written sequentially after a small JSON metadata header.
    """
    v = np.asarray(v, dtype=np.int16)
    ref = np.asarray(ref, dtype=np.uint8)
    seeds = np.asarray(seeds, dtype=np.uint64)
    prob = np.asarray(alias.get("prob"), dtype=np.uint16)
    alias_idx = np.asarray(alias.get("alias"), dtype=np.int32)

    segments = [
        {"name": "v", "dtype": _dtype_str(v.dtype), "shape": list(v.shape)},
        {"name": "ref", "dtype": _dtype_str(ref.dtype), "shape": list(ref.shape)},
        {"name": "seeds", "dtype": _dtype_str(seeds.dtype), "shape": list(seeds.shape)},
        {"name": "alias_prob", "dtype": _dtype_str(prob.dtype), "shape": list(prob.shape)},
        {"name": "alias_alias", "dtype": _dtype_str(alias_idx.dtype), "shape": list(alias_idx.shape)},
    ]

    # Build metadata without offsets first
    meta = {
        "version": 1,
        "segments": segments,
        "rng_state": rng_state,
    }
    meta_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    header = MAGIC + int.to_bytes(len(meta_json), 8, "little") + meta_json

    # Compute sizes
    sizes = [
        _nbytes(tuple(v.shape), v.dtype),
        _nbytes(tuple(ref.shape), ref.dtype),
        _nbytes(tuple(seeds.shape), seeds.dtype),
        _nbytes(tuple(prob.shape), prob.dtype),
        _nbytes(tuple(alias_idx.shape), alias_idx.dtype),
    ]

    # Finalize header without offsets (offsets are derived during read)
    meta_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    header = MAGIC + int.to_bytes(len(meta_json), 8, "little") + meta_json

    # Prepare file of the correct size
    total_size = len(header) + sum(sizes)
    with open(path, "wb") as f:
        f.truncate(total_size)
        f.seek(0)
        f.write(header)

    # Write arrays via memmap at each offset
    seg_arrays = [v, ref, seeds, prob, alias_idx]
    # Offsets start right after header; write sequentially
    off = len(header)
    for seg, arr, sz in zip(segments, seg_arrays, sizes):
        mm = np.memmap(path, dtype=np.dtype(seg["dtype"]), mode="r+", offset=off, shape=tuple(arr.shape))
        mm[...] = arr
        mm.flush()
        del mm
        off += sz


def load_snapshot(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """Load state from a snapshot created by save_snapshot.

    Returns (v:int16, ref:uint8, seeds:uint64, alias:dict, rng_state:dict)
    """
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("Invalid snapshot magic header")
        meta_len = int.from_bytes(f.read(8), "little")
        meta_json = f.read(meta_len)
    meta = json.loads(meta_json.decode("utf-8"))

    segs = meta["segments"]
    loaded: Dict[str, np.ndarray] = {}
    # Data starts after header
    header_len = len(MAGIC) + 8 + len(json.dumps(meta, separators=(",", ":")).encode("utf-8"))
    off = header_len
    for seg in segs:
        name = seg["name"]
        dt = np.dtype(seg["dtype"])
        shape = tuple(int(x) for x in seg["shape"])  # type: ignore[assignment]
        mm = np.memmap(path, dtype=dt, mode="r", offset=off, shape=shape)
        loaded[name] = np.array(mm, copy=True)  # detach from file but preserve dtype/shape
        del mm
        off += _nbytes(shape, dt)

    alias = {
        "prob": loaded["alias_prob"],
        "alias": loaded["alias_alias"],
    }
    return loaded["v"], loaded["ref"], loaded["seeds"], alias, meta.get("rng_state", {})


__all__ = ["save_snapshot", "load_snapshot"]
