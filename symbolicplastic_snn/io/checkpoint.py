from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import numpy as np

from .stable_snapshot import _store_to_arrays, _arrays_to_store


MAGIC = b"SPSNNCPT"


def _dt_meta(dt: np.dtype):
    d = np.dtype(dt)
    return d.descr if d.fields is not None else d.str


def save_checkpoint(
    path: str,
    v: np.ndarray,
    ref: np.ndarray,
    seeds: np.ndarray,
    alias: Dict[str, np.ndarray],
    stable,
    rng_state: Dict[str, Any],
) -> None:
    v = np.asarray(v, dtype=np.int16)
    ref = np.asarray(ref, dtype=np.uint8)
    seeds = np.asarray(seeds, dtype=np.uint64)
    prob = np.asarray(alias.get("prob"), dtype=np.uint16)
    alias_idx = np.asarray(alias.get("alias"), dtype=np.int32)

    # Convert stable store into arrays
    idx, entries = _store_to_arrays(stable)

    segments = [
        {"name": "v", "dtype": _dt_meta(v.dtype), "shape": list(v.shape)},
        {"name": "ref", "dtype": _dt_meta(ref.dtype), "shape": list(ref.shape)},
        {"name": "seeds", "dtype": _dt_meta(seeds.dtype), "shape": list(seeds.shape)},
        {"name": "alias_prob", "dtype": _dt_meta(prob.dtype), "shape": list(prob.shape)},
        {"name": "alias_alias", "dtype": _dt_meta(alias_idx.dtype), "shape": list(alias_idx.shape)},
        {"name": "stable_index", "dtype": _dt_meta(idx.dtype), "shape": list(idx.shape)},
        {"name": "stable_entries", "dtype": _dt_meta(entries.dtype), "shape": list(entries.shape)},
    ]
    meta = {"version": 1, "segments": segments, "rng_state": rng_state}
    meta_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    header = MAGIC + int.to_bytes(len(meta_json), 8, "little") + meta_json

    arrays = [v, ref, seeds, prob, alias_idx, idx, entries]
    sizes = [arr.nbytes for arr in arrays]
    total = len(header) + sum(sizes)
    with open(path, "wb") as f:
        f.truncate(total)
        f.seek(0)
        f.write(header)

    off = len(header)
    for arr in arrays:
        mm = np.memmap(path, dtype=arr.dtype, mode="r+", offset=off, shape=arr.shape)
        mm[...] = arr
        mm.flush()
        del mm
        off += arr.nbytes


def load_checkpoint(path: str):
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("Invalid checkpoint magic header")
        meta_len = int.from_bytes(f.read(8), "little")
        meta_json = f.read(meta_len)
    meta = json.loads(meta_json.decode("utf-8"))
    header_len = len(MAGIC) + 8 + len(meta_json)
    off = header_len
    loaded: Dict[str, np.ndarray] = {}
    for seg in meta["segments"]:
        name = seg["name"]
        dt_meta = seg["dtype"]
        if isinstance(dt_meta, list) and len(dt_meta) > 0 and isinstance(dt_meta[0], list):
            dt = np.dtype([tuple(x) for x in dt_meta])  # type: ignore
        else:
            dt = np.dtype(dt_meta)  # type: ignore
        shape = tuple(int(x) for x in seg["shape"])  # type: ignore
        mm = np.memmap(path, dtype=dt, mode="r", offset=off, shape=shape)
        loaded[name] = np.array(mm, copy=True)
        del mm
        off += loaded[name].nbytes
    alias = {"prob": loaded["alias_prob"], "alias": loaded["alias_alias"]}
    stable = _arrays_to_store(loaded["stable_index"], loaded["stable_entries"])
    return (
        loaded["v"],
        loaded["ref"],
        loaded["seeds"],
        alias,
        stable,
        meta.get("rng_state", {}),
    )


__all__ = ["save_checkpoint", "load_checkpoint"]
