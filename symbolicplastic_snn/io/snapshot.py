from __future__ import annotations

import json
import struct
from typing import Any, Dict, Tuple

import numpy as np


MAGIC = b"SPSNNSNP"  # 8 bytes
SNAP_VERSION = 2  # bump to 2 for versioned header with endianness

# Endianness flag encoding for header (data arrays are written in native endian)
# 0 = little-endian host, 1 = big-endian host
_HOST_ENDIAN_FLAG = 0 if np.little_endian else 1


def _dtype_str(dt: np.dtype) -> str:
    # Preserve endianness and itemsize, e.g., '<i2', '|u1'
    return np.dtype(dt).str


def _nbytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


def _pack_header_v2(meta_json: bytes) -> bytes:
    """Build a V2 header: MAGIC + version(u16) + endian(u8) + reserved(5B) + meta_len(u64) + meta_json."""
    version = SNAP_VERSION
    endian = _HOST_ENDIAN_FLAG
    reserved = b"\x00" * 5
    meta_len = len(meta_json)
    # Header fixed fields are stored in little-endian for portability
    fixed = MAGIC + struct.pack("<H B 5s Q", version, endian, reserved, meta_len)
    return fixed + meta_json


def _try_parse_header(f) -> tuple[int, int, bytes]:
    """Read header prefix and return (version, endian_flag, meta_json).

    Supports both V1 (legacy) and V2 (versioned) headers.
    """
    # Read MAGIC
    magic = f.read(len(MAGIC))
    if magic != MAGIC:
        raise ValueError("Invalid snapshot magic header")

    # Peek next 2+1+5+8 bytes to detect V2; if not plausible, treat as V1
    peek = f.read(2 + 1 + 5 + 8)
    if len(peek) < (2 + 1 + 5 + 8):
        raise ValueError("Corrupt snapshot header (truncated)")
    try:
        ver, endian_flag, _reserved, meta_len = struct.unpack("<H B 5s Q", peek)
        # V2 plausibility: small integer version and endian_flag in {0,1}
        if 1 <= ver <= 0x00FF and endian_flag in (0, 1):
            meta_json = f.read(meta_len)
            if len(meta_json) != meta_len:
                raise ValueError("Corrupt snapshot: meta length mismatch")
            return ver, endian_flag, meta_json
        # Else fall through to V1 path
    except Exception:
        # Fall back to V1 path
        pass

    # Legacy V1 header: after MAGIC comes meta_len (u64 little-endian) + meta_json
    # Re-interpret `peek` bytes as beginning of meta_len
    meta_len = struct.unpack("<Q", peek[:8])[0]
    meta_json = f.read(meta_len)
    if len(meta_json) != meta_len:
        raise ValueError("Corrupt snapshot (legacy): meta length mismatch")
    return 1, _HOST_ENDIAN_FLAG, meta_json


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
        "version": SNAP_VERSION,
        "segments": segments,
        "rng_state": rng_state,
    }
    # Include strides metadata for validation/debugging
    for seg in segments:
        name = seg["name"]
        if name == "v":
            seg["strides"] = list(v.strides)
        elif name == "ref":
            seg["strides"] = list(ref.strides)
        elif name == "seeds":
            seg["strides"] = list(seeds.strides)
        elif name == "alias_prob":
            seg["strides"] = list(prob.strides)
        elif name == "alias_alias":
            seg["strides"] = list(alias_idx.strides)

    meta_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    header = _pack_header_v2(meta_json)

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
    header = _pack_header_v2(meta_json)

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
        ver, endian_flag, meta_json = _try_parse_header(f)
    if endian_flag != _HOST_ENDIAN_FLAG:
        raise ValueError(f"Snapshot endianness mismatch: file={endian_flag} host={_HOST_ENDIAN_FLAG}")
    meta = json.loads(meta_json.decode("utf-8"))

    segs = meta["segments"]
    loaded: Dict[str, np.ndarray] = {}
    # Data starts after header
    # Compute header length depending on version
    if int(meta.get("version", 1)) >= 2:
        header_len = len(MAGIC) + 2 + 1 + 5 + 8 + len(meta_json)
    else:
        header_len = len(MAGIC) + 8 + len(meta_json)
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


def save_runner_snapshot(path: str, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
    """Save a generic runner snapshot with arbitrary named arrays and JSON metadata.

    - arrays: mapping name -> numpy array (dtype/shape preserved)
    - meta: free-form JSON-serializable dict (config, counters, etc.)
    """
    # Build segments description first without offsets
    segments = [
        {"name": name, "dtype": _dtype_str(arr.dtype), "shape": list(arr.shape)}
        for name, arr in arrays.items()
    ]
    # Enrich segments with strides info for validation
    seg_meta = []
    for name, arr in arrays.items():
        seg_meta.append({
            "name": name,
            "dtype": _dtype_str(arr.dtype),
            "shape": list(arr.shape),
            "strides": list(arr.strides),
        })
    hdr = {"version": SNAP_VERSION, "segments": seg_meta, "meta": meta, "endian": _HOST_ENDIAN_FLAG}
    meta_json = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    header = _pack_header_v2(meta_json)

    # Compute sizes and offsets sequentially
    sizes = [_nbytes(tuple(arr.shape), arr.dtype) for arr in arrays.values()]
    offset = len(header)
    offsets = []
    for sz in sizes:
        offsets.append(offset)
        offset += sz

    # Write full header (offsets derivable, not embedded)
    with open(path, "wb") as f:
        f.truncate(len(header) + sum(sizes))
        f.seek(0)
        f.write(header)

    # Write arrays via memmap
    for (name, arr), off in zip(arrays.items(), offsets):
        mm = np.memmap(path, dtype=arr.dtype, mode="r+", offset=off, shape=arr.shape)
        mm[...] = arr
        mm.flush()
        del mm


def load_runner_snapshot(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load arrays and metadata from a runner snapshot created by save_runner_snapshot."""
    with open(path, "rb") as f:
        ver, endian_flag, meta_json = _try_parse_header(f)
    if endian_flag != _HOST_ENDIAN_FLAG:
        raise ValueError(f"Snapshot endianness mismatch: file={endian_flag} host={_HOST_ENDIAN_FLAG}")
    hdr = json.loads(meta_json.decode("utf-8"))
    segs = hdr["segments"]
    arrays: Dict[str, np.ndarray] = {}
    # Offsets derive from header length + previous sizes
    if int(hdr.get("version", 1)) >= 2:
        header_len = len(MAGIC) + 2 + 1 + 5 + 8 + len(meta_json)
    else:
        header_len = len(MAGIC) + 8 + len(meta_json)
    off = header_len
    for seg in segs:
        name = str(seg["name"])
        dt = np.dtype(seg["dtype"])  # type: ignore
        shape = tuple(int(x) for x in seg["shape"])  # type: ignore
        mm = np.memmap(path, dtype=dt, mode="r", offset=off, shape=shape)
        arrays[name] = np.array(mm, copy=True)
        del mm
        off += _nbytes(shape, dt)
    return arrays, hdr.get("meta", {})

__all__.extend(["save_runner_snapshot", "load_runner_snapshot"])
