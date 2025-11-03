from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import numpy as np

from symbolicplastic_snn.plasticity.stable_store import StableStore, StableEdge


MAGIC = b"SPSNNSTB"


_ENTRY_DTYPE = np.dtype(
    [
        ("pre_id", "<i4"),
        ("post_id", "<i4"),
        ("sign", "i1"),
        ("delay", "u1"),
        ("state", "u1"),
        ("age", "<u2"),
        ("corr", "<i2"),
        ("last_used", "<i4"),
    ]
)


def _store_to_arrays(store: StableStore) -> Tuple[np.ndarray, np.ndarray]:
    # Build index (pre_id, start, count) and entries arrays
    pairs = []
    for e in store.iter_all():
        pairs.append((int(e.pre_id), e))
    if not pairs:
        idx = np.zeros((0, 3), dtype=np.int32)
        entries = np.zeros((0,), dtype=_ENTRY_DTYPE)
        return idx, entries
    # Group by pre and sort by pre
    from collections import defaultdict

    grp: Dict[int, list[StableEdge]] = defaultdict(list)
    for pid, e in pairs:
        grp[int(pid)].append(e)
    pre_ids = sorted(grp.keys())
    entries_list = []
    idx_rows = []
    start = 0
    for pid in pre_ids:
        es = grp[pid]
        cnt = len(es)
        idx_rows.append((pid, start, cnt))
        for e in es:
            entries_list.append(
                (
                    int(e.pre_id),
                    int(e.post_id),
                    int(e.sign),
                    int(e.delay),
                    int(e.state),
                    int(e.age),
                    int(e.corr),
                    int(e.last_used),
                )
            )
        start += cnt
    idx = np.array(idx_rows, dtype=np.int32)
    entries = np.array(entries_list, dtype=_ENTRY_DTYPE)
    return idx, entries


def _arrays_to_store(idx: np.ndarray, entries: np.ndarray) -> StableStore:
    st = StableStore()
    if idx.size == 0:
        return st
    for row in idx:
        pid, start, cnt = int(row[0]), int(row[1]), int(row[2])
        for i in range(start, start + cnt):
            rec = entries[i]
            e = StableEdge(
                pre_id=int(rec["pre_id"]),
                post_id=int(rec["post_id"]),
                sign=np.int8(int(rec["sign"])),
                delay=np.uint8(int(rec["delay"])),
                state=np.uint8(int(rec["state"])),
                age=np.uint16(int(rec["age"])),
                corr=np.int16(int(rec["corr"])),
                last_used=np.int32(int(rec["last_used"])),
            )
            st.add(e)
    return st


def save_stable(path: str, store: StableStore) -> None:
    idx, entries = _store_to_arrays(store)
    # Header with segments meta
    def _dt_meta(x: np.dtype):
        return x.descr if x.fields is not None else x.str

    meta = {
        "version": 1,
        "segments": [
            {"name": "stable_index", "dtype": _dt_meta(idx.dtype), "shape": list(idx.shape)},
            {"name": "stable_entries", "dtype": _dt_meta(entries.dtype), "shape": list(entries.shape)},
        ],
    }
    meta_json = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    header = MAGIC + int.to_bytes(len(meta_json), 8, "little") + meta_json

    sizes = [idx.nbytes, entries.nbytes]
    total = len(header) + sum(sizes)
    with open(path, "wb") as f:
        f.truncate(total)
        f.seek(0)
        f.write(header)

    off = len(header)
    for arr, dtype, shape, sz in [
        (idx, idx.dtype, idx.shape, idx.nbytes),
        (entries, entries.dtype, entries.shape, entries.nbytes),
    ]:
        mm = np.memmap(path, dtype=dtype, mode="r+", offset=off, shape=shape)
        mm[...] = arr
        mm.flush()
        del mm
        off += sz


def load_stable(path: str) -> StableStore:
    with open(path, "rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("Invalid stable snapshot magic header")
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
    return _arrays_to_store(loaded["stable_index"], loaded["stable_entries"])


__all__ = ["save_stable", "load_stable"]
