from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _to_adj(adj: np.ndarray) -> np.ndarray:
    a = np.asarray(adj)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adj must be square 2-D array")
    # Boolean adjacency: treat >0 as edge
    a = (a != 0).astype(np.uint8)
    # Zero out diagonal for metrics unless caller explicitly wants self-loops
    np.fill_diagonal(a, 0)
    return a


def degree_hist(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (in_deg_hist, out_deg_hist) as hist arrays of length N+1.

    Input `adj` is an NxN 0/1 matrix where adj[i,j]=1 if i->j edge exists.
    """
    a = _to_adj(adj)
    indeg = a.sum(axis=0)
    outdeg = a.sum(axis=1)
    N = a.shape[0]
    in_hist = np.bincount(indeg, minlength=N + 1)
    out_hist = np.bincount(outdeg, minlength=N + 1)
    return in_hist.astype(np.int64), out_hist.astype(np.int64)


def inout_invariants(adj: np.ndarray) -> Dict[str, float]:
    a = _to_adj(adj)
    N = a.shape[0]
    m = int(a.sum())
    indeg = a.sum(axis=0)
    outdeg = a.sum(axis=1)
    return {
        "N": float(N),
        "M": float(m),
        "mean_in": float(np.mean(indeg)),
        "mean_out": float(np.mean(outdeg)),
        "var_in": float(np.var(indeg)),
        "var_out": float(np.var(outdeg)),
    }


def avg_path_len_approx(adj: np.ndarray, k: int = 256) -> float:
    """Approximate average shortest path length via BFS from up to k sources.

    Disconnected pairs are ignored in the average (standard convention for directed graphs).
    """
    a = _to_adj(adj)
    N = a.shape[0]
    if N == 0:
        return 0.0
    k = int(min(max(1, k), N))
    # Build adjacency lists for BFS
    nbrs = [np.nonzero(a[i])[0].tolist() for i in range(N)]
    sources = np.linspace(0, N - 1, num=k, dtype=int)
    dists = []
    for s in sources:
        dist = [-1] * N
        q = [s]
        dist[s] = 0
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            for v in nbrs[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        d = [d for d in dist if d > 0]
        dists.extend(d)
    if not dists:
        return float("inf")
    return float(np.mean(dists))


def clustering_coeff_approx(adj: np.ndarray, k: int = 256) -> float:
    """Approximate (undirected) clustering coefficient by sampling k nodes.

    Treats the directed graph as undirected for clustering; self-loops ignored.
    """
    a = _to_adj(adj)
    N = a.shape[0]
    if N == 0:
        return 0.0
    au = ((a + a.T) != 0).astype(np.uint8)
    np.fill_diagonal(au, 0)
    k = int(min(max(1, k), N))
    nodes = np.linspace(0, N - 1, num=k, dtype=int)
    coeffs = []
    for u in nodes:
        neigh = np.nonzero(au[u])[0]
        d = int(neigh.size)
        if d < 2:
            continue
        # Count edges among neighbors (undirected)
        sub = au[np.ix_(neigh, neigh)]
        # Each undirected edge counted twice in adjacency symmetry; divide by 2
        e = int(sub.sum() // 2)
        coeffs.append((2.0 * e) / (d * (d - 1)))
    if not coeffs:
        return 0.0
    return float(np.mean(coeffs))


def small_worldness(adj: np.ndarray, samples: int = 128) -> float:
    """Return small-worldness sigma ≈ (C/Cr) / (L/Lr) using ER approximations.

    For an ER graph with N nodes and mean degree k, Cr ≈ k/N; Lr ≈ ln(N)/ln(k).
    """
    a = _to_adj(adj)
    inv = inout_invariants(a)
    N = int(inv["N"]) or 1
    k = float(inv["mean_out"])  # mean degree
    if k <= 1.0:
        return 0.0
    C = clustering_coeff_approx(a, k=min(samples, N))
    L = avg_path_len_approx(a, k=min(samples, N))
    # ER baselines
    Cr = k / float(N)
    Lr = np.log(N) / np.log(max(2.0, k))
    return float((C / max(Cr, 1e-12)) / (L / max(Lr, 1e-12)))


__all__ = [
    "degree_hist",
    "inout_invariants",
    "avg_path_len_approx",
    "clustering_coeff_approx",
    "small_worldness",
]

