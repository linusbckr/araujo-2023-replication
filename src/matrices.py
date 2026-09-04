"""
Spatial weight matrices for the flying-rivers model.
====================================================

The object the paper's Eq. [1] needs is a family of K matrices W_t^[k] linking
each pixel to the cells that lie k geographic steps upwind of it at month t:

    W_t^[k] = shell_k  (elementwise AND)  C_t          binary, zero diagonal

where ``shell_k`` is the set of pixel pairs at Queen-graph geodesic distance
*exactly* k (so the K shells are disjoint, and a pair belongs to at most one)
and ``C_t`` is the circulation matrix, which marks the pairs a 5-day 800 hPa
back-trajectory actually connects in month t.

Three properties matter downstream and are easy to get wrong:

1. **W is binary, never row-normalised.**  W_t^[k] Y is therefore a SUM of LAI
   over the upwind cells at shell k, not their mean.  Applying the published
   beta_k to row-normalised matrices does not reproduce the paper's multiplier
   (Omega^2 mean 1.72 against its reported 2.00, versus 1.96 binary), so the
   binary reading is the one its coefficients belong to.
2. **The shells are disjoint.**  Binary powers (W_t^[1])^k are an alternative
   reading — a k-step chain along the wind rather than a ring at distance k —
   but they overlap, their row sums grow with k (1.2 -> 7.9 over k = 1..20
   against a flat ~1 for shells), and under the paper's own beta_k they would
   imply Omega^2 ~ 6 rather than 2.0.  Shells, not powers.
3. **Row sums are ~1.**  A single back-trajectory touches about one cell per
   shell.  That is what makes Sum_k beta_k the leading term of the multiplier,
   and it is what pins the paper's matrices to be like these ones.

:func:`row_standardise` is provided because the *composition* specification
(Model 2 in this package) needs it — but note that it is applied to the
INSTRUMENT only, never to the regressor or to the matrices entering Omega.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

QUEEN_FACTOR = 2.0 ** 0.5 + 1e-6   # edge- and corner-sharing neighbours
VORONOI_FACTOR = 0.7072            # half-diagonal: nearest-cell assignment


def _binarize(M: sp.spmatrix) -> sp.csr_matrix:
    """
    Binary {0,1} int8 copy: every structural non-zero becomes 1.

    ORDER MATTERS.  Explicit zeros must be pruned BEFORE the data are set to
    one, otherwise a set difference (``A - A AND B``) and every ``setdiag(0)``
    silently come back as ones — which reinstates the diagonal and adds exactly
    1 to every row sum.
    """
    out = M.tocsr(copy=True)
    out.eliminate_zeros()
    if out.nnz:
        out.data = np.ones_like(out.data, dtype=np.int8)
        return out
    return out.astype(np.int8)


def pixel_order(grid_df: pd.DataFrame) -> np.ndarray:
    """Canonical row/column ordering: pixel_id ascending.

    Every matrix and every state vector in this package is indexed this way.
    Note that pixel_id is *preserved, not renumbered*, when a run is restricted
    to an estimation domain, so these values need not be contiguous.
    """
    return np.sort(grid_df["pixel_id"].to_numpy(np.int64))


def queen_adjacency(grid_df: pd.DataFrame, resolution: float) -> sp.csr_matrix:
    """Static Queen's-contiguity (8-neighbour) adjacency over the pixel grid."""
    order = pixel_order(grid_df)
    coords = (grid_df.set_index("pixel_id").loc[order, ["latitude", "longitude"]]
              .to_numpy(np.float64))
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=QUEEN_FACTOR * resolution, output_type="ndarray")
    n = len(order)
    if len(pairs) == 0:
        return sp.csr_matrix((n, n), dtype=np.int8)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    G = sp.coo_matrix((np.ones(len(rows), np.int8), (rows, cols)),
                      shape=(n, n)).tocsr()
    G.setdiag(0)
    return _binarize(G)


def geodesic_shells(G: sp.csr_matrix, K: int) -> Dict[int, sp.csr_matrix]:
    """
    shell_k[i, j] = 1 iff the Queen-graph shortest path from i to j is exactly k.

    Breadth-first over the adjacency, subtracting everything already reached, so
    each shell is a thin disjoint ring and the set stays sparse.  Shells beyond
    the graph diameter come back empty rather than raising.
    """
    n = G.shape[0]
    A = _binarize(G)
    A.setdiag(0)
    A = _binarize(A)
    shells: Dict[int, sp.csr_matrix] = {}
    reached = _binarize(sp.identity(n, format="csr"))
    frontier = _binarize(sp.identity(n, format="csr"))
    for k in range(1, K + 1):
        nxt = _binarize(frontier @ A)
        shell = _binarize(nxt - nxt.multiply(reached))
        shell.setdiag(0)
        shell = _binarize(shell)
        shells[k] = shell
        reached = _binarize(reached + nxt)
        frontier = shell
        if shell.nnz == 0:
            for kk in range(k + 1, K + 1):
                shells[kk] = sp.csr_matrix((n, n), dtype=np.int8)
            break
    return shells


def upwind_matrices(C_t: sp.spmatrix, shells: Dict[int, sp.csr_matrix],
                    K: int) -> Dict[int, sp.csr_matrix]:
    """W_t^[k] = shell_k AND C_t, binary, zero diagonal, float64 for matvecs."""
    n = C_t.shape[0]
    C = _binarize(C_t)
    out: Dict[int, sp.csr_matrix] = {}
    for k in range(1, K + 1):
        shell = shells[k]
        if shell.nnz == 0 or C.nnz == 0:
            out[k] = sp.csr_matrix((n, n), dtype=np.float64)
            continue
        W = _binarize(shell.multiply(C))
        W.setdiag(0)
        out[k] = _binarize(W).astype(np.float64)
    return out


def row_standardise(W: sp.spmatrix) -> sp.csr_matrix:
    """W / rowsum, leaving all-zero rows at zero.  For the INSTRUMENT only."""
    Wf = W.astype(np.float64).tocsr()
    rs = np.asarray(Wf.sum(axis=1)).ravel()
    inv = np.where(rs > 0, 1.0 / np.where(rs > 0, rs, 1.0), 0.0)
    return (sp.diags(inv) @ Wf).tocsr()


def circulation_from_trajectories(traj_lat: np.ndarray, traj_lon: np.ndarray,
                                  origin_row: np.ndarray,
                                  coords: np.ndarray) -> sp.csr_matrix:
    """
    Map trajectory points onto grid cells by NEAREST-CELL assignment.

    A point counts as visiting a cell when it falls inside that cell's Voronoi
    region, i.e. within ``VORONOI_FACTOR * resolution`` of its centre.  No
    spatial buffer is applied around the path.  That is deliberate: the paper's
    own beta_k, applied to matrices whose shells carried more mass than these,
    would overshoot its own reported multiplier — a uniform inflation of only
    2.7% already reaches Omega^2 = 2.00, and dispersion growing as sqrt(k)
    reaches 4.63.
    """
    n = len(coords)
    tree = cKDTree(coords)
    dist, j = tree.query(np.column_stack([traj_lat, traj_lon]), k=1)
    res = float(np.min(np.diff(np.unique(coords[:, 0])))) if n > 1 else 1.0
    keep = (dist <= VORONOI_FACTOR * res) & (origin_row >= 0)
    if not keep.any():
        return sp.csr_matrix((n, n), dtype=np.int8)
    C = sp.coo_matrix(
        (np.ones(int(keep.sum()), np.int8), (origin_row[keep], j[keep])),
        shape=(n, n)).tocsr()
    C.setdiag(0)
    return _binarize(C)


def time_average(per_timestamp: Iterable[Dict[int, sp.csr_matrix]],
                 K: int, n: int, standardise: bool = False
                 ) -> Dict[int, sp.csr_matrix]:
    """
    Wbar^[k] = (1/T) sum_t W_t^[k].

    Accumulated in batches: repeatedly adding to a growing CSR is markedly
    slower than summing a batch in one pass.
    """
    acc: Dict[int, List[sp.csr_matrix]] = {k: [] for k in range(1, K + 1)}
    T = 0
    for W_k in per_timestamp:
        T += 1
        for k in range(1, K + 1):
            W = W_k[k]
            acc[k].append(row_standardise(W) if standardise else W)
        if T % 30 == 0:
            for k in range(1, K + 1):
                acc[k] = [sum(acc[k])]
    if T == 0:
        return {k: sp.csr_matrix((n, n)) for k in range(1, K + 1)}
    return {k: (sum(acc[k]) / T if acc[k] else sp.csr_matrix((n, n)))
            for k in range(1, K + 1)}


def mean_row_sums(Wbar: Dict[int, sp.csr_matrix], K: int) -> np.ndarray:
    return np.array([float(np.asarray(Wbar[k].sum(axis=1)).ravel().mean())
                     for k in range(1, K + 1)])


def omega_margins(beta: Sequence[float], Wbar: Dict[int, sp.csr_matrix],
                  K: int, delta: int = 1, max_iter: int = 500):
    """
    Row and column sums of Omega^(delta+1), Omega = (I - sum_k beta_k Wbar^[k])^-1.

    The paper defines its indices of influence and exposure on Omega^(Delta+1)
    and maps Delta = 1, so its "average multiplier effect is 2" and its ~2.64
    maximum are **Omega squared** quantities.  Comparing Omega's own margins
    against 2.0 is the wrong benchmark and is invisible while Omega ~ I.

    Computed by repeated matvec on the Neumann series, so Omega is never
    densified.  Returns (row_margins, col_margins), or (None, None) if the
    series does not converge — which is itself informative, since it means the
    implied spectral radius is >= 1.
    """
    n = Wbar[1].shape[0]
    A = sp.csr_matrix((n, n))
    for k in range(1, K + 1):
        if beta[k - 1] != 0.0 and Wbar[k].nnz:
            A = A + float(beta[k - 1]) * Wbar[k]
    A = A.tocsr()

    def apply(M: sp.csr_matrix, v: np.ndarray):
        total, cur = v.copy(), v.copy()
        for _ in range(max_iter):
            cur = M @ cur
            total = total + cur
            if np.abs(cur).max() < 1e-13:
                return total
        return None

    ones = np.ones(n)
    r = ones
    c = ones
    for _ in range(delta + 1):
        r = apply(A, r)
        if r is None:
            return None, None
    AT = A.T.tocsr()
    for _ in range(delta + 1):
        c = apply(AT, c)
        if c is None:
            return None, None
    return r, c
