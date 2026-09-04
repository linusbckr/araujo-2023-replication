"""
Numerically stable FD-2SLS, and the two specifications the package compares.
============================================================================

The model (Araujo et al. 2023, Eq. [1]) has pixel fixed effects, so it is
estimated in first differences (Eq. [4]):

    dY_it = alpha dY_it-1 + sum_k beta_k d(W_t^[k] Y_it) + e_it

with two sources of endogeneity — dY_it-1 is mechanically correlated with the
differenced error (Nickell 1981), and the spatial lag is simultaneous (Manski's
reflection problem) — instrumented by Y_it-2 and by d(W_t^[k] Y_0), where Y_0 is
a baseline forest state that does not respond to Y_t.

Two specifications are estimated on identical samples.

MODEL 1 — BASELINE BINARY IV.  The instrument as written: d(W_t^[k] Y_0) with
W_t^[k] binary.  This reproduces the *shape* of Table S1 — twenty positive
coefficients, spread across lags much as the paper's are — at roughly a
eleventh of its level.

MODEL 2 — COMPOSITION IV.  W_t^[k] Y is a PRODUCT of the number of cells the
trajectory clips at shell k and their mean LAI:

    W_t^[k] Y = N_it^[k] * (Wtilde_t^[k] Y)_i

and those two factors behave completely differently.  The differenced regressor
splits exactly by the symmetric (Bennet) identity

    d(W_t^[k] Y) = (dN) * mbar  +  Nbar * (dm),     m = Wtilde_t^[k] Y

with mbar, Nbar the two-period midpoints.  Both terms are in dY's units, they
sum to the original regressor with no interaction residual, and constraining
their coefficients equal returns Model 1 exactly — so the comparison is a strict
nesting test, and :func:`bennet_split` asserts the identity numerically before
anything is estimated.

Why the split matters: the count factor is shared between the regressor and the
instrument, so it enters Cov(z, x) without entering Cov(z, dy), and since
beta_IV = Cov(z, dy) / Cov(z, x) it inflates the denominator alone.  In the real
panel it supplies ~79% of the binary instrument's variance at k = 1.  Isolating
the composition term recovers most of the paper's magnitude — but concentrates
it at k = 1 rather than spreading it over twenty lags.  Neither specification
reproduces both the level and the shape; that trade-off is the finding.

Numerics.  Projections go through a thin QR (X_hat = Q Q' X), never a formed
(Z'Z)^-1 or an n x n projector, and the second stage through an SVD with an
explicit rank cutoff.  Standard errors are cluster-robust at the pixel level and
— the usual manual-2SLS trap — the residuals are formed with the ACTUAL X, not
with X_hat, which would understate the variance.

Diagnostics.  Three statistics are reported alongside the coefficients, and the
first two answer questions the estimator cannot answer about itself.

  SANDERSON-WINDMEIJER conditional F, per endogenous regressor -- the weak-IV
    statistic that is valid when several regressors are endogenous (see tsls).

  ARELLANO-BOND m(2) -- Y_it-2 identifies dY_it-1 only if eps_it is serially
    uncorrelated, which is exactly the hypothesis that the differenced residual
    has no SECOND-order autocorrelation.  Every version of this model here has
    asserted that condition in a docstring; :func:`arellano_bond_m` tests it.
    m(1) is reported beside it and is expected to REJECT by construction, since
    d eps_t and d eps_t-1 share the term -eps_t-1; an m(1) that fails to reject
    is a sign the residuals are not what they are taken to be.

  HANSEN J -- not computable for the headline specifications, and that is a
    property of the model rather than an omission: K+1 endogenous regressors
    against K+1 excluded instruments leaves q - p = 0 degrees of freedom, so
    there is no overidentifying restriction to test.  :func:`fit_overidentified`
    adds Y_it-3 to obtain q > p and reports the J that results.  It is a
    DIAGNOSTIC on a DIFFERENT SAMPLE -- the deeper lag costs one further
    transition per pixel -- so its coefficients are never substituted for the
    headline ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
from scipy.stats import chi2, norm

from .matrices import row_standardise, upwind_matrices

# Araujo et al. (2023), SI Table S1.
PAPER_BETA = np.array([
    0.0811, 0.0391, 0.0262, 0.0197, 0.0152, 0.0100, 0.0076, 0.0093, 0.0091,
    0.0058, 0.0050, 0.0043, 0.0028, 0.0036, 0.0026, 0.0029, 0.0028, 0.0050,
    0.0017, 0.0040])
PAPER_ALPHA = 0.2200
PAPER_N_OBS = 3_300_494
PAPER_N_CLUSTERS = 9_539
PAPER_OMEGA2_MEAN = 2.00
PAPER_OMEGA2_MAX = 2.64
STOCK_YOGO_1IV = 16.38


@dataclass
class Fit:
    name:     str
    alpha:    float
    alpha_se: float
    beta:     np.ndarray
    beta_se:  np.ndarray
    n_obs:    int
    n_clust:  int
    sw_f_min: float
    cond_Z:   float

    # Arellano-Bond serial-correlation tests on the differenced residual.
    # ar2 is the one that licenses Y_it-2 as an instrument and must NOT reject;
    # ar1 is expected to reject by construction and is a sanity check on the
    # residuals rather than a specification test.  NaN means "not computed"
    # (the sample carried no timestamps), never "passed".
    ar1_m:    float = np.nan
    ar1_p:    float = np.nan
    ar2_m:    float = np.nan
    ar2_p:    float = np.nan

    @property
    def sum_beta(self) -> float:
        return float(self.beta.sum())

    def shape(self) -> Dict[str, float]:
        return shape_metrics(self.beta)


# ── linear algebra ──────────────────────────────────────────────────────────

def cluster_starts(cluster_ids: np.ndarray) -> np.ndarray:
    if np.any(np.diff(cluster_ids) < 0):
        raise ValueError("rows must be sorted by cluster id")
    return np.flatnonzero(np.r_[True, np.diff(cluster_ids) != 0])


def _sandwich(X_score, resid, starts, bread, n, p):
    S = np.add.reduceat(X_score * resid[:, None], starts, axis=0)
    G = len(starts)
    corr = (G / (G - 1)) * ((n - 1) / max(n - p, 1))
    return corr * (bread @ (S.T @ S) @ bread)


def demean(M: np.ndarray) -> np.ndarray:
    """Partial out the constant by Frisch-Waugh."""
    return M - M.mean(axis=0, keepdims=True)


def tsls(y, X, Z, starts, want_sw: bool = True):
    """
    Two-stage least squares, thin-QR projection + SVD second stage.

    Returns (coef, se, sw_F, cond_Z, aux).  ``aux`` carries the pieces the
    post-estimation tests need — the residuals, the projected design X_hat, the
    bread (X_hat'X_hat)^-1 and the clustered vcov — so that
    :func:`arellano_bond_m` can apply the Arellano-Bond estimation-effect
    correction instead of treating the residuals as if delta were known.

    ``sw_F`` is the Sanderson-Windmeijer
    conditional F per endogenous regressor — the statistic that is valid when
    several regressors are endogenous.  The naive per-variable partial F of all
    excluded instruments is NOT: it credits each regressor with identifying
    power spent on the others, and on the real panel it reports ~1e6 for a model
    that is only just identified.  SW numerator df is q - p + 1, which is 1 when
    the system is exactly identified (Stock-Yogo reference 16.38).
    """
    n, p = X.shape
    q = Z.shape[1]
    Q, _ = np.linalg.qr(Z, mode="reduced")          # orthonormal basis for col(Z)
    Xh = Q @ (Q.T @ X)                              # P_Z X, no (Z'Z)^-1 formed

    U, s, Vt = la.svd(Xh, full_matrices=False)      # rank-aware second stage
    tol = s[0] * max(Xh.shape) * np.finfo(float).eps
    S_inv = np.where(s > tol, 1.0 / np.where(s > tol, s, 1.0), 0.0)
    coef = Vt.T @ (S_inv * (U.T @ y))

    bread = Vt.T @ np.diag(S_inv ** 2) @ Vt         # = (Xh'Xh)^-1, rank-safe
    resid = y - X @ coef                            # ACTUAL X, not X_hat
    v = _sandwich(Xh, resid, starts, bread, n, p)
    se = np.sqrt(np.maximum(np.diag(v), 0.0))

    sw = np.full(p, np.nan)
    if want_sw:
        df_num = max(q - p + 1, 1)
        Sinv = bread
        for j in range(p):
            d = -Sinv[:, j] / Sinv[j, j]
            d[j] = 0.0
            xp = X[:, j] + X @ d                    # x_j - X_{-j} delta_j
            rss_u = float(np.sum((xp - Q @ (Q.T @ xp)) ** 2))
            rss_r = float(np.sum(xp ** 2))
            sw[j] = (((rss_r - rss_u) / df_num)
                     / max(rss_u / max(n - q - 1, 1), 1e-300))

    sv = la.svdvals(Z)
    aux = {"resid": resid, "Xh": Xh, "bread": bread, "vcov": v, "X": X}
    return coef, se, sw, float(sv[0] / max(sv[-1], 1e-300)), aux


# ── post-estimation tests ───────────────────────────────────────────────────

def _lag_within_pixel(vals: np.ndarray, pid: np.ndarray, tidx: np.ndarray,
                      order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    ``vals`` shifted ``order`` timestamps back WITHIN each pixel.

    Rows must be sorted by (pixel, time), which makes the composite key
    monotone and lets a single searchsorted do the whole lookup.  Rows with no
    partner — the first ``order`` transitions of a pixel, and any row whose
    partner was dropped by the finite-value mask — come back as zero and are
    flagged in the returned mask, so an unbalanced panel contributes nothing
    rather than contributing a wrong pairing.
    """
    span = int(tidx.max()) + 1
    key = pid.astype(np.int64) * span + tidx
    want = pid.astype(np.int64) * span + (tidx - order)
    pos = np.searchsorted(key, want)
    pos_c = np.clip(pos, 0, len(key) - 1)
    ok = (pos < len(key)) & (key[pos_c] == want) & (tidx >= order)
    out = np.zeros_like(vals, dtype=np.float64)
    out[ok] = vals[pos_c[ok]]
    return out, ok


def arellano_bond_m(resid, pid, tidx, X, Xh, bread, vcov, starts, order=2):
    """
    Arellano-Bond m-statistic for order-``order`` serial correlation in the
    differenced residual.  Standard normal under the null of no such
    correlation; returns (m, p_two_sided).

    Why the correction terms.  The naive statistic
    sum_t e_t e_t-k / sqrt(sum_i (sum_t e_t e_t-k)^2) treats e as if delta were
    known.  It is not — e = eps - X(delta_hat - delta) — and ignoring that
    understates the variance, which biases the test TOWARDS rejecting.  The
    variance below is Arellano & Bond's (1991, the expression following their
    eq. 10) in its cluster-robust form:

        v = sum_i a_i^2  -  2 c' (Xh'Xh)^-1 sum_i b_i a_i  +  c' Var(delta) c

    with a_i = e_-k,i' e_i the per-pixel scalar, b_i = Xh_i' e_i the per-pixel
    score, and c = e_-k' X.  The second and third terms are exactly the
    estimation effect; dropping them is the usual shortcut and is not taken
    here.

    Interpretation for THIS model.  The FD-2SLS instrument Y_it-2 is valid iff
    eps_it is serially uncorrelated, i.e. iff m(2) does not reject.  m(1)
    rejecting is mechanical: d eps_t and d eps_t-1 both contain -eps_t-1.
    """
    e_lag, _ = _lag_within_pixel(resid, pid, tidx, order)
    d = float(e_lag @ resid)

    a = np.add.reduceat(e_lag * resid, starts)                 # (G,)
    B = np.add.reduceat(Xh * resid[:, None], starts, axis=0)   # (G, p)
    c = e_lag @ X                                              # (p,)

    v = (float(a @ a)
         - 2.0 * float(c @ bread @ (B.T @ a))
         + float(c @ vcov @ c))
    if not np.isfinite(v) or v <= 0.0:
        return np.nan, np.nan
    m = d / np.sqrt(v)
    return float(m), float(2.0 * norm.sf(abs(m)))


def gmm_j(y, X, Z, starts):
    """
    Two-step cluster-robust GMM and its Hansen J.
    Returns (coef, se, J, df, p).

    df = q - p.  When df == 0 the model is exactly identified, J is identically
    zero and p is NaN: there is no overidentifying restriction, and reporting a
    "passed" test in that case would be worse than reporting none.
    """
    n, p = X.shape
    q = Z.shape[1]
    df = q - p
    b1, _, _, _, _ = tsls(y, X, Z, starts, want_sw=False)
    e1 = y - X @ b1
    S = np.add.reduceat(Z * e1[:, None], starts, axis=0)
    Om_inv = la.pinv(S.T @ S)

    ZtX, Zty = Z.T @ X, Z.T @ y
    A = ZtX.T @ Om_inv
    b2 = la.solve(A @ ZtX, A @ Zty)
    g = Z.T @ (y - X @ b2)
    J = float(g @ Om_inv @ g)

    V = la.pinv(ZtX.T @ Om_inv @ ZtX)
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return b2, se, J, df, (float(chi2.sf(J, df)) if df > 0 else np.nan)


# ── channels ────────────────────────────────────────────────────────────────

def build_channels(panel, Y0_table: np.ndarray, progress: bool = False):
    """
    Per (t, i, k): the row sum N, the composition m = Wtilde Y_t, and the
    baseline composition m0 = Wtilde Y_0.  Returns three (T, N, K) arrays.
    """
    from .matrices import geodesic_shells
    K, N, T = panel.K, panel.N, panel.T
    shells = geodesic_shells(panel.G, K)
    Yf = np.nan_to_num(panel.Y, nan=0.0)
    Nc = np.zeros((T, N, K)); M = np.zeros((T, N, K)); M0 = np.zeros((T, N, K))
    for i, ts in enumerate(panel.times):
        W_k = upwind_matrices(panel.C_t[ts], shells, K)
        y0 = Y0_table[:, ts.month - 1]
        for k in range(1, K + 1):
            W = W_k[k]
            if W.nnz == 0:
                continue
            rs = np.asarray(W.sum(axis=1)).ravel()
            Wt = row_standardise(W)
            Nc[i, :, k - 1] = rs
            M[i, :, k - 1] = Wt @ Yf[:, i]
            M0[i, :, k - 1] = Wt @ y0
        if progress and (i + 1) % 60 == 0:
            print(f"      … {i + 1}/{T} months", flush=True)
    return Nc, M, M0


def bennet_split(Nc, M, M0):
    """
    Exact symmetric decomposition of the differenced regressor and instrument.

    Asserts  count + composition == d(W Y)  to 1e-8 on both, because the whole
    nesting argument rests on the identity holding exactly rather than
    approximately.
    """
    dN = Nc[1:] - Nc[:-1]
    Nbar = 0.5 * (Nc[1:] + Nc[:-1])
    x_cnt = dN * (0.5 * (M[1:] + M[:-1]))
    x_cmp = Nbar * (M[1:] - M[:-1])
    z_cnt = dN * (0.5 * (M0[1:] + M0[:-1]))
    z_cmp = Nbar * (M0[1:] - M0[:-1])
    dx = Nc[1:] * M[1:] - Nc[:-1] * M[:-1]
    dz = Nc[1:] * M0[1:] - Nc[:-1] * M0[:-1]
    err_x = float(np.abs(x_cnt + x_cmp - dx).max())
    err_z = float(np.abs(z_cnt + z_cmp - dz).max())
    if not (err_x < 1e-8 and err_z < 1e-8):
        raise AssertionError(
            f"Bennet identity violated: {err_x:.3e} (regressor), "
            f"{err_z:.3e} (instrument)")
    return x_cnt, x_cmp, z_cnt, z_cmp, dx, dz, (err_x, err_z)


def assemble(panel, Nc, M, M0, spec):
    """
    Build the estimation sample for one :class:`~src.data_loader.Spec`.

    Rows are timestamp indices 1..T-1; the first two are lost to lagging, which
    is what makes a balanced panel come out at N x (T - 2) observations.

    With a season-matched Y_0 the baseline year's transitions are the ones on
    which the excluded instrument is identically the endogenous regressor, and
    the two season-matched specifications differ ONLY in what they do about it:
    ``season-matched`` drops those rows, ``authors-346`` retains them because the
    paper's own observation count implies it did.  Either way the count is
    printed, so the log records which sample produced which column instead of
    leaving it to be inferred.  This is a property of each specification, not a
    tunable option -- see Spec's docstring.
    """
    x_cnt, x_cmp, z_cnt, z_cmp, dx, dz, err = bennet_split(Nc, M, M0)
    Y, T, N, K = panel.Y, panel.T, panel.N, panel.K

    dy = (Y[:, 1:] - Y[:, :-1]).T                                   # (T-1, N)
    dyl = np.full_like(dy, np.nan); dyl[1:] = (Y[:, 1:T - 1] - Y[:, :T - 2]).T
    l2 = np.full_like(dy, np.nan);  l2[1:] = Y[:, :T - 2].T
    # Y_it-3, for the overidentified diagnostic only.  It is deliberately NOT
    # part of the finite-value mask below: including it would cost one further
    # transition per pixel in EVERY specification, turning the 346 of
    # authors-346 into 345 and silently moving all six headline estimates.
    l3 = np.full_like(dy, np.nan)
    if T >= 4:
        l3[2:] = Y[:, :T - 3].T

    def flat(a):                       # (T-1, N, K) -> (N*(T-1), K), pixel-major
        return np.transpose(a, (1, 0, 2)).reshape(N * (T - 1), a.shape[2])

    def flat2(a):                      # (T-1, N) -> (N*(T-1), 1)
        return a.T.reshape(N * (T - 1), 1)

    cols = {"dy": flat2(dy), "dyl": flat2(dyl), "l2": flat2(l2),
            "l3": flat2(l3),
            "xc": flat(x_cnt), "xm": flat(x_cmp),
            "zc": flat(z_cnt), "zm": flat(z_cmp),
            "dx": flat(dx),   "dz": flat(dz)}
    pid = np.repeat(panel.grid_df["pixel_id"].to_numpy(np.int64), T - 1)
    ts = np.tile(panel.times[1:].to_numpy(), N)

    stacked = np.column_stack([cols[c] for c in
                               ("dy", "dyl", "l2", "xc", "xm", "zc", "zm",
                                "dx", "dz")])
    keep = np.isfinite(stacked).all(axis=1)
    if spec.y0_kind == "season":
        base_year = int(panel.times.year.min())
        if panel.Y0_table is not None:
            # Demo mode draws Y_0 from a PRE-SAMPLE year, so there is no overlap
            # either to remove or to retain -- which also means the two
            # season-matched columns coincide exactly in demo mode.  Stated
            # rather than skipped silently.
            print(f"    [{spec.name}] baseline is pre-sample: no overlapping "
                  "rows, so this column coincides with the other season-matched "
                  "one in demo mode")
        elif spec.excludes_baseline_year:
            n0 = int(keep.sum())
            keep &= pd.DatetimeIndex(ts).year > base_year
            n1 = int(keep.sum())
            print(f"    [{spec.name}] excluded baseline year {base_year}: "
                  f"{n0:,} -> {n1:,} rows ({1 - n1 / n0:.2%})")
        else:
            n_keep = int(keep.sum())
            n_in = int((keep & (pd.DatetimeIndex(ts).year == base_year)).sum())
            print(f"    [{spec.name}] baseline year {base_year} RETAINED: "
                  f"{n_in:,} of {n_keep:,} rows ({n_in / max(n_keep, 1):.2%}) on "
                  "which the excluded instrument IS the endogenous regressor")
    out = {c: cols[c][keep] for c in cols}
    out["pixel_id"] = pid[keep]
    out["ts"] = ts[keep]          # needed to pair residuals for Arellano-Bond
    out["bennet_err"] = err
    return out


# ── the two specifications ──────────────────────────────────────────────────

def _period_index(ts: np.ndarray) -> np.ndarray:
    """
    Timestamps -> their rank among the distinct timestamps present.

    Rank rather than month arithmetic, so that "two periods back" means two
    steps of whatever cadence the panel actually has.  The production panel is
    monthly and the two agree there, but an 8-day panel would collide under
    month arithmetic and silently pair the wrong rows.
    """
    return np.unique(np.asarray(ts), return_inverse=True)[1].astype(np.int64)


def _attach_ar(fit: Fit, sample, order, pid, starts, aux) -> None:
    """Run the AB m(1) and m(2) tests for one fit and record them on it."""
    if "ts" not in sample:
        return
    tidx = _period_index(sample["ts"][order])
    for k, (mf, pf) in ((1, ("ar1_m", "ar1_p")), (2, ("ar2_m", "ar2_p"))):
        m, p = arellano_bond_m(aux["resid"], pid, tidx, aux["X"], aux["Xh"],
                               aux["bread"], aux["vcov"], starts, order=k)
        setattr(fit, mf, m)
        setattr(fit, pf, p)


def fit_models(sample) -> Tuple[Fit, Fit]:
    pid = sample["pixel_id"]
    order = np.argsort(pid, kind="stable")
    pid = pid[order]
    starts = cluster_starts(pid)
    g = lambda c: demean(sample[c][order])                          # noqa: E731

    y = g("dy").ravel()
    dyl, l2 = g("dyl"), g("l2")
    DX, DZ, XC, XM, ZC, ZM = (g(c) for c in ("dx", "dz", "xc", "xm", "zc", "zm"))
    K = DX.shape[1]
    n, G = len(y), len(starts)

    b1, se1, sw1, c1, aux1 = tsls(y, np.column_stack([dyl, DX]),
                                  np.column_stack([l2, DZ]), starts)
    m1 = Fit("Model 1 — baseline binary IV", float(b1[0]), float(se1[0]),
             b1[1:], se1[1:], n, G, float(np.nanmin(sw1)), c1)
    _attach_ar(m1, sample, order, pid, starts, aux1)

    b2, se2, sw2, c2, aux2 = tsls(y, np.column_stack([dyl, XC, XM]),
                                  np.column_stack([l2, ZC, ZM]), starts)
    m2 = Fit("Model 2 — composition IV", float(b2[0]), float(se2[0]),
             b2[K + 1:], se2[K + 1:], n, G, float(np.nanmin(sw2)), c2)
    m2.beta_count = b2[1:K + 1]                                     # type: ignore[attr-defined]
    m2.beta_count_se = se2[1:K + 1]                                 # type: ignore[attr-defined]
    _attach_ar(m2, sample, order, pid, starts, aux2)

    for f in (m1, m2):
        if np.isfinite(f.ar2_m):
            tag = "REJECTS -> Y_it-2 is not a valid instrument" \
                  if f.ar2_p < 0.05 else "does not reject"
            print(f"    [{f.name.split('—')[1].strip()}] Arellano-Bond "
                  f"m(1)={f.ar1_m:+.2f} (p={f.ar1_p:.3f}, expected to reject) | "
                  f"m(2)={f.ar2_m:+.2f} (p={f.ar2_p:.3f}, {tag})")
    return m1, m2


def fit_overidentified(sample) -> Optional[Dict[str, float]]:
    """
    Add Y_it-3 to the instrument set so that q > p, and report the Hansen J.

    This exists because the headline specification CANNOT be tested for
    overidentification: K+1 endogenous regressors against K+1 excluded
    instruments is exact identification, q - p = 0.  Adding a deeper level lag
    buys one degree of freedom and with it the only available test of the
    exclusion restriction on Delta(W_t^[k] Y_0).

    It is a diagnostic and nothing else.  The deeper lag costs one further
    transition per pixel, so this runs on a DIFFERENT SAMPLE from every number
    in the summary table; its alpha and Sum beta_k are printed so the size of
    that sample shift is visible, and they are not substituted for the
    headline estimates anywhere.
    """
    if "l3" not in sample:
        return None
    ok = np.isfinite(sample["l3"]).ravel()
    if int(ok.sum()) < 100:
        return None

    pid = sample["pixel_id"][ok]
    order = np.argsort(pid, kind="stable")
    pid = pid[order]
    starts = cluster_starts(pid)
    g = lambda c: demean(sample[c][ok][order])                      # noqa: E731

    y = g("dy").ravel()
    X = np.column_stack([g("dyl"), g("dx")])
    Z = np.column_stack([g("l2"), g("l3"), g("dz")])
    b, se, J, df, p = gmm_j(y, X, Z, starts)
    return {"J": J, "df": float(df), "p": p, "n": float(len(y)),
            "n_dropped": float(len(ok) - int(ok.sum())),
            "alpha": float(b[0]), "sum_beta": float(b[1:].sum())}


# ── profile shape ───────────────────────────────────────────────────────────

def shape_metrics(beta: Sequence[float]) -> Dict[str, float]:
    """
    How the total is distributed across lags — the dimension on which the two
    specifications differ from Table S1 in opposite directions.

    share_k1   beta_1 / sum beta_k        (paper 0.315)
    eff_lags   (sum b)^2 / sum b^2, an inverse Herfindahl: the effective number
               of contributing lags (paper 6.71)
    corr_paper Pearson correlation with Table S1 over the K lags
    """
    b = np.asarray(beta, float)
    s, ss = float(b.sum()), float((b ** 2).sum())
    ref = PAPER_BETA[:len(b)]
    return {
        "share_k1":   float(b[0] / s) if abs(s) > 1e-12 else np.nan,
        "eff_lags":   (s ** 2 / ss) if ss > 0 else np.nan,
        "corr_paper": float(np.corrcoef(b, ref)[0, 1]) if len(b) > 1 else np.nan,
        "n_pos":      int((b > 0).sum()),
    }
