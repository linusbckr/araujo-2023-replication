"""
Data loading for both modes, plus the synthetic panel generator.
================================================================

FULL MODE reads one pickle holding ``(C_t_dict, G, panel_df)``: the circulation
matrices, the geography matrix and the LAI/wind panel.  ``src/build_inputs.py``
builds it from two monthly NetCDF files, and ``src/fetch_raw.py`` downloads
those from their public sources, so full mode depends on no artefact that has to
be obtained by request.  If the pickle is absent,
:func:`full_mode_instructions` prints what to run rather than failing with a
traceback.

DEMO MODE needs nothing.  It generates a small balanced panel from the *same
structural model* the paper estimates, with parameters the caller chooses, so
running it verifies that the estimator recovers coefficients that are known to
be there.  That is the part a reader can check in five seconds without
downloading anything.

On the LAI gap-fill
-------------------
Cloud-masked LAI is missing information, not bare ground.  Because W^[k] is
binary, W^[k]Y_t is a SUM over upwind neighbours, so a NaN passed downstream and
zero-filled enters as an LAI of 0 while still being counted in the row sum —
a silent downward bias on every lag that touches it.  The paper's own panel has
no gaps at all (SI Table S1's 3,300,494 observations on 9,539 clusters is
exactly 9,539 x 346, a balanced panel over 348 months less the two lost to
lagging), so its LAI must have been gap-filled.  :func:`interpolate_lai_gaps`
does the same, within pixel and along time.
"""

from __future__ import annotations

import pickle
import sys
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .matrices import (circulation_from_trajectories, geodesic_shells,
                       pixel_order, queen_adjacency, upwind_matrices)


@dataclass
class Panel:
    """Everything the estimation needs, in one object."""
    grid_df:  pd.DataFrame                       # [pixel_id, latitude, longitude]
    times:    pd.DatetimeIndex
    Y:        np.ndarray                         # (N, T) LAI, NaN = missing
    C_t:      Dict[pd.Timestamp, sp.csr_matrix]  # circulation per month
    G:        sp.csr_matrix                      # Queen adjacency
    K:        int
    label:    str
    truth:    Optional[Dict[str, object]] = field(default=None)
    # (N, 12) season-matched baseline. Supplied directly in demo mode, where it
    # comes from a PRE-SAMPLE year; computed from the sample's first year in
    # full mode, matching the reference analysis. See season_matched_baseline.
    Y0_table: Optional[np.ndarray] = field(default=None)

    @property
    def N(self) -> int:
        return self.Y.shape[0]

    @property
    def T(self) -> int:
        return self.Y.shape[1]


# ── gap-fill ────────────────────────────────────────────────────────────────

def interpolate_lai_gaps(Y: np.ndarray, times: pd.DatetimeIndex
                         ) -> Tuple[np.ndarray, int, int]:
    """
    Fill LAI gaps within each pixel, along time.  Returns (Y, n_before, n_after).

    Linear in time so unequal month lengths are respected; leading and trailing
    gaps are extended by the nearest value, so a pixel whose record starts late
    still yields a usable series.  Purely within-pixel: no spatial information
    is introduced and no cross-pixel correlation can be manufactured.
    """
    n_before = int(np.isnan(Y).sum())
    if n_before == 0:
        return Y, 0, 0
    wide = pd.DataFrame(Y, columns=pd.DatetimeIndex(times))
    filled = wide.interpolate(method="time", axis=1, limit_direction="both")
    out = filled.to_numpy(np.float64)
    return out, n_before, int(np.isnan(out).sum())


@dataclass(frozen=True)
class Spec:
    """
    One identification specification: how Y_0 is defined, and which rows are
    admissible as a consequence.

    Three exist, and they are different ESTIMANDS rather than versions of one
    thing.  All are reported by default, because the differences between them are
    themselves findings.  ``y0_kind`` says how Y_0 is built, and
    ``excludes_baseline_year`` says which rows that choice admits; the pair, not
    either alone, defines the estimand.

    PAPER_LITERAL — Y_0 is each pixel's first observation, a single fixed
        cross-section held for all t.  This is the literal reading of Eq. [6]
        ("initial forest status") and is what the parent pipeline estimates.
        The authors have since confirmed (correspondence, 2026-08-18) that this
        is NOT their construction, so it is retained as a diagnostic — the price
        of the fixed cross-section — and no longer as a reading of the paper.

    SEASON_MATCHED — Y_0 is the baseline year's value for the SAME CALENDAR
        MONTH as t; forest frozen in 1985, only the season aligned.  This is the
        authors' construction: the same correspondence confirmed that Y_0 is
        fixed at the sample's first year -- 1985 in their application -- but
        matched by calendar month.

        This variant additionally EXCLUDES the baseline year from the estimation
        sample.  Inside it, Y_0^m(t) is literally Y_t, so the excluded
        instrument is identically the endogenous regressor (verified: max
        deviation exactly 0) and those rows behave like OLS.  They are 2.89% of
        the real sample and move the estimates by 40-50%.  It is therefore the
        OVERLAP-EXCLUDED version of the authors' specification, and the estimate
        to prefer on econometric grounds.

    AUTHORS_346 — the same season-matched Y_0, estimated on ALL 346 transitions,
        i.e. with the baseline year retained.  This reconstructs the authors'
        own sample rather than a preferred estimate: their SI Table S1 reports
        3,300,494 observations on 9,539 clusters, which is exactly 9,539 x 346,
        the whole record.  Two further signals agree — on this sample alpha comes
        out at 0.22011 against their published 0.2200 (the corrected sample gives
        0.2338), and all twenty beta_k are positive, which is the sign pattern
        they report (the corrected sample gives 18/20).

        It therefore carries the overlap described above BY CONSTRUCTION,
        and that is the point of it: it is what their stated procedure produces
        on our data.  Report it beside SEASON_MATCHED, never instead of it.
    """
    name:                    str
    label:                   str
    excludes_baseline_year:  bool
    y0_kind:                 str = "season"          # "first-obs" | "season"

    def baseline(self, Y: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
        """The (N, 12) table of Y_0 values, indexed by calendar month of t."""
        if self.y0_kind == "first-obs":
            return first_observation_baseline(Y)
        if self.y0_kind == "season":
            return season_matched_baseline(Y, times)
        raise ValueError(f"unknown baseline kind {self.y0_kind!r}")


PAPER_LITERAL = Spec("paper-literal", "paper-literal $Y_0$ (first observation)",
                     excludes_baseline_year=False, y0_kind="first-obs")
SEASON_MATCHED = Spec("season-matched", "season-matched $Y_0$ ($Y_{1985,m}$)",
                      excludes_baseline_year=True, y0_kind="season")
AUTHORS_346 = Spec("authors-346",
                   "author-confirmed $Y_0$, all 346 transitions",
                   excludes_baseline_year=False, y0_kind="season")
# Insertion order is the column order in every table and figure: the two
# readings of Y_0 first, then the authors' own sample.  Names must not contain
# " / ", which run_replication.to_latex uses to split "<spec> / <instrument>".
SPECS = {s.name: s for s in (PAPER_LITERAL, SEASON_MATCHED, AUTHORS_346)}


def first_observation_baseline(Y: np.ndarray) -> np.ndarray:
    """
    Y_0 = each pixel's earliest valid LAI, held fixed for every t.

    Returned as an (N, 12) table with all twelve columns identical, so that
    callers can index it by calendar month uniformly regardless of spec.
    """
    out = np.empty(Y.shape[0])
    for i, row in enumerate(Y):
        valid = np.flatnonzero(~np.isnan(row))
        out[i] = row[valid[0]] if valid.size else 0.0
    return np.repeat(np.nan_to_num(out, nan=0.0)[:, None], 12, axis=1)


def season_matched_baseline(Y: np.ndarray, times: pd.DatetimeIndex,
                            baseline_year: Optional[int] = None) -> np.ndarray:
    """
    Y_0 matched to the calendar month of t: an (N, 12) table whose column m-1 is
    each pixel's baseline-year value for calendar month m.

    Why not one fixed cross-section.  W_t swings with the annual wind cycle, so
    pairing every month's network with a single January cross-section gives
    Delta(W_t Y_0) an annual cycle that is pure geometry evaluated off-season.
    In the real panel that pairing drives beta_1 to -0.0011; matching the season
    — forest still frozen in the baseline year, only the month aligned — gives
    +0.0049 and turns all twenty coefficients positive.  Averaging Y_0 over the
    baseline year or the whole sample changes nothing, so it is the seasonal
    pairing that matters, not noise in the level.

    The baseline year must be excluded from the estimation sample when this
    baseline is used; :class:`Spec` carries that constraint and explains why.
    """
    year = int(times.year.min()) if baseline_year is None else baseline_year
    sel = times.year == year
    if not sel.any():
        sel = np.ones(len(times), bool)
    # np.errstate does not cover this: an all-NaN slice raises a warnings-module
    # RuntimeWarning, not a floating-point error.  All-NaN pixel-months are
    # expected (a pixel with no retrieval in that month of the baseline year),
    # and the fallback handles them, so the warning is noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fallback = np.nanmean(Y[:, sel], axis=1)
        out = np.empty((Y.shape[0], 12))
        for m in range(1, 13):
            cols = sel & (times.month == m)
            col = np.nanmean(Y[:, cols], axis=1) if cols.any() else fallback
            out[:, m - 1] = np.where(np.isnan(col), fallback, col)
    return np.nan_to_num(out, nan=0.0)


# ── demo mode ───────────────────────────────────────────────────────────────

def make_demo_panel(n_lat: int = 10, n_lon: int = 10, T: int = 24, K: int = 5,
                    alpha: float = 0.22,
                    beta: Optional[np.ndarray] = None,
                    seed: int = 20260813,
                    resolution: float = 0.5,
                    seasonal_amp: float = 0.15) -> Panel:
    """
    Generate a balanced synthetic panel from the paper's own structural model.

    The point of the demo is not realism, it is *verifiability*: the data are
    generated by

        (I - sum_k beta_k W_t^[k]) Y_t = alpha Y_t-1 + seasonal_t + eta + eps_t

    solved for Y_t at every step, so W_t^[k] Y_t is genuinely simultaneous with
    Y_t — the reflection problem the paper's instrument exists to handle — and
    the coefficients that come out can be checked against the ones that went in.

    Ingredients that make the exercise non-trivial:
      * eta            pixel fixed effects, which is why first-differencing is
                       needed and why Delta Y_t-1 is endogenous (Nickell bias);
      * eps_t          a spatially smoothed common shock, which is what makes
                       FD-OLS biased and gives the instrument something to fix;
      * seasonal_t     an annual canopy cycle with a pixel-specific phase, so
                       the season-matched baseline is doing real work;
      * a rotating wind direction, so C_t and hence W_t^[k] vary month to month
        — that variation is the entire identifying signal.
    """
    rng = np.random.default_rng(seed)
    if beta is None:                                  # the paper's first K
        beta = np.array([0.0811, 0.0391, 0.0262, 0.0197, 0.0152])[:K]
    beta = np.asarray(beta, float)

    lats = -10.0 + resolution * np.arange(n_lat)
    lons = -70.0 + resolution * np.arange(n_lon)
    LON, LAT = np.meshgrid(lons, lats)
    grid_df = pd.DataFrame({
        "pixel_id":  np.arange(LAT.size, dtype=np.int64),
        "latitude":  LAT.ravel(),
        "longitude": LON.ravel(),
    })
    N = len(grid_df)
    xy = grid_df[["longitude", "latitude"]].to_numpy(np.float64)

    G = queen_adjacency(grid_df, resolution)
    shells = geodesic_shells(G, K)
    A = (G > 0).astype(np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()[:, None]
    smoother = (sp.identity(N, format="csr") + A) / (1.0 + deg)

    eta = rng.standard_normal(N)
    # Seasonal canopy cycle, pixel-specific phase.  Amplitude is deliberately
    # modest relative to the shock: the estimated equation has no time effects
    # (the paper omits them), so a deterministic seasonal is an omitted term
    # that Y_t-2 predicts, and it biases alpha upward in proportion to its
    # share of the variance.  At 0.15 that bias is small; at 0.75 — roughly the
    # shock's own scale — alpha comes out near 0.63 against a planted 0.22.
    # This is a real property of the specification, not an artefact of the demo.
    amp = seasonal_amp * (0.7 + 0.6 * rng.random(N))
    phase = 2 * np.pi * rng.random(N)
    times = pd.date_range("2000-01-01", periods=T, freq="MS")

    coords = (grid_df.set_index("pixel_id")
              .loc[np.sort(grid_df["pixel_id"]), ["latitude", "longitude"]]
              .to_numpy(np.float64))
    path_steps = np.arange(0.5, K + 1.5, 0.5) * resolution
    origin_row = np.repeat(np.arange(N), len(path_steps))

    def circulation(theta: float) -> sp.csr_matrix:
        """
        Trace a straight back-trajectory from every pixel and assign each point
        to its nearest cell — the same construction the real pipeline uses.

        A real trajectory is a LINE, not a cone, so it touches roughly ONE cell
        per geodesic shell; the real matrices have mean row sums near 1 at every
        k.  Sampling a cone instead would inflate the row sums and make them
        grow with k, which is precisely the geometry the paper's own
        coefficients rule out (see matrices.circulation_from_trajectories).
        A small per-pixel angular jitter keeps the paths from being exactly
        parallel.
        """
        jitter = np.deg2rad(6.0) * rng.standard_normal(N)
        ang = theta + jitter
        lats = coords[:, 0][:, None] + np.sin(ang)[:, None] * path_steps[None, :]
        lons = coords[:, 1][:, None] + np.cos(ang)[:, None] * path_steps[None, :]
        return circulation_from_trajectories(
            lats.ravel(), lons.ravel(), origin_row, coords)

    def step_state(Y_prev, step, month):
        theta = 2 * np.pi * ((step % 12) / 12.0) + 0.3 * rng.standard_normal()
        C = circulation(theta)
        W_k = upwind_matrices(C, shells, K)
        seasonal = amp * np.sin(2 * np.pi * (month - 1) / 12.0 + phase)
        eps = (smoother @ (0.5 * rng.standard_normal(N))
               + 0.3 * rng.standard_normal(N))
        M = sp.identity(N, format="csr")
        for k in range(1, K + 1):
            if beta[k - 1] != 0.0 and W_k[k].nnz:
                M = M - beta[k - 1] * W_k[k]
        return spsolve(M.tocsc(), alpha * Y_prev + 2.0 + seasonal + eta + eps), C

    # burn-in, then a PRE-SAMPLE baseline year, then the estimation sample.
    # The baseline must sit outside the sample: inside it, Y_0^m(t) is literally
    # Y_t for that year's transitions, so the instrument would coincide with the
    # regressor.  Harmless at T = 348, fatal at T = 24.
    Y = np.full(N, 4.5)
    for step in range(-24, 0):
        Y, _ = step_state(Y, step, (step % 12) + 1)

    Y0_cols = []
    for m in range(1, 13):
        Y, _ = step_state(Y, m - 1, m)
        Y0_cols.append(Y.copy())
    Y0_table = np.column_stack(Y0_cols)                 # (N, 12), pre-sample

    Y_cols, C_t = [], {}
    for step in range(T):
        month = (step % 12) + 1
        Y, C = step_state(Y, step, month)
        C_t[times[step]] = C
        Y_cols.append(Y.copy())

    return Panel(
        grid_df=grid_df, times=times, Y=np.column_stack(Y_cols), C_t=C_t, G=G,
        K=K, label=f"demo (synthetic, N={N}, T={T}, K={K})",
        truth={"alpha": alpha, "beta": beta}, Y0_table=Y0_table,
    )


# ── full mode ───────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "data" / "steps_4_6.pkl"


def full_mode_instructions(path: Path) -> str:
    return textwrap.dedent(f"""
    ------------------------------------------------------------------------
    FULL MODE NEEDS THE PANEL INPUT, AND IT IS NOT THERE.

        looked for: {path}

    That file is a pickle of (C_t_dict, G, panel_df) — the circulation
    matrices, the Queen adjacency and the LAI/wind panel.  Build it, in two
    steps that need nothing from anyone:

    1. FETCH THE TWO RAW INPUTS (monthly LAI and monthly 800 hPa winds).

         python run_replication.py --fetch-raw --what both --project <gcp-project>

       LAI comes from the NOAA CDR AVHRR collection through Earth Engine
       (`pip install earthengine-api`, then `earthengine authenticate`);
       winds come from the Copernicus CDS (`pip install cdsapi`, key in
       ~/.cdsapirc).  Costs and the two wind routes: src/fetch_raw.py.

    2. BUILD THE INPUT (grid -> panel -> trajectories -> C_t, G).

         python run_replication.py --build-inputs

       About five minutes for the full 348-month, 9,550-pixel domain.  The
       estimation domain itself ships with the package, so no GIS stack and no
       boundary download is needed.

    Check the machinery first, without any data at all:

         python run_replication.py --build-inputs --smoke   # all five stages
         python run_replication.py --demo                   # the estimator

    Or point at an input you already have:

         python run_replication.py --full --checkpoint /path/to/steps_4_6.pkl
    ------------------------------------------------------------------------
    """).strip()


def load_full(checkpoint: Path, K: int = 20, interpolate: bool = True) -> Panel:
    """Load a Step-4/6 checkpoint into a :class:`Panel`."""
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        print(full_mode_instructions(checkpoint))
        sys.exit(2)

    print(f"  reading {checkpoint} …", flush=True)
    with open(checkpoint, "rb") as fh:
        C_t_dict, G, panel_df = pickle.load(fh)

    wide = (panel_df["lai"].unstack("timestamp").sort_index(axis=1))
    order = np.sort(wide.index.to_numpy(np.int64))
    wide = wide.reindex(order)
    times = pd.DatetimeIndex(wide.columns)
    Y = wide.to_numpy(np.float64)

    if interpolate:
        Y, before, after = interpolate_lai_gaps(Y, times)
        if before:
            print(f"  LAI gap-fill: {before:,} NaN -> {after:,} "
                  f"({before - after:,} cells filled, within pixel along time)")

    grid_df = pd.DataFrame({"pixel_id": order,
                            "latitude": np.nan, "longitude": np.nan})
    return Panel(grid_df=grid_df, times=times, Y=Y, C_t=C_t_dict, G=G, K=K,
                 label=f"full (N={len(order):,}, T={len(times)}, K={K})")


def resolve_panel(mode: str, checkpoint: Optional[Path], K: int,
                  seed: int) -> Panel:
    if mode == "full":
        return load_full(checkpoint or DEFAULT_CHECKPOINT, K=K)
    return make_demo_panel(T=24, K=min(K, 5), seed=seed)
