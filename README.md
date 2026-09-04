# Replication of Araujo et al. (2023)

A standalone, self-contained replication of

> Araujo, R., Assunção, J., Hirota, M., Scheinkman, J. A. (2023).
> *Estimating the spatial amplification of damage caused by degradation in the
> Amazon.* **PNAS** 120(46), e2312451120.

The paper turns atmospheric moisture back-trajectories into a spatial
input–output multiplier

$$\Omega = \Big(I - \sum_{k=1}^{K}\beta_k \bar{W}^{[k]}\Big)^{-1}$$

quantifying how forest degradation at one Amazon pixel propagates, via "flying
rivers", to others. This package re-estimates the β_k and the multiplier they
imply, and reports where the replication succeeds and where it does not.

It is a **reimplementation from the paper's equations**, not a wrapper around the
research pipeline it was written alongside. That was deliberate: independently
reproducing the same numbers tests both, and in the event it caught a real bug in
this package and a sample defect in the pipeline.

It is also **self-contained**. Every input can be downloaded and rebuilt from
public sources by the code in `src/`; nothing has to be requested from anyone.

---

## Quickstart

```bash
pip install -r requirements.txt

python run_replication.py --demo                  # no data at all, ~2 seconds
python run_replication.py --build-inputs --smoke  # the build chain, ~1 minute
```

`--demo` is the default and needs no data, no credentials and no network: a
synthetic panel is generated in memory from the paper's own structural model with
coefficients you choose, and the estimator is asked to recover them.

For the real 1985–2013 Amazon panel, fetch the two monthly inputs and build:

```bash
python run_replication.py --fetch-raw --what both --project <gcp-project>
python run_replication.py --build-inputs          # ~5 min, 348 months
python run_replication.py --full                  # the estimation, ~4 min
```

Credentials and costs are in [`data/README.md`](data/README.md); the short
version is Earth Engine for LAI (~30 MB) and Copernicus CDS for winds (~210 MB).
The committed results in `outputs/` are estimated on exactly that rebuilt panel,
so they are reproducible from public sources alone. Against the parent research
pipeline's own run of the same specification they agree on everything that is
quoted: β₁ = 0.0049, α = 0.2201, Ω² mean 1.050, correlation 0.945, 20/20 positive,
3,302,224 observations on 9,544 clusters, minimum SW F 1.862e5; Σβ_k differs in
the fourth decimal (0.0223 against 0.0224). Stage-by-stage numbers are in
`data/README.md`.

```bash
python run_replication.py --full --checkpoint /path/to/steps_4_6.pkl
python run_replication.py --full --spec season-matched     # one specification only
python run_replication.py --full --spec authors-346        # the authors' own sample
```

`--full` writes to `outputs/`: `summary.txt`, `coefficients.csv`,
`table_s1_comparison.tex` (booktabs), and two figures; the committed copies are
from `--full` on the panel described under Reproducibility. `--demo` writes to
`outputs_demo/` instead, so a demo run cannot overwrite them. Figures need
matplotlib; without it the run completes and says so, which leaves any older
figures in place — regenerate or delete them rather than trusting them beside
fresh tables.

```
.
├── data/
│   ├── grid_amazon_0p25.csv    the 9,550-pixel estimation domain
│   ├── amazon_domain.geojson   the polygon it derives from
│   └── README.md               inputs, costs, licences, rebuild fidelity
├── outputs/                    committed tables and figures (--full)
├── src/
│   ├── fetch_raw.py            download monthly LAI (GEE) and winds (CDS)
│   ├── build_inputs.py         grid, panel, trajectories, C_t, G  (+ --smoke)
│   ├── data_loader.py          specifications, LAI gap-fill, synthetic generator
│   ├── matrices.py             W_t^[k] construction, time averages, Omega margins
│   └── estimation.py           FD-2SLS (thin QR + SVD), count/composition split
├── run_replication.py          single entry point for all four actions
├── requirements.txt            four packages, plus two optional for the rebuild
├── LICENSE                     MIT, with the data terms noted
└── README.md
```

---

## What is estimated: three specifications × two instruments

There are exactly six estimates and no hidden switches. Anything that would
change a coefficient is a **named specification that is always reported**, never
an option with a default.

**The authors settled both axes in August 2026**, in reply to a replication
query: W_t^[k]Y₀ is a binary sum over the upwind pixels, with no row
normalisation and no additional distance weighting, and with the own pixel
excluded; and Y₀ is fixed at the first year of the sample — 1985 in their
application — but matched by calendar month. Their construction is therefore
**season-matched Y₀ with the binary instrument**, and the other columns are
diagnostics against it.

**Axis 1 — how Y₀ is defined** in the instrument Δ(W_t^[k]Y₀), and which rows that
admits. These are different estimands rather than versions of one thing:

- **paper-literal** — each pixel's first observation, one fixed cross-section
  held for all t. Our earlier reading of "initial forest status", now known from
  the authors' reply to be wrong; kept as the price of a fixed cross-section.
- **season-matched** — the baseline year's value for the *same calendar month* as
  t; forest frozen in 1985, only the season aligned. The authors' construction.
  This column *excludes* the baseline year from the sample: inside it Y₀^{m(t)} is
  literally Y_t, so the instrument is identically the regressor (verified, max
  deviation exactly 0) and those rows behave like OLS.
- **authors-346** — the same Y₀ on all 346 transitions, i.e. with the baseline year
  retained. Table S1's 3,300,494 = 9,539 × 346 is the whole record, so the paper
  did not drop it; on that sample α comes out at 0.2201 against their published
  0.2200 and all twenty β_k are positive, as they report. It is a reconstruction of
  their procedure, baseline-year overlap included, reported beside the
  overlap-excluded column and never instead of it.

Whether the baseline year is in or out is part of each specification, not a flag.

**Axis 2 — the form of the instrument.** W_t^[k]Y is a *product* of the number of
grid cells the trajectory intersects at shell k and their mean LAI:

$$W_t^{[k]} Y = N_{it}^{[k]} \cdot \big(\tilde{W}_t^{[k]} Y\big)_i$$

- **binary** — Δ(W_t^[k]Y₀), as written and as the authors confirmed.
- **composition** — the exact (Bennet) split of the differenced regressor into a
  count part and a composition part, estimated separately:

$$\Delta\big(W_t^{[k]}Y\big) = (\Delta N)\,\bar m \;+\; \bar N\,(\Delta m),
  \qquad m = \tilde{W}_t^{[k]}Y$$

  with no interaction residual. Constraining the two coefficients equal returns
  the binary specification exactly, so this is a strict nesting test; the identity
  is asserted numerically before anything is estimated. Since the authors have
  confirmed the binary form, composition is a **stated deviation** from the paper
  rather than a candidate reading of it, and its β₁ must not be quoted as
  reproducing theirs.

---

## Results (full mode)

Against Table S1: β₁ = 0.0811, Σβ_k = 0.2578, α = 0.2200, Ω² mean 2.00.

| | β₁ | Σβ_k | α | Ω² mean | corr. Table S1 | positive | n |
|---|---|---|---|---|---|---|---|
| paper-literal / binary | −0.0011 | 0.0036 | 0.2223 | 1.007 | −0.337 | 15/20 | 3,302,224 |
| paper-literal / composition | 0.0037 | 0.0014 | 0.2221 | 1.004 | 0.630 | 7/20 | 3,302,224 |
| season-matched / binary | 0.0032 | 0.0129 | 0.2317 | 1.029 | **0.956** | 18/20 | 3,206,784 |
| season-matched / composition | 0.0908 | 0.1024 | 0.2158 | 1.237 | 0.901 | 12/20 | 3,206,784 |
| **authors-346 / binary** ← the authors' construction | **0.0049** | **0.0223** | **0.2201** | **1.050** | **0.945** | **20/20** | 3,302,224 |
| authors-346 / composition | 0.1469 | 0.1988 | 0.1932 | 1.530 | 0.914 | 17/20 | 3,302,224 |

### What matches

**The panel.** Table S1's 3,300,494 observations on 9,539 clusters is exactly
$9{,}539 \times 346$ — a balanced panel over 348 months less the two lost to
lagging — so the paper's LAI must have been gap-filled. Interpolating ours along
time within each pixel gives 3,302,224 on 9,544 clusters (our grid has 9,550
pixels; six have no valid retrieval anywhere).

**The persistence coefficient.** α = 0.2201 against a published 0.2200 on the
authors' own specification and sample; 0.1932 to 0.2317 across all six columns.

**The matrices.** Applying the paper's *own* β_k to our W̄^[k] reproduces its
reported multiplier distribution: Ω² mean 1.957, maximum 2.707, against 2.00 and
2.64. Mean row sums run 1.221 at k=1 to 0.851 at k=20 — about one cell per shell.

**The shape of the decay profile, on the authors' own construction.** Correlation
0.945 across the twenty lags, all twenty positive as they report, and 10.9% of the
total beyond k = 13 against their 9.9%. If anything it is slightly *too* spread:
8.42 effective contributing lags against their 6.71.

### What does not match

**The level of the spatial effect.** On the author-confirmed construction
β₁ = 0.0049 against 0.0811 and Σβ_k = 0.0223 against 0.2578 — factors of 16.5 and
11.6 — and the implied Ω² mean of 1.050 recovers 5% of their excess over unity.
The residual is a *level* gap at approximately correct shape, spread evenly over
the twenty shells rather than concentrated at any distance.

**The author-confirmed column is the weakest of the six on the multiplier.** 1.050
against their 2.00, where season-matched/composition reaches 1.237 and
authors-346/composition 1.530. Their β₁ is bracketed by the two instrument forms on
their own sample (0.0049 < 0.0811 < 0.1469); their Σβ_k is above both. The
composition instrument also buys β at α's expense — 0.1932 on their sample, 12%
below the published value and the worst α of the six — which is independent
evidence against it.

### Why the binary instrument attenuates

The count factor is shared between the regressor and the instrument, so it enters
Cov(z, x) without entering Cov(z, Δy); since β_IV = Cov(z, Δy)/Cov(z, x) it
inflates the denominator alone. At k = 1 it supplies **83%** of the binary
instrument's variance, which accounts for the ~28× gap between the two
instruments. Estimated separately, the composition coefficient is 0.0908
(t = 18.1) and the count coefficient is 0.0003 (t = 1.4); the restriction that
they are equal — which the binary specification imposes — is rejected at
χ²(20) = 472.6.

---

## What the demo verifies

The synthetic panel is generated by solving the structural equation at every step

$$\big(I - \textstyle\sum_k \beta_k W_t^{[k]}\big) Y_t
  = \alpha Y_{t-1} + \text{seasonal}_t + \eta_i + \varepsilon_t$$

so $W_t^{[k]} Y_t$ is genuinely simultaneous with $Y_t$ — Manski's reflection
problem, which is why the paper instruments it — and $\eta_i$ makes
$\Delta Y_{t-1}$ endogenous, which is the Nickell (1981) problem. The shock
carries a spatially smoothed common component, so FD-OLS is biased and the
instrument has something to fix. Back-trajectories are traced as straight lines
and assigned to their nearest cell, exactly as in the full pipeline, so each shell
holds about one cell.

A representative run ($N=100$, $T=24$, $K=5$, default `--seed 20260813`):

| | planted | recovered |
|---|---|---|
| $\alpha$ | 0.2200 | 0.2485 (SE 0.0366) |
| $\beta_1$ | 0.0811 | 0.0764 |
| $\sum\beta_k$ | 0.1813 | 0.1626 |

All specification/instrument combinations recover the planted values. A near-zero
$\sum\beta_k$ on real data is therefore a property of the data, not of this code.

Two honest notes. $\alpha$ is biased slightly upward because the estimated
equation has no time effects (the paper omits them) while the data-generating
process has a deterministic seasonal cycle that $Y_{t-2}$ predicts; the bias grows
with `seasonal_amp` and reaches ≈0.63 at an amplitude comparable to the shock.
And the demo's baseline year is drawn from a *pre-sample* period, so the
instrument-equals-regressor overlap described above cannot arise — at $T=24$ it
would be half the sample and would make $Z$ singular. One consequence, printed
rather than left implicit: in `--demo` the `season-matched` and `authors-346`
columns are numerically identical, since they differ only in what they do about an
overlap the synthetic panel does not have.

---

## Method notes

**FD-2SLS.** Pixel fixed effects are removed by first differencing (the paper's
Eq. 4); ΔY_{t−1} is instrumented by Y_{t−2} and the spatial lags by Δ(W_t^[k]Y₀).
The system is exactly identified, so each coefficient is pinned by one
instrument's worth of independent variation — which is why the
Sanderson-Windmeijer numerator degrees of freedom is 1, and why the naive
per-variable partial F is *not* a valid weak-instrument statistic here.

**Numerics.** Projections use a thin QR, $\hat X = QQ'X$: no $(Z'Z)^{-1}$ and no
$n\times n$ projector is ever formed. The second stage is solved by SVD with an
explicit rank cutoff. Standard errors are cluster-robust at the pixel level, and
residuals are formed with the *actual* $X$, not $\hat X$ — the usual manual-2SLS
trap, which would understate the variance.

**Ω versus Ω².** The paper defines its indices of influence and exposure on
$\Omega^{\Delta+1}$ and maps $\Delta=1$, so its "average multiplier effect is 2"
and its ~2.64 maximum are **Ω²** quantities. Comparing Ω's own margins against
2.0 is the wrong benchmark, and the error is invisible while $\Omega\approx I$.
Margins are computed by repeated matrix-vector products on the Neumann series, so
Ω is never densified.

**Weight matrices are binary, never row-standardised.** W_t^[k]Y is therefore a
*sum* over upwind cells. Row-standardising the matrices that enter Ω does not
reproduce the paper's multiplier from its own coefficients (Ω² mean 1.72 against
its 2.00, versus 1.96 binary), so the binary reading is the one its coefficients
belong to. Row-standardisation appears only in the *instrument*, as the
composition channel.

---

## Reproducibility

Demo mode is deterministic given `--seed`. Figures use a categorical palette
validated for colour-vision deficiency (worst all-pairs ΔE 9.2, normal-vision
24.0) and encode series identity by marker and dash pattern as well as hue, so
they survive greyscale printing.

The committed full-mode results are estimated on: monthly, 0.25°, 1985–2013, NOAA
CDR AVHRR LAI with cloud gaps interpolated, 5-day 800 hPa back-trajectories,
K = 20, on the 9,550-pixel Amazon domain in `data/`. `--fetch-raw` plus
`--build-inputs` reconstruct that panel; `data/README.md` reports how closely,
with numbers, and names the two construction choices that are yours rather than
defaulted.

One known construction difference from the paper remains: the trajectories are
effectively **2-D isobaric**, because the wind input carries a single pressure
level, while the paper says "kinematic". This was the leading candidate while the
residual looked like a *shape* difference, since 3-D advection turns and lengthens
paths through vertical shear without inflating row sums. It has since been
measured on a 2×2 pilot over cadence × dimensionality and is inert: at monthly
cadence 3-D moves shell mass by 0.3% and upwind composition by 0.1%, and at
sub-daily cadence, where the vertical channel is demonstrably live, the network is
almost entirely rewired (2–12% link overlap) yet delivers the same upwind LAI to
within 2% at every shell. On the author-confirmed specification the residual is a
level difference at approximately correct shape, which vertical shear would not
produce.

## Citation

Please cite the original paper:

> Araujo, R., Assunção, J., Hirota, M., & Scheinkman, J. A. (2023). Estimating
> the spatial amplification of damage caused by degradation in the Amazon.
> *Proceedings of the National Academy of Sciences*, 120(46), e2312451120.

This package is MIT-licensed (see `LICENSE`, which also records the terms of the
data it downloads and of the boundary data it redistributes). If you use the
replication itself, cite it as: L. Becker, *Standalone replication of Araujo et
al. (2023)*, Potsdam Institute for Climate Impact Research, 2026.
