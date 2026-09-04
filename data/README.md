# Data

Three small artefacts ship here. Everything else is downloaded or rebuilt, and
nothing has to be requested from anyone.

| file | size | what it is |
|---|---|---|
| `grid_amazon_0p25.csv` | 224 KB | the 9,550 pixel centres of the estimation domain — the grid behind every published figure in this package |
| `amazon_domain.geojson` | 824 KB | the domain polygon those cells came from, so the grid can be re-derived rather than trusted |
| `README.md` | — | this file |

## Demo mode needs nothing

```bash
python run_replication.py --demo
```

The panel is generated in memory by `src/data_loader.make_demo_panel` from the
paper's own structural model, with parameters the caller chooses. Nothing is read
from disk, and results go to `outputs_demo/` so they cannot overwrite the
committed full-mode ones.

## Full mode: two downloads, then one build

```bash
# 1. the raw inputs (see src/fetch_raw.py for credentials and cost)
python run_replication.py --fetch-raw --what both --project <gcp-project>

# 2. panel, trajectories, circulation matrices  (~5 min, 348 months)
python run_replication.py --build-inputs

# 3. the estimation
python run_replication.py --full
```

Step 1 writes two files, and they are the only large inputs:

| file | size | source | terms |
|---|---|---|---|
| `lai_monthly_0p25_avhrr.nc` | ~30 MB | NOAA CDR AVHRR LAI/FAPAR V5, via Google Earth Engine | free, registration required |
| `era5_monthly_0p25.nc` | ~210 MB | ERA5 800 hPa u/v/ω, via Copernicus CDS | free, registration required; Copernicus licence |

Step 2 writes `steps_4_6.pkl` (~0.8 GB), a pickle of `(C_t_dict, G, panel_df)`:

| object | what it is |
|---|---|
| `C_t_dict` | `{Timestamp: csr_matrix}` — which pixel pairs a 5-day back-trajectory connects, per month |
| `G` | Queen's-contiguity adjacency over the domain |
| `panel_df` | MultiIndex `[pixel_id, timestamp]`, columns `[lai, u, v, omega]` |

Check the build machinery before spending anything on downloads:

```bash
python run_replication.py --build-inputs --smoke     # synthetic fields, ~1 min
```

It runs all five stages and asserts every contract — binary C_t, zero diagonal,
no surviving bidirectional link, a balanced gap-filled panel, and shells that
actually intersect the circulation.

## How faithful the rebuild is

The builder is a reimplementation of the research pipeline this package was
written alongside, so the agreement is measured rather than assumed. The results
committed in `outputs/` are estimated on the builder's own 348-month panel — that
is, on an input reproducible from public sources alone — and they agree with the
research pipeline's run of the same specification:

| quantity | research pipeline | this package (`outputs/`) |
|---|---|---|
| β₁, authors-346 / binary | 0.0049 | **0.0049** |
| Σβ_k | 0.0224 | 0.0223 |
| α | 0.2201 | **0.2201** |
| Ω² mean | 1.050 | **1.050** |
| correlation with Table S1 | 0.945 | **0.945** |
| positive coefficients | 20/20 | **20/20** |
| n, clusters | 3,302,224 on 9,544 | **3,302,224 on 9,544** |
| min Sanderson-Windmeijer F | 1.862e5 | **1.862e5** |

Stage by stage, against the original run's own artefact:

| stage | agreement |
|---|---|
| `panel_df` — LAI and all three wind columns | **bit-identical**, max abs difference 0, same 6 pixels dropped |
| `G`, the Queen adjacency | **identical** — 74,762 links |
| `C_t` restricted to shells 1–20, i.e. what the estimation sees | Jaccard 0.9996 |
| mean row sums of W^[k], k = 1…20 | ratio 1.0000–1.0004, total mass ×1.0002 |
| `C_t` including all links | Jaccard 0.973–0.977 |

The 2–3% of `C_t` that differs lies entirely beyond k = 20 — trajectory points
too far from their origin to enter any W^[k]. What remains inside the shells is
worth **1.2e-4** at most across all 120 coefficients (`--build-inputs --verify`
prints the table), and that worst case is a composition coefficient at k = 16;
the binary specifications, which are the paper's, stay within 3.0e-5.

One detail is worth knowing before changing the sampling code, because it took a
6% deviation in β₁ to find: **the LAI grid is offset from the pixel lattice by
half a cell** (LAI centres at −55.875°, pixels at −56.000°), so every pixel sits
exactly equidistant from four LAI cells and "nearest cell" is a four-way tie.
Which of the four wins depends on the distance metric — unit-sphere, as here and
in the original run, or degree-space — and the difference is not cosmetic: it
moved α by 1.3%, β₁ by 6%, and the sign count from 20/20 to 19/20. Averaging the
four cells would be the defensible alternative, and it is not what the published
panel did.

## Two choices that change the output

Neither is a silent default.

**`--wind-field`** decides what a parcel sees over the ocean. `era5-native`
(default) uses the wind file as it is. `panel-fill` restricts the field to land
and refills the rest by a row/column-mean cascade, which is what the original run
did as a side effect of carrying winds on a land-masked pixel panel. Measured on
those same three months, the two agree to within 0.001 on every shell's row sums:
an over-ocean point can create no entry in `C_t`, because there is no pixel there
to hit.

**`--wind-source`** (in `--fetch-raw`) decides how the monthly wind field is
made. `hourly` averages hourly analyses per calendar month, as the original run
did: faithful, ~30 GB, a day or more of CDS queueing. `monthly-means` takes
ECMWF's own product in a few requests, ~200 MB — and deviates from the hourly
average by up to 0.63% of the field's standard deviation at 800 hPa. Small, not
zero, and its effect on β_k has not been measured. Say which one you used.

## The estimation domain

The paper names no shapefile. `grid_amazon_0p25.csv` holds the domain used here:
0.25° cells whose centre falls inside the union of the HydroBASINS level-3 Amazon
watershed and the RESOLVE 2017 Amazonia ecoregions, intersected with a Natural
Earth land mask — 9,550 pixels, against the 9,539 clusters of the paper's
Table S1 (+0.1%). Either boundary alone gives ~7,740, which is 19% short.

`--build-inputs --domain-from-polygon` re-derives the grid from
`amazon_domain.geojson` instead of reading the CSV. That route needs no land-cover
download and returns the same 9,550 cells **plus 14** coastal cells around the
Amazon estuary that the land mask removes; the difference is reported, not
hidden. Those cells carry no upwind neighbours, so the coefficients do not turn
on them.

`pixel_id` is the index into the continental 0.25° lattice
(lat −56…13, lon −82…−34) and is *preserved, not renumbered*, which is why the
values are not contiguous.

## Two data facts that matter for interpretation

- **The LAI record stops at 2013 on purpose.** The AVHRR→VIIRS handover at
  2014-01 is a 26% multiplicative break, despite the two collections sharing a
  band name, a grid and a scale factor. 1985–2013 is also exactly the paper's
  period. A residual ~5% NOAA-14→NOAA-16 break at 2000→2001 is accepted.
- **Cloud-masked LAI is gap-filled, not zero-filled.** W^[k] is binary, so
  W^[k]Y is a *sum* over upwind neighbours: a missing neighbour passed through as
  zero would enter as bare ground. Filling is within pixel and along time only,
  so it introduces no spatial information. See
  `src/data_loader.interpolate_lai_gaps`.

## Attribution

HydroBASINS v1c — WWF / HydroSHEDS, free for non-commercial use with
attribution. RESOLVE Ecoregions 2017 — CC BY 4.0. Both are redistributed here in
derived form, as the domain polygon and grid. ERA5 — Copernicus Climate Change
Service; NOAA CDR AVHRR LAI/FAPAR — NOAA Climate Data Record. Neither is
redistributed; both are downloaded from source. See `../LICENSE`.
