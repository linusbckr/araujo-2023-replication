"""
Rebuild the full-mode input from monthly LAI and wind fields.
=============================================================

This module is the standalone replacement for the 1.5 GB checkpoint that full
mode used to require.  Given two NetCDF files — monthly LAI and monthly 800 hPa
winds, both on a 0.25 deg grid over South America — it reconstructs everything
the estimation needs and writes it as one pickle:

    (C_t_dict, G, panel_df)

    C_t_dict : {Timestamp: csr_matrix}  circulation matrix per month
    G        : csr_matrix               Queen's-contiguity adjacency
    panel_df : DataFrame                MultiIndex [pixel_id, timestamp],
                                        columns [lai, u, v, omega]

``src/fetch_raw.py`` produces the two NetCDF inputs from their public sources.
The two together mean the package depends on no artefact it cannot rebuild.

The five stages, and where each number comes from
------------------------------------------------
1. **Domain.**  The paper names no shapefile.  ``data/grid_amazon_0p25.csv``
   holds the 9,550 pixel centres used for every published figure here: the
   0.25 deg cells whose centre falls inside the union of the HydroBASINS level-3
   Amazon watershed and the RESOLVE 2017 Amazonia ecoregions, intersected with a
   Natural Earth land mask.  ``--domain-from-polygon`` re-derives it from the
   shipped ``data/amazon_domain.geojson`` instead, which reproduces the CSV plus
   14 coastal cells around the Amazon estuary that the land mask removes (the
   polygon route needs no land-cover download, which is why it is offered; the
   14-cell difference is reported rather than hidden).  Either way ``pixel_id``
   is *preserved, not renumbered*, matching the convention in ``matrices``.

2. **Panel.**  LAI and (u, v, omega) are sampled at each pixel centre by
   nearest-cell lookup on each file's own grid — the two grids are offset by half
   a cell (LAI centres at -55.875, wind at -56.000), which is why this is a
   lookup and not an array slice.  Cloud gaps in LAI are then filled within
   pixel along time; see ``data_loader.interpolate_lai_gaps`` for why a NaN left
   in place would enter the spatial sums as bare ground.

3. **Trajectories.**  Every pixel-month launches a parcel at 800 hPa and
   integrates it backward for ``--hours-back`` one-hour Euler steps through the
   wind field, with linear interpolation in time between the bracketing monthly
   snapshots and bilinear interpolation in space.  Displacements are geodetic:

       dlat = v * 3600 / R * (180/pi)
       dlon = u * 3600 / (R cos(lat)) * (180/pi)
       dp   = omega * 3600 / 100                       [Pa/s -> hPa/hr]

   A parcel leaving the wind domain, or leaving the 100-950 hPa band, is
   truncated there; its recorded points up to that step are kept.  All parcels of
   one origin month are stepped together as arrays, which is what makes a rebuild
   minutes rather than the day the equivalent per-trajectory loop takes.

4. **Circulation.**  C_t[i, j] = 1 iff pixel i's parcel passed through pixel j's
   Voronoi cell in month t — nearest-cell assignment within
   ``VORONOI_FACTOR * resolution``, no buffer around the path.  Bidirectional
   pairs (w_ij and w_ji both set) are resolved to the upper-triangular direction,
   as the paper's unidirectionality assumption requires; the fraction is reported
   because the paper quotes ~0.003% for its own wind field.

5. **Geography.**  G is Queen's contiguity over the domain, from
   ``matrices.queen_adjacency``: the same function the estimation uses, so the
   shells and the matrices cannot disagree about what a neighbour is.

Two choices that change the output, both named rather than defaulted
-------------------------------------------------------------------
``--wind-field`` decides what the parcel sees over the ocean.  ``era5-native``
uses the wind file as it is.  ``panel-fill`` first restricts the field to land
cells and refills the rest by a row/column-mean cascade, which is what the
published run did as a side effect of carrying winds on a land-masked pixel
panel.  The difference is confined to parcels that cross the coast and return,
because an over-ocean point can create no entry in C_t — there is no pixel
there to hit — but it is a real difference and it is not silently chosen.

``--land-mask`` decides how ``panel-fill`` identifies land: ``lai-valid`` (a
cell with at least one finite LAI observation) needs no extra download and is
the default when it is needed at all.

What this reproduces, and what has not been checked
---------------------------------------------------
The algorithms above are those of the run that produced ``outputs/``, and the
contracts are asserted at every stage boundary (``--smoke`` exercises all five
on synthetic fields in about a minute).  What has *not* been done is a full
rebuild from raw sources followed by a coefficient-by-coefficient comparison
against ``outputs/summary.txt``: that needs the two downloads and several hours.
``--verify`` performs exactly that comparison if you run it, and prints the
per-coefficient deviation instead of asserting a tolerance nobody has measured.
Treat a rebuild as reproducing the published construction, not as certified
bit-identical, until that comparison has been run once.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from .matrices import VORONOI_FACTOR, _binarize, queen_adjacency

log = logging.getLogger("replication.build_inputs")

# ── Physical and integration constants (as in the published run) ─────────────

EARTH_RADIUS_M = 6_371_000.0
SECONDS_PER_HOUR = 3_600.0
P_INIT_HPA = 800.0        # launch level
P_MIN_HPA = 100.0         # tropopause ceiling: above this the parcel is dropped
P_MAX_HPA = 950.0         # boundary-layer floor
DEG = np.pi / 180.0

DEFAULT_BBOX = (-56.0, 13.0, -82.0, -34.0)   # lat_min, lat_max, lon_min, lon_max
DEFAULT_RESOLUTION = 0.25
DEFAULT_HOURS_BACK = 120                     # 5 days

_DATA = Path(__file__).resolve().parent.parent / "data"
GRID_CSV = _DATA / "grid_amazon_0p25.csv"
DOMAIN_GEOJSON = _DATA / "amazon_domain.geojson"

# Variable-name aliases: the same field is called different things by CDS
# vintage, by ARCO mirrors and by the two NOAA CDR collections.
_ALIASES = {
    "lai": ("lai", "LAI", "Lai", "leaf_area_index"),
    "u": ("u", "u_component_of_wind", "eastward_wind", "U"),
    "v": ("v", "v_component_of_wind", "northward_wind", "V"),
    "omega": ("w", "omega", "vertical_velocity", "lagrangian_tendency_of_air_pressure"),
}
_TIME_ALIASES = ("time", "valid_time", "date", "month")
_LAT_ALIASES = ("latitude", "lat", "y")
_LON_ALIASES = ("longitude", "lon", "x")
_LEV_ALIASES = ("pressure_level", "level", "plev", "isobaricInhPa")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — the estimation domain
# ─────────────────────────────────────────────────────────────────────────────

def load_domain_grid(path: Path = GRID_CSV) -> pd.DataFrame:
    """The 9,550 pixel centres behind every published figure in this package."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It ships with the package; re-clone, or derive "
            f"it with --domain-from-polygon."
        )
    grid = pd.read_csv(path)
    missing = {"pixel_id", "latitude", "longitude"} - set(grid.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    return grid.sort_values("pixel_id").reset_index(drop=True)


def _polygon_mask(pts_lon_lat: np.ndarray, geojson: Path) -> np.ndarray:
    """Point-in-polygon over a (Multi)Polygon GeoJSON, holes honoured.

    Uses ``matplotlib.path`` rather than shapely/geopandas: matplotlib is
    already a dependency for the figures, and a replication package should not
    need a GIS stack to say which cells are in the domain.
    """
    from matplotlib.path import Path as MPath

    geom = json.loads(geojson.read_text(encoding="utf-8"))
    if geom.get("type") == "FeatureCollection":
        polys: List[list] = []
        for feat in geom["features"]:
            g = feat["geometry"]
            if g["type"] == "Polygon":
                polys.append(g["coordinates"])
            elif g["type"] == "MultiPolygon":
                polys.extend(g["coordinates"])
    elif geom.get("type") == "MultiPolygon":
        polys = list(geom["coordinates"])
    elif geom.get("type") == "Polygon":
        polys = [geom["coordinates"]]
    else:
        raise ValueError(f"unsupported GeoJSON type {geom.get('type')!r}")

    inside = np.zeros(len(pts_lon_lat), dtype=bool)
    for rings in polys:
        m = MPath(np.asarray(rings[0], dtype=float)).contains_points(pts_lon_lat)
        for hole in rings[1:]:
            m &= ~MPath(np.asarray(hole, dtype=float)).contains_points(pts_lon_lat)
        inside |= m
    return inside


def continental_lattice(bbox=DEFAULT_BBOX, resolution=DEFAULT_RESOLUTION
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """The 1-D lattice the wind domain and every pixel_id are defined on."""
    lat_min, lat_max, lon_min, lon_max = bbox
    half = resolution / 2.0
    return (np.arange(lat_min, lat_max + half, resolution),
            np.arange(lon_min, lon_max + half, resolution))


def derive_domain_grid(geojson: Path = DOMAIN_GEOJSON, bbox=DEFAULT_BBOX,
                       resolution: float = DEFAULT_RESOLUTION,
                       compare_to: Optional[Path] = GRID_CSV) -> pd.DataFrame:
    """Re-derive the domain from the polygon, and report the difference.

    ``pixel_id`` is the index into the continental lattice, exactly as in the
    published run, so ids are non-contiguous and comparable across domains.
    """
    lats, lons = continental_lattice(bbox, resolution)
    lon2, lat2 = np.meshgrid(lons, lats)
    pts = np.column_stack([lon2.ravel(), lat2.ravel()])
    inside = _polygon_mask(pts, geojson)

    grid = pd.DataFrame({
        "pixel_id": np.flatnonzero(inside).astype(np.int64),
        "latitude": lat2.ravel()[inside],
        "longitude": lon2.ravel()[inside],
    })
    log.info("domain from polygon: %d cells inside %s", len(grid), geojson.name)

    if compare_to is not None and Path(compare_to).exists():
        ref = pd.read_csv(compare_to)
        mine = set(map(tuple, np.round(grid[["latitude", "longitude"]].to_numpy(), 4)))
        theirs = set(map(tuple, np.round(ref[["latitude", "longitude"]].to_numpy(), 4)))
        log.info(
            "  vs %s: %d shipped, %d derived, %d derived-only, %d shipped-only",
            Path(compare_to).name, len(theirs), len(mine),
            len(mine - theirs), len(theirs - mine),
        )
        if mine - theirs:
            log.info(
                "  the derived-only cells are coastal: the published grid also "
                "passed a Natural Earth land mask, which this route does not. "
                "Coefficients are not sensitive to them (they carry no upwind "
                "neighbours), but the two domains are not identical."
            )
    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — the panel
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(ds, role: str) -> str:
    names = _ALIASES[role] if role in _ALIASES else ()
    for cand in names:
        if cand in ds.variables:
            return cand
    raise KeyError(
        f"no variable for {role!r} in {list(ds.data_vars)}; extend _ALIASES "
        f"rather than renaming the file"
    )


def _resolve_coord(ds, aliases: Sequence[str]) -> str:
    for cand in aliases:
        if cand in ds.coords or cand in ds.dims:
            return cand
    raise KeyError(f"none of {aliases} in {list(ds.coords)}")


def _open(path: Path):
    import xarray as xr
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found — see data/README.md")
    return xr.open_dataset(path)


def _to_cartesian(lat_deg, lon_deg) -> np.ndarray:
    """Unit-sphere (x, y, z).  Euclidean distance there is monotone in
    great-circle distance, so the nearest neighbour is the same one."""
    lat_r, lon_r = np.deg2rad(np.ravel(lat_deg)), np.deg2rad(np.ravel(lon_deg))
    return np.column_stack([np.cos(lat_r) * np.cos(lon_r),
                            np.cos(lat_r) * np.sin(lon_r),
                            np.sin(lat_r)])


def _nearest_cell(target_lat, target_lon, src_lat_1d, src_lon_1d,
                  max_distance_deg: float = 1.0):
    """Nearest source cell per target pixel, on the unit sphere.

    Spherical rather than degree-space, because that is what the published run
    used and the choice is not cosmetic: **the LAI grid is offset from the pixel
    lattice by half a cell** (LAI centres at -55.875, pixels at -56.000), so
    every pixel is exactly equidistant from four LAI cells and "nearest" is a
    four-way tie.  Which one wins is decided by the metric and by cKDTree's
    index order, and it changes the LAI a pixel is assigned.  Reproducing the
    published panel therefore means reproducing the tie-break, not merely the
    intent.  Averaging the four would be the defensible alternative and would
    change the estimates; it is not what the published run did.

    A pixel whose nearest source cell is farther than ``max_distance_deg`` is
    flagged: the field does not cover it, and that must surface as NaN rather
    than as the value of some distant cell.
    """
    lon2, lat2 = np.meshgrid(src_lon_1d, src_lat_1d)
    tree = cKDTree(_to_cartesian(lat2, lon2))
    dist, flat = tree.query(_to_cartesian(target_lat, target_lon), k=1)
    rows, cols = np.unravel_index(flat, lat2.shape)
    # Chord length on the unit sphere -> degrees, for the coverage guard only.
    deg = np.degrees(2.0 * np.arcsin(np.clip(dist / 2.0, 0.0, 1.0)))
    return rows, cols, deg > max_distance_deg


def build_panel(grid: pd.DataFrame, lai_nc: Path, wind_nc: Path,
                interpolate_lai: bool = True) -> pd.DataFrame:
    """MultiIndex [pixel_id, timestamp] panel of [lai, u, v, omega]."""
    ds_lai, ds_wind = _open(lai_nc), _open(wind_nc)
    try:
        v_lai = _resolve(ds_lai, "lai")
        v_u, v_v, v_w = (_resolve(ds_wind, r) for r in ("u", "v", "omega"))

        t_lai = _resolve_coord(ds_lai, _TIME_ALIASES)
        t_wind = _resolve_coord(ds_wind, _TIME_ALIASES)
        times = pd.DatetimeIndex(
            np.intersect1d(ds_lai[t_lai].values, ds_wind[t_wind].values))
        if len(times) == 0:
            raise ValueError("LAI and wind time axes do not intersect")
        log.info("panel: %d shared months, %s … %s",
                 len(times), times[0].date(), times[-1].date())

        ds_lai = ds_lai.sel({t_lai: times})
        ds_wind = ds_wind.sel({t_wind: times})

        lev = next((c for c in _LEV_ALIASES if c in ds_wind.coords), None)
        if lev is not None and ds_wind.sizes.get(lev, 1) > 1:
            levels = np.asarray(ds_wind[lev].values, dtype=float)
            levels = levels / 100.0 if levels.max() > 2000.0 else levels
            ds_wind = ds_wind.isel({lev: int(np.abs(levels - P_INIT_HPA).argmin())})
        elif lev is not None:
            ds_wind = ds_wind.squeeze(lev, drop=True)

        plat = grid["latitude"].to_numpy(float)
        plon = grid["longitude"].to_numpy(float)

        out = {}
        for label, ds, var in (("lai", ds_lai, v_lai), ("u", ds_wind, v_u),
                               ("v", ds_wind, v_v), ("omega", ds_wind, v_w)):
            la = _resolve_coord(ds, _LAT_ALIASES)
            lo = _resolve_coord(ds, _LON_ALIASES)
            rows, cols, bad = _nearest_cell(plat, plon, ds[la].values, ds[lo].values)
            arr = np.asarray(ds[var].values)[:, rows, cols].astype(np.float32)
            if bad.any():
                arr[:, bad] = np.nan
                log.warning("%s: %d pixels beyond 1 deg of any cell -> NaN",
                            label, int(bad.sum()))
            out[label] = arr                      # (T, N)

        pixel_ids = grid["pixel_id"].to_numpy(np.int64)
        index = pd.MultiIndex.from_arrays(
            [np.repeat(pixel_ids, len(times)), np.tile(times.values, len(grid))],
            names=["pixel_id", "timestamp"])
        panel = pd.DataFrame({k: v.T.ravel() for k, v in out.items()}, index=index)
    finally:
        ds_lai.close()
        ds_wind.close()

    if interpolate_lai:
        n_before = int(panel["lai"].isna().sum())
        wide = panel["lai"].unstack("timestamp").sort_index(axis=1)
        filled = wide.interpolate(method="time", axis=1, limit_direction="both")
        panel["lai"] = (filled.stack(future_stack=True)
                        .reindex(panel.index).astype(panel["lai"].dtype))
        n_after = int(panel["lai"].isna().sum())
        log.info("LAI gap-fill: %d NaN -> %d (%d cells filled, within pixel "
                 "along time)", n_before, n_after, n_before - n_after)
        if n_after:
            log.warning("%d LAI cells still NaN: pixels with no valid retrieval "
                        "anywhere. Their rows drop out of the estimation.", n_after)
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — back-trajectories
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WindField:
    """Monthly 800 hPa (u, v, omega) on a regular lattice, ready to interpolate."""
    times: pd.DatetimeIndex
    lats: np.ndarray
    lons: np.ndarray
    u: np.ndarray        # (T, n_lat, n_lon)
    v: np.ndarray
    w: np.ndarray

    def interpolators(self, t_idx: int):
        grid = (self.lats, self.lons)
        return tuple(
            RegularGridInterpolator(grid, arr[t_idx].astype(np.float64),
                                    method="linear", bounds_error=False,
                                    fill_value=np.nan)
            for arr in (self.u, self.v, self.w))


def _row_col_mean_fill(field: np.ndarray) -> np.ndarray:
    """Row-mean then column-mean cascade, as the published run's panel did.

    This is what a land-masked wind panel does to its ocean cells: they arrive
    as NaN and leave as the mean of their row, then of their column.  Kept
    because reproducing the published construction is the point; see the module
    docstring for why it barely reaches C_t.
    """
    if not np.isnan(field).any():
        return field
    out = field.copy()
    for axis in (1, 0):
        means = np.nanmean(out, axis=axis)
        nan = np.isnan(out)
        if not nan.any():
            break
        idx = np.where(nan)[0] if axis == 1 else np.where(nan)[1]
        out[nan] = np.take(means, idx)
    out[np.isnan(out)] = 0.0
    return out


def load_wind_field(wind_nc: Path, times: pd.DatetimeIndex, mode: str,
                    lai_nc: Optional[Path] = None,
                    bbox=DEFAULT_BBOX) -> WindField:
    """Read the wind file onto its lattice under the chosen ``--wind-field``."""
    if mode not in ("era5-native", "panel-fill"):
        raise ValueError("--wind-field must be era5-native or panel-fill")
    ds = _open(wind_nc)
    try:
        t = _resolve_coord(ds, _TIME_ALIASES)
        la, lo = _resolve_coord(ds, _LAT_ALIASES), _resolve_coord(ds, _LON_ALIASES)
        lev = next((c for c in _LEV_ALIASES if c in ds.coords), None)
        if lev is not None:
            if ds.sizes.get(lev, 1) > 1:
                levels = np.asarray(ds[lev].values, dtype=float)
                levels = levels / 100.0 if levels.max() > 2000.0 else levels
                ds = ds.isel({lev: int(np.abs(levels - P_INIT_HPA).argmin())})
            else:
                ds = ds.squeeze(lev, drop=True)
        ds = ds.sel({t: times})
        lats = np.asarray(ds[la].values, float)
        lons = np.asarray(ds[lo].values, float)
        arrays = [np.asarray(ds[_resolve(ds, r)].values, np.float32)
                  for r in ("u", "v", "omega")]
    finally:
        ds.close()

    if lats[0] > lats[-1]:          # RegularGridInterpolator needs ascending axes
        lats = lats[::-1]
        arrays = [a[:, ::-1, :] for a in arrays]

    if mode == "panel-fill":
        land = _land_mask_from_lai(lai_nc, lats, lons) if lai_nc else None
        if land is None:
            raise ValueError("--wind-field panel-fill needs the LAI file to "
                             "identify land cells (--land-mask lai-valid)")
        log.info("panel-fill: %d/%d cells land; the rest refilled by a "
                 "row/column-mean cascade", int(land.sum()), land.size)
        for i, a in enumerate(arrays):
            filled = np.empty_like(a)
            for k in range(a.shape[0]):
                snap = a[k].astype(np.float64)
                snap[~land] = np.nan
                filled[k] = _row_col_mean_fill(snap)
            arrays[i] = filled

    return WindField(times=times, lats=lats, lons=lons,
                     u=arrays[0], v=arrays[1], w=arrays[2])


def _land_mask_from_lai(lai_nc: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Land = a cell with at least one finite LAI observation, on the wind lattice."""
    ds = _open(lai_nc)
    try:
        la, lo = _resolve_coord(ds, _LAT_ALIASES), _resolve_coord(ds, _LON_ALIASES)
        valid = np.isfinite(np.asarray(ds[_resolve(ds, "lai")].values)).any(axis=0)
        src_lat = np.asarray(ds[la].values, float)
        src_lon = np.asarray(ds[lo].values, float)
    finally:
        ds.close()
    lon2, lat2 = np.meshgrid(lons, lats)
    rows, cols, _ = _nearest_cell(lat2.ravel(), lon2.ravel(), src_lat, src_lon)
    return valid[rows, cols].reshape(lat2.shape)


def trajectories_for_month(wind: WindField, t_idx: int, origin_lat: np.ndarray,
                           origin_lon: np.ndarray, hours_back: int
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step every parcel of one origin month backward, together.

    Returns (lat, lon, origin_index) of every *valid* recorded point.  A parcel
    is dropped from the integration at the step where it leaves the wind domain,
    leaves the 100-950 hPa band, or meets a NaN wind; points recorded before
    that are kept, which is the published run's truncation rule.
    """
    n = len(origin_lat)
    lat = origin_lat.astype(np.float64).copy()
    lon = origin_lon.astype(np.float64).copy()
    pres = np.full(n, P_INIT_HPA)
    alive = np.ones(n, bool)
    origin_idx = np.arange(n)

    lat_min, lat_max = wind.lats[0], wind.lats[-1]
    lon_min, lon_max = wind.lons[0], wind.lons[-1]

    # Hour h of the backward walk sits h hours before the origin month's stamp;
    # interpolate linearly between the two monthly snapshots bracketing it, and
    # clamp at the ends rather than extrapolating (as the published run did).
    origin_time = wind.times[t_idx]
    cache: Dict[int, tuple] = {}

    def interp_at(hours_before: float, pts: np.ndarray):
        target = origin_time - pd.Timedelta(hours=hours_before)
        after = int(np.searchsorted(wind.times.values, target.to_datetime64(), "right"))
        if after <= 0:
            idx, wts = (0,), (1.0,)
        elif after >= len(wind.times):
            idx, wts = (len(wind.times) - 1,), (1.0,)
        else:
            t0, t1 = wind.times[after - 1], wind.times[after]
            span = (t1 - t0).total_seconds()
            w1 = ((target - t0).total_seconds() / span) if span > 0 else 0.5
            idx, wts = (after - 1, after), (1.0 - w1, w1)
        acc = np.zeros((len(pts), 3))
        for i, wt in zip(idx, wts):
            if wt == 0.0:
                continue
            if i not in cache:
                cache[i] = wind.interpolators(i)
            fu, fv, fw = cache[i]
            acc[:, 0] += wt * fu(pts)
            acc[:, 1] += wt * fv(pts)
            acc[:, 2] += wt * fw(pts)
        return acc

    out_lat: List[np.ndarray] = []
    out_lon: List[np.ndarray] = []
    out_idx: List[np.ndarray] = []

    for h in range(hours_back):
        if not alive.any():
            break
        in_domain = ((lat >= lat_min) & (lat <= lat_max)
                     & (lon >= lon_min) & (lon <= lon_max))
        in_band = (pres >= P_MIN_HPA) & (pres <= P_MAX_HPA)
        alive &= in_domain & in_band
        if not alive.any():
            break

        sel = np.flatnonzero(alive)
        out_lat.append(lat[sel].copy())
        out_lon.append(lon[sel].copy())
        out_idx.append(origin_idx[sel].copy())

        wnd = interp_at(h, np.column_stack([lat[sel], lon[sel]]))
        bad = ~np.isfinite(wnd).all(axis=1)
        if bad.any():
            alive[sel[bad]] = False
            sel = sel[~bad]
            wnd = wnd[~bad]
        if len(sel) == 0:
            break

        cos_lat = np.maximum(np.cos(lat[sel] * DEG), np.cos(89.0 * DEG))
        lat[sel] -= wnd[:, 1] * SECONDS_PER_HOUR / EARTH_RADIUS_M / DEG
        lon[sel] -= wnd[:, 0] * SECONDS_PER_HOUR / (EARTH_RADIUS_M * cos_lat) / DEG
        pres[sel] -= wnd[:, 2] * SECONDS_PER_HOUR / 100.0
        lon[sel] = ((lon[sel] + 180.0) % 360.0) - 180.0

    if not out_idx:
        empty = np.zeros(0)
        return empty, empty, np.zeros(0, np.int64)
    return (np.concatenate(out_lat), np.concatenate(out_lon),
            np.concatenate(out_idx))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — circulation matrices
# ─────────────────────────────────────────────────────────────────────────────

def enforce_unidirectionality(C: sp.csr_matrix) -> Tuple[sp.csr_matrix, float]:
    """Drop the lower-triangular direction of any bidirectional pair.

    The paper attributes its ~0.003% of bidirectional links to wind measurement
    error; removing one deterministic direction resolves the conflict without
    touching the single-direction signal.  Returns (cleaned, fraction_removed).
    """
    C = _binarize(C)
    both = C.multiply(C.T).tocsr()
    both.setdiag(0)
    both.eliminate_zeros()
    n_off = int(C.nnz - np.count_nonzero(C.diagonal()))
    frac = (both.nnz / n_off) if n_off else 0.0
    drop = sp.tril(both, k=-1)
    if drop.nnz:
        C = _binarize(_binarize(C) - _binarize(drop))
        C.eliminate_zeros()
    return C.tocsr(), frac


def circulation_matrix(traj_lat, traj_lon, origin_row, coords, resolution: float
                       ) -> sp.csr_matrix:
    """C_t by nearest-cell assignment inside the Voronoi radius; no buffer."""
    n = len(coords)
    if len(traj_lat) == 0:
        return sp.csr_matrix((n, n), dtype=np.int8)
    tree = cKDTree(coords)
    dist, j = tree.query(np.column_stack([traj_lat, traj_lon]), k=1)
    keep = dist <= VORONOI_FACTOR * resolution
    if not keep.any():
        return sp.csr_matrix((n, n), dtype=np.int8)
    C = sp.coo_matrix((np.ones(int(keep.sum()), np.int8),
                       (origin_row[keep], j[keep])), shape=(n, n)).tocsr()
    C.setdiag(0)
    return _binarize(C)


# ─────────────────────────────────────────────────────────────────────────────
# The build
# ─────────────────────────────────────────────────────────────────────────────

def build(lai_nc: Path, wind_nc: Path, out: Path, *, grid: pd.DataFrame,
          resolution: float = DEFAULT_RESOLUTION,
          hours_back: int = DEFAULT_HOURS_BACK,
          wind_field: str = "era5-native",
          interpolate_lai: bool = True,
          months: Optional[int] = None) -> Path:
    """Run all five stages and write ``(C_t_dict, G, panel_df)`` to ``out``."""
    log.info("=" * 78)
    log.info("BUILDING FULL-MODE INPUT")
    log.info("  LAI          %s", lai_nc)
    log.info("  wind         %s  (--wind-field %s)", wind_nc, wind_field)
    log.info("  domain       %d pixels at %.3f deg", len(grid), resolution)
    log.info("  trajectories %d hourly backward steps from %.0f hPa",
             hours_back, P_INIT_HPA)
    log.info("=" * 78)

    panel = build_panel(grid, lai_nc, wind_nc, interpolate_lai=interpolate_lai)
    times = pd.DatetimeIndex(panel.index.get_level_values("timestamp").unique())
    if months:
        times = times[:months]
        log.warning("--months %d: building a TRUNCATED input, for testing only",
                    months)
        panel = panel[panel.index.get_level_values("timestamp").isin(times)]

    wind = load_wind_field(wind_nc, times, wind_field, lai_nc=lai_nc)

    order = np.sort(grid["pixel_id"].to_numpy(np.int64))
    coords = (grid.set_index("pixel_id").loc[order, ["latitude", "longitude"]]
              .to_numpy(float))
    G = queen_adjacency(grid, resolution)
    log.info("G: %d neighbour links, mean degree %.2f", G.nnz, G.nnz / len(order))

    C_t: Dict[pd.Timestamp, sp.csr_matrix] = {}
    fracs: List[float] = []
    for t_idx, ts in enumerate(times):
        tlat, tlon, oidx = trajectories_for_month(
            wind, t_idx, coords[:, 0], coords[:, 1], hours_back)
        C = circulation_matrix(tlat, tlon, oidx, coords, resolution)
        C, frac = enforce_unidirectionality(C)
        C_t[pd.Timestamp(ts)] = C
        fracs.append(frac)
        if (t_idx + 1) % 24 == 0 or t_idx == len(times) - 1:
            log.info("  %3d/%d months | %s | nnz=%d | mean row sum %.3f",
                     t_idx + 1, len(times), pd.Timestamp(ts).date(), C.nnz,
                     C.nnz / len(order))

    if fracs:
        worst = max(fracs)
        log.info("unidirectionality: mean %.5f%%, worst %.5f%% of links were "
                 "bidirectional before cleaning (the paper quotes ~0.003%%)",
                 100 * float(np.mean(fracs)), 100 * worst)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump((C_t, G, panel), fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("wrote %s (%.2f GB)", out, out.stat().st_size / 1e9)
    log.info("run it:  python run_replication.py --full --checkpoint %s", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test — the whole chain on synthetic fields, no credentials, no network
# ─────────────────────────────────────────────────────────────────────────────

def smoke(tmpdir: Optional[Path] = None) -> None:
    """Build a tiny input end to end and assert every stage contract.

    This is what makes the builder testable without the two downloads: the
    fields are synthetic but the code path is the production one, so a contract
    that breaks here would break on the real inputs too.
    """
    import tempfile

    import xarray as xr

    tmp = Path(tmpdir or tempfile.mkdtemp(prefix="build_inputs_smoke_"))
    res = 0.25
    lats = np.arange(-6.0, -2.0 + res / 2, res)
    lons = np.arange(-62.0, -58.0 + res / 2, res)
    times = pd.date_range("1985-01-01", periods=6, freq="MS")

    # A steady easterly with a seasonal wobble: parcels travel west, so upwind
    # neighbours are east of each origin and the shells are populated.
    rng = np.random.default_rng(20260824)
    T, ny, nx = len(times), len(lats), len(lons)
    season = np.cos(2 * np.pi * np.arange(T) / 12)[:, None, None]
    u = np.full((T, ny, nx), -6.0) + 0.5 * season + 0.1 * rng.standard_normal((T, ny, nx))
    v = 0.8 * season + 0.1 * rng.standard_normal((T, ny, nx))
    w = 0.01 * rng.standard_normal((T, ny, nx))
    lai = (4.0 + 0.5 * season
           + 0.3 * rng.standard_normal((T, ny, nx))
           + np.linspace(0, 1, nx)[None, None, :])
    lai[1, 0, 0] = np.nan                     # a cloud gap for the gap-fill

    wind_nc = tmp / "wind.nc"
    lai_nc = tmp / "lai.nc"
    xr.Dataset(
        {"u": (("time", "latitude", "longitude"), u),
         "v": (("time", "latitude", "longitude"), v),
         "w": (("time", "latitude", "longitude"), w)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    ).to_netcdf(wind_nc)
    xr.Dataset(
        {"lai": (("time", "latitude", "longitude"), lai)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    ).to_netcdf(lai_nc)

    lon2, lat2 = np.meshgrid(lons, lats)
    grid = pd.DataFrame({"pixel_id": np.arange(lat2.size, dtype=np.int64),
                         "latitude": lat2.ravel(), "longitude": lon2.ravel()})

    out = build(lai_nc, wind_nc, tmp / "steps_4_6.pkl", grid=grid,
                resolution=res, hours_back=48, wind_field="era5-native")

    with open(out, "rb") as fh:
        C_t, G, panel = pickle.load(fh)

    n = len(grid)
    assert set(panel.columns) == {"lai", "u", "v", "omega"}, panel.columns
    assert panel.index.names == ["pixel_id", "timestamp"], panel.index.names
    assert len(panel) == n * len(times), (len(panel), n * len(times))
    assert not panel["lai"].isna().any(), "gap-fill left a NaN behind"
    assert len(C_t) == len(times), (len(C_t), len(times))
    assert G.shape == (n, n) and G.diagonal().sum() == 0
    for ts, C in C_t.items():
        assert C.shape == (n, n), C.shape
        assert set(np.unique(C.data)) <= {1}, "C_t is not binary"
        assert C.diagonal().sum() == 0, "C_t has a self-loop"
        both = C.multiply(C.T).tocsr()
        both.setdiag(0)
        both.eliminate_zeros()   # setdiag(0) INSERTS explicit zeros; count after
        assert both.nnz == 0, f"{ts}: bidirectional link survived"
    nnz = np.mean([C.nnz for C in C_t.values()])
    assert nnz > 0, "no circulation links at all — the integrator did nothing"

    # The estimation must accept the artefact: shells x C_t must reach k >= 1.
    from .matrices import geodesic_shells, upwind_matrices
    shells = geodesic_shells(G, 5)
    W = upwind_matrices(next(iter(C_t.values())), shells, 5)
    assert W[1].nnz > 0, "W^[1] is empty: shells and C_t do not intersect"

    print(f"\n  SMOKE OK  N={n} T={len(times)} mean C_t nnz={nnz:.1f} "
          f"W1 nnz={W[1].nnz}\n  artefacts under {tmp}")


def verify(rebuilt: Path, committed: Path) -> int:
    """Diff two ``coefficients.csv`` files, coefficient by coefficient.

    Deviations are reported, not asserted against a tolerance: the legitimate
    rebuild-to-rebuild spread is 1e-4 on Sum beta_k (see data/README.md), and
    inventing a threshold would be worse than printing the number.  Produce the
    two files with::

        python run_replication.py --build-inputs
        python run_replication.py --full --checkpoint data/steps_4_6.pkl \\
            --outputs-dir outputs_rebuild
        python run_replication.py --build-inputs --verify
    """
    for p in (rebuilt, committed):
        if not Path(p).exists():
            print(f"  missing {p} — see verify.__doc__ for the two commands")
            return 2
    a = pd.read_csv(rebuilt)
    b = pd.read_csv(committed)
    keys = [c for c in ("spec", "instrument", "k", "lag") if c in a.columns]
    num = [c for c in a.columns if c not in keys
           and pd.api.types.is_numeric_dtype(a[c])]
    if keys:
        a, b = a.set_index(keys).sort_index(), b.set_index(keys).sort_index()
        if not a.index.equals(b.index):
            print("  the two files do not describe the same rows; compare by hand")
            return 2
    print(f"  {rebuilt}  vs  {committed}")
    print(f"  {'column':<16s} {'max |dev|':>12s} {'where':>28s}")
    worst = 0.0
    for col in num:
        d = (a[col] - b[col]).abs()
        i = int(d.to_numpy().argmax())
        worst = max(worst, float(d.iloc[i]))
        print(f"  {col:<16s} {d.iloc[i]:12.3e} {str(d.index[i]):>28s}")
    print(f"\n  worst deviation over all coefficients: {worst:.3e}")
    print("  the authors-346 / binary column is the one to read first: it must "
          "give beta_1 = 0.0049, alpha = 0.2201, Omega^2 mean 1.050.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild the full-mode input from monthly LAI and wind NetCDFs.")
    ap.add_argument("--lai", type=Path, default=Path("data/lai_monthly_0p25_avhrr.nc"))
    ap.add_argument("--wind", type=Path, default=Path("data/era5_monthly_0p25.nc"))
    ap.add_argument("--out", type=Path, default=Path("data/steps_4_6.pkl"))
    ap.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    ap.add_argument("--hours-back", type=int, default=DEFAULT_HOURS_BACK)
    ap.add_argument("--wind-field", choices=("era5-native", "panel-fill"),
                    default="era5-native",
                    help="what a parcel sees over the ocean; panel-fill "
                         "reproduces the published run's land-masked panel")
    ap.add_argument("--domain-from-polygon", action="store_true",
                    help="re-derive the domain from data/amazon_domain.geojson "
                         "instead of the shipped grid CSV, and report the diff")
    ap.add_argument("--no-interpolate-lai", action="store_true",
                    help="leave cloud gaps as NaN (they then enter the spatial "
                         "sums as bare ground — see the module docstring)")
    ap.add_argument("--months", type=int, default=None,
                    help="build only the first N months (testing)")
    ap.add_argument("--smoke", action="store_true",
                    help="run the whole chain on synthetic fields and assert "
                         "every contract; needs no data")
    ap.add_argument("--verify", action="store_true",
                    help="diff outputs_rebuild/coefficients.csv against the "
                         "committed outputs/coefficients.csv")
    ap.add_argument("--rebuilt-coefs", type=Path,
                    default=Path("outputs_rebuild/coefficients.csv"))
    ap.add_argument("--committed-coefs", type=Path,
                    default=Path("outputs/coefficients.csv"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    if args.smoke:
        smoke()
        return 0
    if args.verify:
        return verify(args.rebuilt_coefs, args.committed_coefs)

    grid = (derive_domain_grid(resolution=args.resolution)
            if args.domain_from_polygon else load_domain_grid())
    build(args.lai, args.wind, args.out, grid=grid, resolution=args.resolution,
          hours_back=args.hours_back, wind_field=args.wind_field,
          interpolate_lai=not args.no_interpolate_lai, months=args.months)
    return 0


if __name__ == "__main__":
    sys.exit(main())
