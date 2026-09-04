"""
Download the two raw inputs from their public sources.
======================================================

``build_inputs.py`` needs two NetCDF files and nothing else:

    data/lai_monthly_0p25_avhrr.nc    monthly LAI,   0.25 deg, 1985-2013
    data/era5_monthly_0p25.nc         monthly u/v/w, 0.25 deg, 800 hPa

This module produces both.  Together with ``build_inputs`` it means every
artefact this package consumes can be rebuilt from public data, so nothing here
rests on a file that has to be requested by email.

    python run_replication.py --fetch-raw --what lai      # Earth Engine
    python run_replication.py --fetch-raw --what wind     # Copernicus CDS
    python run_replication.py --fetch-raw --what both

Credentials, and what each one costs
------------------------------------
**LAI — NOAA CDR AVHRR LAI/FAPAR V5, via Google Earth Engine.**  Needs
``pip install earthengine-api``, one ``earthengine authenticate``, and a Google
Cloud project with the Earth Engine API enabled (``--project``).  All reduction
is server-side: the daily images of each month are averaged, scaled by 0.001 to
physical LAI, bin-averaged from the native 0.05 deg to 0.25 deg and pinned to
the target grid, so what comes over the wire is one small array per month.
348 months, roughly 1-2 hours wall-clock, ~30 MB on disk.  Months are cached
individually, so an interrupted run resumes.

**Wind — ERA5 at 800 hPa, via the Copernicus Climate Data Store.**  Needs
``pip install cdsapi`` and a ``~/.cdsapirc``.  Two routes, and the choice
matters enough to be explicit rather than defaulted:

``--wind-source hourly`` downloads hourly analyses and averages them per
calendar month, which is what the published run did.  It is faithful and it is
expensive: ~30 GB of transfer and, with CDS queueing, a day or more.

``--wind-source monthly-means`` takes ECMWF's own monthly-means product in a
handful of requests, ~200 MB, typically under an hour.  It is *not* identical:
against the published run's hourly-averaged field the worst deviation measured
at 800 hPa was 0.63% of the field's own standard deviation.  That is small, and
it is not zero, and no one has yet measured what it does to beta_k.  If you use
this route, say so when you report numbers.

Why the LAI record stops at 2013
--------------------------------
The AVHRR CDR ends in 2013 and the VIIRS CDR takes over in 2014.  They share a
band name, a grid and a scale factor, which makes the splice look free — but at
the handover the level jumps by 26%, a multiplicative break that no linear
control absorbs.  1985-2013 is also exactly the paper's period, so the record is
truncated at 2013 rather than corrected.  A residual ~5% NOAA-14 -> NOAA-16 break
at 2000-2001 is accepted and documented.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("replication.fetch_raw")

# On networks with a TLS-inspecting proxy the CA lives in the OS trust store,
# not in certifi, and both downloads fail with CERTIFICATE_VERIFY_FAILED.
# truststore fixes that if it is installed; if it is not, nothing changes.
try:  # pragma: no cover - environment-dependent
    import truststore

    truststore.inject_into_ssl()
    log.debug("truststore injected: using the OS certificate store")
except Exception:  # noqa: BLE001 — never let TLS hardening break the CLI
    pass

BBOX = (-56.0, 13.0, -82.0, -34.0)      # lat_min, lat_max, lon_min, lon_max
RESOLUTION = 0.25
PRESSURE_HPA = 800

AVHRR_COLLECTION = "NOAA/CDR/AVHRR/LAI_FAPAR/V5"
AVHRR_LAST_YEAR = 2013
LAI_BAND = "LAI"
LAI_SCALE = 0.001
_FILL = -9999.0


def month_starts(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """Month-start stamps, the axis both files are labelled on.

    Both inputs must carry the *same* labels or the panel's time-axis
    intersection silently comes back short, so this is the one definition.
    """
    return pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")


def wind_target_grid() -> Tuple[np.ndarray, np.ndarray]:
    """The lattice ERA5 is delivered on, and the one pixel_id is defined against:
    cell centres *on* the bbox bounds (-56.000, -82.000), 277 x 193."""
    lat_min, lat_max, lon_min, lon_max = BBOX
    half = RESOLUTION / 2.0
    return (np.arange(lat_min, lat_max + half, RESOLUTION),
            np.arange(lon_min, lon_max + half, RESOLUTION))


def lai_target_grid() -> Tuple[np.ndarray, np.ndarray]:
    """The LAI grid, which is offset from the wind lattice by HALF A CELL.

    Aggregating 0.05 deg LAI to 0.25 deg puts cell *edges* on the bbox bounds and
    therefore centres at -55.875, -81.875 — 276 x 192, not 277 x 193.  This is
    not a detail to tidy up: because the pixel centres sit on the wind lattice,
    every pixel ends up exactly equidistant from four LAI cells, and which one
    the panel assigns is decided by a tie-break in
    ``build_inputs._nearest_cell``.  Writing LAI on the wind lattice instead
    would remove the tie and silently change the panel — and hence beta_1 by
    about 6% and alpha by 1.3%.  See data/README.md.
    """
    lat_min, lat_max, lon_min, lon_max = BBOX
    half = RESOLUTION / 2.0
    return (np.arange(lat_min + half, lat_max, RESOLUTION),
            np.arange(lon_min + half, lon_max, RESOLUTION))


def target_grid() -> Tuple[np.ndarray, np.ndarray]:
    """Deprecated alias for :func:`wind_target_grid`; the two grids differ."""
    return wind_target_grid()


# ─────────────────────────────────────────────────────────────────────────────
# LAI — Google Earth Engine
# ─────────────────────────────────────────────────────────────────────────────

def fetch_lai(out: Path, project: Optional[str], start_year: int = 1985,
              end_year: int = AVHRR_LAST_YEAR,
              cache_dir: Optional[Path] = None) -> Path:
    """Monthly 0.25 deg AVHRR LAI over the domain, assembled into one NetCDF."""
    try:
        import ee
    except ImportError:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "earthengine-api is not installed.\n"
            "  pip install earthengine-api\n"
            "  earthengine authenticate\n"
            "and pass --project <your-google-cloud-project>.")
    import xarray as xr

    if end_year > AVHRR_LAST_YEAR:
        raise SystemExit(
            f"--end-year {end_year} crosses the 2014 VIIRS handover, which is a "
            f"26% multiplicative break. Stop at {AVHRR_LAST_YEAR} (also the "
            f"paper's last year) or splice-correct first — see the module "
            f"docstring.")

    try:
        ee.Initialize(project=project)
    except Exception as exc:                     # pragma: no cover
        raise SystemExit(f"ee.Initialize failed: {exc}\nRun `earthengine "
                         f"authenticate` and pass --project.")

    lats, lons = lai_target_grid()      # half-cell offset — see that docstring
    if (len(lats), len(lons)) != (276, 192):
        log.warning("LAI grid is %d x %d; the published file is 276 x 192. A "
                    "different bbox changes which cells exist, not only how "
                    "many, so the panel will not match.", len(lats), len(lons))
    cache = Path(cache_dir or out.parent / "gee_lai_cache")
    cache.mkdir(parents=True, exist_ok=True)

    # reduceResolution needs a fixed projection, and a mean() composite has
    # none, so the native 0.05 deg projection is re-attached before aggregating
    # and the result is pinned to the exact target grid.  The translate() is to
    # the top-left cell EDGE, which is what puts the centres at the half-cell
    # offset the published LAI file carries.
    proj = ee.Projection("EPSG:4326").translate(
        lons[0] - RESOLUTION / 2, lats[-1] + RESOLUTION / 2).scale(
        RESOLUTION, -RESOLUTION)

    stamps = month_starts(start_year, end_year)
    frames: List[np.ndarray] = []
    for ts in stamps:
        npy = cache / f"lai_{ts.year}_{ts.month:02d}.npy"
        if npy.exists():
            frames.append(np.load(npy))
            continue
        start = ee.Date.fromYMD(ts.year, ts.month, 1)
        end = start.advance(1, "month")
        coll = (ee.ImageCollection(AVHRR_COLLECTION).select(LAI_BAND)
                .filterDate(start, end))
        native = coll.first().projection()
        img = (coll.mean().multiply(LAI_SCALE).setDefaultProjection(native)
               .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
               .reproject(proj).unmask(_FILL))
        url = img.getDownloadURL({
            "bands": [LAI_BAND], "crs": "EPSG:4326",
            "dimensions": [len(lons), len(lats)],
            "region": ee.Geometry.Rectangle(
                [lons[0] - RESOLUTION / 2, lats[0] - RESOLUTION / 2,
                 lons[-1] + RESOLUTION / 2, lats[-1] + RESOLUTION / 2],
                proj="EPSG:4326", geodesic=False),
            "format": "NPY"})
        import io
        import urllib.request
        with urllib.request.urlopen(url) as fh:
            raw = np.load(io.BytesIO(fh.read()))
        arr = np.asarray(raw[LAI_BAND] if raw.dtype.names else raw, np.float32)
        arr[arr == _FILL] = np.nan
        arr = arr[::-1, :]                       # delivered north-to-south
        np.save(npy, arr)
        frames.append(arr)
        log.info("  %s  finite %.1f%%", ts.date(),
                 100 * float(np.isfinite(arr).mean()))

    data = np.stack(frames, axis=0)
    ds = xr.Dataset({"lai": (("time", "latitude", "longitude"), data)},
                    coords={"time": stamps, "latitude": lats, "longitude": lons},
                    attrs={"source": AVHRR_COLLECTION,
                           "note": "monthly mean of daily retrievals, scaled to "
                                   "physical LAI, bin-averaged 0.05 -> 0.25 deg"})
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    log.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Wind — Copernicus CDS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_wind(out: Path, source: str, start_year: int = 1985,
               end_year: int = 2013, cache_dir: Optional[Path] = None) -> Path:
    """Monthly 800 hPa u/v/omega over the domain, assembled into one NetCDF."""
    try:
        import cdsapi
    except ImportError:  # pragma: no cover
        raise SystemExit("cdsapi is not installed.\n  pip install cdsapi\n"
                         "and put your key in ~/.cdsapirc "
                         "(https://cds.climate.copernicus.eu/how-to-api)")
    import xarray as xr

    if source not in ("hourly", "monthly-means"):
        raise SystemExit("--wind-source must be hourly or monthly-means")
    if source == "monthly-means":
        log.warning(
            "--wind-source monthly-means: ECMWF's own monthly product deviates "
            "from the published run's hourly-averaged field by up to 0.63%% of "
            "the field sd at 800 hPa. Cheap and not identical — report which "
            "route you used.")

    lat_min, lat_max, lon_min, lon_max = BBOX
    area = [lat_max, lon_min, lat_min, lon_max]      # N, W, S, E
    client = cdsapi.Client()
    cache = Path(cache_dir or out.parent / "era5_cache")
    cache.mkdir(parents=True, exist_ok=True)

    years = [str(y) for y in range(start_year, end_year + 1)]
    variables = ["u_component_of_wind", "v_component_of_wind", "vertical_velocity"]
    paths: List[Path] = []

    if source == "monthly-means":
        for year in years:
            target = cache / f"era5_monthly_{year}.nc"
            if not target.exists():
                log.info("  requesting %s …", target.name)
                client.retrieve(
                    "reanalysis-era5-pressure-levels-monthly-means",
                    {"product_type": "monthly_averaged_reanalysis",
                     "variable": variables,
                     "pressure_level": [str(PRESSURE_HPA)],
                     "year": [year],
                     "month": [f"{m:02d}" for m in range(1, 13)],
                     "time": "00:00", "area": area, "grid": [RESOLUTION, RESOLUTION],
                     "data_format": "netcdf"},
                    str(target))
            paths.append(target)
        ds = xr.open_mfdataset([str(p) for p in paths], combine="by_coords")
    else:
        for year in years:
            for month in range(1, 13):
                target = cache / f"era5_hourly_{year}_{month:02d}.nc"
                if not target.exists():
                    ndays = calendar.monthrange(int(year), month)[1]
                    log.info("  requesting %s …", target.name)
                    client.retrieve(
                        "reanalysis-era5-pressure-levels",
                        {"product_type": "reanalysis", "variable": variables,
                         "pressure_level": [str(PRESSURE_HPA)],
                         "year": [year], "month": [f"{month:02d}"],
                         "day": [f"{d:02d}" for d in range(1, ndays + 1)],
                         "time": [f"{h:02d}:00" for h in range(24)],
                         "area": area, "grid": [RESOLUTION, RESOLUTION],
                         "data_format": "netcdf"},
                        str(target))
                paths.append(target)
        log.info("averaging %d hourly files to calendar months …", len(paths))
        ds = (xr.open_mfdataset([str(p) for p in paths], combine="by_coords")
              .resample(time="MS").mean())

    # Both routes must land on month-start labels, or the panel's time-axis
    # intersection with LAI silently comes back short.
    tname = "time" if "time" in ds.coords else "valid_time"
    ds = ds.rename({tname: "time"}) if tname != "time" else ds
    ds = ds.assign_coords(time=pd.DatetimeIndex(ds["time"].values).to_period("M")
                          .to_timestamp())
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.attrs["aggregation"] = f"800 hPa, {source}"
    ds.to_netcdf(out)
    ds.close()
    log.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Download the monthly LAI and wind inputs from public sources.")
    ap.add_argument("--what", choices=("lai", "wind", "both"), required=True)
    ap.add_argument("--lai-out", type=Path,
                    default=Path("data/lai_monthly_0p25_avhrr.nc"))
    ap.add_argument("--wind-out", type=Path,
                    default=Path("data/era5_monthly_0p25.nc"))
    ap.add_argument("--project", default=None,
                    help="Google Cloud project with the Earth Engine API enabled")
    ap.add_argument("--wind-source", choices=("hourly", "monthly-means"),
                    default="hourly",
                    help="hourly reproduces the published run; monthly-means is "
                         "far cheaper and deviates by up to 0.63%% of field sd")
    ap.add_argument("--start-year", type=int, default=1985)
    ap.add_argument("--end-year", type=int, default=2013)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    if args.what in ("lai", "both"):
        fetch_lai(args.lai_out, args.project, args.start_year, args.end_year)
    if args.what in ("wind", "both"):
        fetch_wind(args.wind_out, args.wind_source, args.start_year, args.end_year)
    print("\nNext:  python run_replication.py --build-inputs "
          f"--lai {args.lai_out} --wind {args.wind_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
