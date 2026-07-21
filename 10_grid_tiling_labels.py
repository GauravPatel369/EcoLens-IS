"""
EcoLens Objective 4 - Step 10: Grid Tiling + Forest-Loss Labels

Turns the project's ~30-45 forest/mangrove BASE LOCATIONS into
hundreds-to-thousands of independent CELL-YEAR samples suitable for
training a forest-loss risk model.

---------------------------------------------------------------------
WHY THIS SCRIPT EXISTS (see README.md -> "Sample size" for the full
discussion)
---------------------------------------------------------------------
A supervised loss-risk classifier needs many independent examples,
and specifically many POSITIVE (loss) examples -- with ~30-45 named
forest/mangrove locations and maybe 20% showing loss in any given
year, that's single digits of positive examples, nowhere near enough
to fit a stable decision boundary. Standard practice in the
deforestation-risk literature (Hansen-label-based systems like
GLAD/DETER) works with thousands of grid cells, not named places.

This script:
  1. Takes each forest/mangrove base location from config.PATCH_LOCATIONS
  2. Tiles a GRID_REGION_BUFFER_KM-radius region around it into
     GRID_CELL_SIZE_M x GRID_CELL_SIZE_M cells
  3. For each cell and each observation year, pulls REAL labels from
     Hansen Global Forest Change (treecover2000 + lossyear only --
     Hansen stopped publishing a separate "loss" binary layer as of
     GFC v1.4, since it's redundant with lossyear > 0; requesting it
     404s)
  4. Computes driver features per cell: baseline tree cover, distance
     to the nearest already-lost pixel (deforestation spreads from
     existing edges -- this is one of the strongest known predictors),
     and real climate/elevation/protection status via geo_lookups.py
  5. Writes a cell-year table to config.RISK_FEATURES_PATH for
     11_forest_risk_forecast.py to train on.

---------------------------------------------------------------------
NETWORK REQUIREMENT
---------------------------------------------------------------------
Hansen GFC tiles are large (~1 degree tile can be tens of MB) public
GeoTIFFs on Google Cloud Storage, no authentication required:
    https://storage.googleapis.com/earthenginepartners-hansen/<version>/...
This script downloads the tiles it needs on first use and caches them
under config.HANSEN_DATA_DIR. It needs outbound network access to
storage.googleapis.com -- if you're running this inside a sandboxed
environment without that access, download the relevant tiles manually
first (see hansen_tile_url() below for the exact URL pattern) and
place them in HANSEN_DATA_DIR before running.

Run:
    python 10_grid_tiling_labels.py
"""

import csv
import math
import os
import urllib.request

import numpy as np

from config import (
    PATCH_LOCATIONS, RISK_FOREST_ECOSYSTEMS, GRID_CELL_SIZE_M,
    GRID_REGION_BUFFER_KM, HANSEN_DATA_DIR, HANSEN_BASE_URL,
    HANSEN_TREECOVER_THRESHOLD, RISK_HORIZON_YEARS, RISK_FEATURES_PATH,
    RISK_MODEL_DIR,
)
import geo_lookups

HANSEN_LAYERS = ["treecover2000", "lossyear"]

# Hansen GFC provides annual loss through this dataset year (update if
# you download a newer HANSEN_VERSION in config.py). Observation years
# must leave room for a full RISK_HORIZON_YEARS lookahead within the
# data, so the latest usable observation year is capped below that.
HANSEN_DATA_THROUGH_YEAR = 2023
OBS_YEAR_START = 2005
OBS_YEAR_END = HANSEN_DATA_THROUGH_YEAR - RISK_HORIZON_YEARS


# ---------------------------------------------------------------
# Hansen tile naming / download
# ---------------------------------------------------------------

def hansen_tile_name(lat, lon):
    """
    Hansen GFC tiles are 10x10 degree GeoTIFFs named by their
    NORTHWEST corner, e.g. a point at (21.95 N, 88.85 E) -- the
    Sundarbans -- falls in tile "30N_080E" (the tile spanning
    20N-30N, 80E-90E).
    """
    tile_lat = math.ceil(lat / 10.0) * 10
    tile_lon = math.floor(lon / 10.0) * 10
    lat_str = f"{abs(tile_lat):02d}{'N' if tile_lat >= 0 else 'S'}"
    lon_str = f"{abs(tile_lon):03d}{'E' if tile_lon >= 0 else 'W'}"
    return f"{lat_str}_{lon_str}"


def hansen_tile_url(layer, lat, lon):
    from config import HANSEN_VERSION
    tile = hansen_tile_name(lat, lon)
    return f"{HANSEN_BASE_URL}/Hansen_{HANSEN_VERSION}_{layer}_{tile}.tif"


def ensure_hansen_tile(layer, lat, lon):
    """
    Download (once, cached) the Hansen tile covering (lat, lon) for
    the given layer. Returns the local path, or None if the download
    failed (e.g. no network access) -- callers must handle None
    rather than crash, so a missing tile degrades gracefully to
    "no label available for this cell" instead of killing the run.
    """
    tile = hansen_tile_name(lat, lon)
    os.makedirs(HANSEN_DATA_DIR, exist_ok=True)
    local_path = os.path.join(HANSEN_DATA_DIR, f"Hansen_{layer}_{tile}.tif")
    if os.path.exists(local_path):
        return local_path

    url = hansen_tile_url(layer, lat, lon)
    print(f"  Downloading Hansen {layer} tile {tile} ...")
    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception as e:
        print(f"  WARNING: could not download {url} ({e}). "
              f"Cells in tile {tile} will have no Hansen label this run. "
              f"See this script's module docstring for manual download instructions.")
        if os.path.exists(local_path):
            os.remove(local_path)  # remove any partial download
        return None


# ---------------------------------------------------------------
# Geometry: km <-> degrees, grid generation
# ---------------------------------------------------------------

def km_to_deg(km, lat):
    """
    Convert a distance in km to (delta_lat_deg, delta_lon_deg) at the
    given latitude. Same approach as 02_preprocess_patches.py's
    calculate_offset_coords, generalized to arbitrary km.
    """
    d_lat = km / 110.54
    d_lon = km / (111.32 * math.cos(math.radians(lat)) + 1e-8)
    return d_lat, d_lon


def generate_grid_cells(center_lon, center_lat, buffer_km, cell_size_m):
    """
    Generate a square grid of cell centers covering a
    buffer_km-radius region around (center_lon, center_lat).

    Returns a list of (cell_lon, cell_lat) tuples.
    """
    cell_km = cell_size_m / 1000.0
    d_lat_buffer, d_lon_buffer = km_to_deg(buffer_km, center_lat)
    d_lat_cell, d_lon_cell = km_to_deg(cell_km, center_lat)

    n_steps = max(1, int(buffer_km / cell_km))

    cells = []
    for i in range(-n_steps, n_steps + 1):
        for j in range(-n_steps, n_steps + 1):
            cell_lat = center_lat + i * d_lat_cell
            cell_lon = center_lon + j * d_lon_cell
            # Keep only cells within the circular buffer, not the full square
            dist_km = math.hypot(i * cell_km, j * cell_km)
            if dist_km <= buffer_km:
                cells.append((cell_lon, cell_lat))
    return cells


# ---------------------------------------------------------------
# Hansen raster feature extraction
# ---------------------------------------------------------------

def read_window(raster_path_or_dataset, min_lon, min_lat, max_lon, max_lat):
    """
    Read a lon/lat bounding box window from a raster (assumed already
    in EPSG:4326, which Hansen GFC tiles are) and return (array,
    pixel_size_x_deg, pixel_size_y_deg). Returns None if the file
    doesn't exist or the window is empty.

    Accepts EITHER a path (opens and closes the file for this one
    read -- convenient for one-off callers like predict_risk()) OR an
    already-open rasterio dataset object (no open/close overhead --
    used by the main tiling loop, which calls this hundreds of
    thousands of times against the same handful of Hansen tiles;
    reopening the file on every call there was the dominant cost of
    the whole script, dwarfing the actual array math).
    """
    import rasterio
    from rasterio.windows import from_bounds

    if raster_path_or_dataset is None:
        return None, None, None

    if isinstance(raster_path_or_dataset, str):
        if not os.path.exists(raster_path_or_dataset):
            return None, None, None
        with rasterio.open(raster_path_or_dataset) as src:
            return _read_window_from_open_dataset(src, min_lon, min_lat, max_lon, max_lat)
    else:
        # Already an open rasterio dataset -- read directly, no open/close.
        return _read_window_from_open_dataset(raster_path_or_dataset, min_lon, min_lat, max_lon, max_lat)


def _read_window_from_open_dataset(src, min_lon, min_lat, max_lon, max_lat):
    from rasterio.windows import from_bounds

    window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
    window = window.round_offsets().round_lengths()
    if window.width <= 0 or window.height <= 0:
        return None, None, None
    arr = src.read(1, window=window)
    px_x = src.transform.a
    px_y = -src.transform.e
    return arr, px_x, px_y


def distance_to_prior_loss_m(lossyear_arr, px_x_deg, px_y_deg, lat, obs_year_offset):
    """
    Distance (meters) from the center of lossyear_arr to the nearest
    pixel that was ALREADY lost as of obs_year_offset (i.e.
    1 <= lossyear <= obs_year_offset, where lossyear is years since
    2000). Deforestation spreading from an existing cleared edge is
    one of the most predictive real signals in this literature --
    this is a real geometric computation, not a placeholder.

    Returns None if there's no prior loss anywhere in the window
    (distance is effectively "far", but we don't invent a number --
    callers should treat None as "no nearby loss detected within the
    sampled window", which is itself informative).
    """
    if lossyear_arr is None or lossyear_arr.size == 0:
        return None

    from scipy.ndimage import distance_transform_edt

    prior_loss_mask = (lossyear_arr >= 1) & (lossyear_arr <= obs_year_offset)
    if not prior_loss_mask.any():
        return None

    # distance_transform_edt gives distance (in pixels) from every
    # "background" (False) pixel to the nearest "foreground" (True)
    # pixel when we invert the mask -- so invert prior_loss_mask.
    px_size_m_x = px_x_deg * 111320.0 * math.cos(math.radians(lat))
    px_size_m_y = px_y_deg * 110540.0
    dist_px = distance_transform_edt(~prior_loss_mask, sampling=(px_size_m_y, px_size_m_x))

    center_row, center_col = dist_px.shape[0] // 2, dist_px.shape[1] // 2
    return float(dist_px[center_row, center_col])


def compute_cell_label_and_features(cell_lon, cell_lat, cell_half_km, tiles, obs_year):
    """
    For one cell and one observation year, compute:
      - baseline treecover2000 % (must exceed HANSEN_TREECOVER_THRESHOLD
        AND not already have been lost before obs_year, to count as
        "forested at the observation point" -- otherwise there's
        nothing left to lose and the sample shouldn't be included)
      - label: was there loss in (obs_year, obs_year + RISK_HORIZON_YEARS]
      - distance_to_prior_loss_m: driver feature

    Returns a dict, or None if this cell/year isn't a valid sample
    (not forested at baseline, or Hansen data unavailable for this tile).

    `tiles` is (treecover_source, lossyear_source), where each source
    is either a file path (opened and closed for this one read -- used
    by predict_risk()'s single-point queries) or an already-open
    rasterio dataset (reused across many calls -- used by main()'s
    tiling loop, see read_window()'s docstring for why this matters).
    """
    treecover_src, lossyear_src = tiles
    d_lat, d_lon = km_to_deg(cell_half_km, cell_lat)
    min_lon, max_lon = cell_lon - d_lon, cell_lon + d_lon
    min_lat, max_lat = cell_lat - d_lat, cell_lat + d_lat

    treecover_arr, _, _ = read_window(treecover_src, min_lon, min_lat, max_lon, max_lat)
    lossyear_arr, px_x, px_y = read_window(lossyear_src, min_lon, min_lat, max_lon, max_lat)

    if treecover_arr is None or lossyear_arr is None:
        return None

    baseline_treecover_pct = float(np.mean(treecover_arr))
    if baseline_treecover_pct < HANSEN_TREECOVER_THRESHOLD:
        return None

    obs_offset = obs_year - 2000  # Hansen lossyear encodes years since 2000

    # Already lost before the observation year? Then it's not "at risk"
    # forest at this observation point -- skip (avoid double-counting
    # already-cleared land as a fresh at-risk sample).
    already_lost = (lossyear_arr >= 1) & (lossyear_arr <= obs_offset)
    if already_lost.mean() > 0.5:
        return None

    # Label: any loss in the window within the horizon after obs_year
    horizon_mask = (lossyear_arr > obs_offset) & (lossyear_arr <= obs_offset + RISK_HORIZON_YEARS)
    label = int(horizon_mask.any())

    dist_prior_loss = distance_to_prior_loss_m(lossyear_arr, px_x, px_y, cell_lat, obs_offset)

    return {
        "baseline_treecover_pct": round(baseline_treecover_pct, 2),
        "distance_to_prior_loss_m": round(dist_prior_loss, 1) if dist_prior_loss is not None else None,
        "label_loss_within_horizon": label,
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    import rasterio
    from contextlib import ExitStack
    from tqdm import tqdm

    forest_locations = [loc for loc in PATCH_LOCATIONS if loc["ecosystem"] in RISK_FOREST_ECOSYSTEMS]
    print(f"\n{'='*70}")
    print(f"EcoLens Step 10: Grid Tiling + Hansen Forest-Loss Labels")
    print(f"{'='*70}")
    print(f"Regions to tile: {len(forest_locations)} ({RISK_FOREST_ECOSYSTEMS})")
    print(f"Cell size: {GRID_CELL_SIZE_M}m, buffer radius: {GRID_REGION_BUFFER_KM}km")
    print(f"Observation years: {OBS_YEAR_START}-{OBS_YEAR_END}, horizon: {RISK_HORIZON_YEARS} years\n")

    cell_half_km = (GRID_CELL_SIZE_M / 1000.0) / 2.0
    rows = []
    path_cache = {}      # tile_name -> (treecover_path, lossyear_path)
    dataset_cache = {}   # tile_name -> (treecover_dataset, lossyear_dataset), OPENED ONCE
    n_missing_tiles = 0

    # ExitStack guarantees every opened rasterio dataset gets closed when
    # main() exits, even if a location errors out partway through --
    # avoids leaking file handles across a run touching dozens of tiles.
    with ExitStack() as stack:
        for loc in forest_locations:
            print(f"\n[{loc['id']}] Tiling region around {loc['name']} ({loc['lon']}, {loc['lat']})...")
            cells = generate_grid_cells(loc["lon"], loc["lat"], GRID_REGION_BUFFER_KM, GRID_CELL_SIZE_M)
            print(f"  {len(cells)} cells generated")

            for cell_lon, cell_lat in tqdm(cells, desc=f"  {loc['id']} cells"):
                tile_name = hansen_tile_name(cell_lat, cell_lon)

                if tile_name not in path_cache:
                    tc_path = ensure_hansen_tile("treecover2000", cell_lat, cell_lon)
                    lossyear_path = ensure_hansen_tile("lossyear", cell_lat, cell_lon)
                    path_cache[tile_name] = (tc_path, lossyear_path)
                    if tc_path is None or lossyear_path is None:
                        n_missing_tiles += 1

                    # Open each raster ONCE here and keep the handle alive
                    # for the rest of the run -- this is the fix. Every
                    # cell/year that falls in this tile reuses the same
                    # open dataset instead of paying rasterio.open()'s
                    # header-parsing cost again.
                    tc_ds = stack.enter_context(rasterio.open(tc_path)) if tc_path else None
                    ly_ds = stack.enter_context(rasterio.open(lossyear_path)) if lossyear_path else None
                    dataset_cache[tile_name] = (tc_ds, ly_ds)

                tiles = dataset_cache[tile_name]

                for obs_year in range(OBS_YEAR_START, OBS_YEAR_END + 1):
                    result = compute_cell_label_and_features(cell_lon, cell_lat, cell_half_km, tiles, obs_year)
                    if result is None:
                        continue

                    geo = geo_lookups.get_physical_descriptors(cell_lon, cell_lat)

                    rows.append({
                        "region_id": loc["id"],
                        "ecosystem": loc["ecosystem"],
                        "cell_lon": round(cell_lon, 5),
                        "cell_lat": round(cell_lat, 5),
                        "obs_year": obs_year,
                        "baseline_treecover_pct": result["baseline_treecover_pct"],
                        "distance_to_prior_loss_m": result["distance_to_prior_loss_m"],
                        "protected_area": geo["protected_area"],
                        "temp_c": geo["temp_c"],
                        "rainfall_mm": geo["rainfall_mm"],
                        "elevation_m": geo["elevation_m"],
                        "label_loss_within_horizon": result["label_loss_within_horizon"],
                    })

    os.makedirs(RISK_MODEL_DIR, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(RISK_FEATURES_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    n_positive = sum(r["label_loss_within_horizon"] for r in rows)
    print(f"\n{'='*70}")
    print(f"Done. {len(rows)} cell-year samples written to {RISK_FEATURES_PATH}")
    print(f"  Positive (loss within horizon): {n_positive} ({100*n_positive/max(len(rows),1):.1f}%)")
    print(f"  Negative (stable):              {len(rows) - n_positive}")
    print(f"  Unique Hansen tiles touched: {len(path_cache)}")
    if n_missing_tiles:
        print(f"\n  WARNING: {n_missing_tiles} Hansen tile(s) could not be downloaded --")
        print(f"  some cells were skipped as a result. See this script's module")
        print(f"  docstring if you're running without network access to")
        print(f"  storage.googleapis.com.")
    if len(rows) < 200:
        print(f"\n  NOTE: {len(rows)} samples is on the low side for training a")
        print(f"  stable classifier (see README.md -> 'Sample size'). Consider")
        print(f"  increasing GRID_REGION_BUFFER_KM, adding more forest/mangrove")
        print(f"  base locations, or widening the observation-year range.")


if __name__ == "__main__":
    main()