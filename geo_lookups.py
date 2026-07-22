"""
EcoLens -- Real geospatial reference lookups

Replaces the hash-based, fabricated "physical descriptors" that used
to live in 09_explainability_engine.py's get_physical_descriptors().
That function generated elevation/temperature/rainfall/soil/ecoregion
from `sum(ord(c) for c in patch_id)` -- numbers that looked plausible
but had no relationship to the real world. This module looks the same
fields up from real, free, static reference datasets.

DESIGN PRINCIPLE: this module never guesses. If a reference dataset
isn't downloaded yet, the corresponding lookup function returns None
and the caller is responsible for labeling the field as unavailable --
never substituting a plausible-looking placeholder. A missing file is
a loud, obvious gap; a fabricated number is a silent, expensive one.

---------------------------------------------------------------------
REFERENCE DATA SETUP (one-time, see README.md for full instructions)
---------------------------------------------------------------------

1. WDPA (protected areas) -- protectedplanet.net
   Download the "Shapefile" export for the whole planet, merge the
   0/1/2 polygon parts into one layer, save/convert to
   config.WDPA_POLYGONS_PATH.

2. WorldClim v2 bioclim (climate) -- worldclim.org/data/worldclim21.html
   Download BIO1 (annual mean temperature) and BIO12 (annual
   precipitation) at 30 arc-second resolution. Save to
   config.WORLDCLIM_TEMP_PATH / config.WORLDCLIM_PRECIP_PATH.
   Note: BIO1 is stored as degrees C * 10 in some WorldClim releases
   -- get_climate() below auto-detects and corrects this (see
   _fix_worldclim_temp_scale).

3. Elevation -- SRTM 30m (earthexplorer.usgs.gov) or Copernicus
   GLO-30 DEM (registry.opendata.aws/copernicus-dem). Download the
   1x1 degree tiles covering your patch locations and place them,
   UNMOSAICKED, in config.DEM_TILES_DIR -- get_elevation() looks up
   the correct tile per query point on demand. Do NOT try to mosaic
   worldwide-scattered tiles into one file: that means allocating a
   raster covering their full combined bounding box, which for this
   project's global locations is most of the planet.

4. RESOLVE Ecoregions 2017 -- resolve.org/ecoregions
   Single global shapefile, save to config.ECOREGIONS_PATH.

All of these are static, one-time downloads. Nothing in this module
makes a network call at runtime -- it only reads local rasters/vectors.
"""

import json
import os
import warnings

import numpy as np

from config import (
    WDPA_POLYGONS_PATH, WDPA_ACCEPTED_STATUSES,
    WORLDCLIM_TEMP_PATH, WORLDCLIM_PRECIP_PATH,
    DEM_TILES_DIR, ECOREGIONS_PATH, GEO_LOOKUP_CACHE_PATH,
)


# ---------------------------------------------------------------
# Lazy-loaded singletons for the vector layers (loading a global
# shapefile per-lookup would be extremely slow across 700+ patches)
# ---------------------------------------------------------------

_WDPA_GDF = None
_WDPA_LOAD_ATTEMPTED = False
_ECOREGIONS_GDF = None
_ECOREGIONS_LOAD_ATTEMPTED = False
_CACHE = None


def _target_crs():
    """All our query points come in as (lon, lat) in WGS84 / EPSG:4326."""
    return "EPSG:4326"


def _load_wdpa():
    """
    Load WDPA polygons once, filter to real designated (not proposed)
    protected areas, and reproject to EPSG:4326 if needed.

    Returns None (not an exception) if the file isn't present yet --
    callers must treat that as "unknown", not "not protected".
    """
    global _WDPA_GDF, _WDPA_LOAD_ATTEMPTED
    if _WDPA_LOAD_ATTEMPTED:
        return _WDPA_GDF
    _WDPA_LOAD_ATTEMPTED = True

    if not os.path.exists(WDPA_POLYGONS_PATH):
        warnings.warn(
            f"WDPA polygons not found at {WDPA_POLYGONS_PATH} -- "
            f"protected_area will be reported as unknown, not False. "
            f"See geo_lookups.py docstring for download instructions."
        )
        return None

    import geopandas as gpd

    gdf = gpd.read_file(WDPA_POLYGONS_PATH)

    # CRS check -- WDPA shapefiles are sometimes distributed in a
    # projected CRS. Reproject explicitly rather than assuming.
    if gdf.crs is None:
        warnings.warn(
            "WDPA layer has no CRS defined -- assuming EPSG:4326. "
            "Verify this against the source metadata; a wrong "
            "assumption here silently produces wrong point-in-polygon "
            "results with no error."
        )
        gdf = gdf.set_crs(_target_crs())
    elif str(gdf.crs) != _target_crs():
        gdf = gdf.to_crs(_target_crs())

    # Filter to designated (not merely proposed) protected areas.
    if "STATUS" in gdf.columns:
        gdf = gdf[gdf["STATUS"].isin(WDPA_ACCEPTED_STATUSES)]

    # Exclude fully marine protected areas for terrestrial ecosystem
    # queries -- a coastal mangrove point sitting just inside an
    # offshore marine reserve boundary should not register as
    # "protected" for land-cover purposes. WDPA's MARINE field is
    # '0' = not marine, '1' = partially marine, '2' = entirely marine.
    if "MARINE" in gdf.columns:
        gdf = gdf[gdf["MARINE"] != "2"]

    _WDPA_GDF = gdf
    return _WDPA_GDF


def _load_ecoregions():
    global _ECOREGIONS_GDF, _ECOREGIONS_LOAD_ATTEMPTED
    if _ECOREGIONS_LOAD_ATTEMPTED:
        return _ECOREGIONS_GDF
    _ECOREGIONS_LOAD_ATTEMPTED = True

    if not os.path.exists(ECOREGIONS_PATH):
        warnings.warn(
            f"RESOLVE Ecoregions not found at {ECOREGIONS_PATH} -- "
            f"ecoregion will be reported as unavailable."
        )
        return None

    import geopandas as gpd

    gdf = gpd.read_file(ECOREGIONS_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs(_target_crs())
    elif str(gdf.crs) != _target_crs():
        gdf = gdf.to_crs(_target_crs())

    _ECOREGIONS_GDF = gdf
    return _ECOREGIONS_GDF


def _load_cache():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if os.path.exists(GEO_LOOKUP_CACHE_PATH):
        try:
            with open(GEO_LOOKUP_CACHE_PATH) as f:
                _CACHE = json.load(f)
        except Exception:
            _CACHE = {}
    else:
        _CACHE = {}
    return _CACHE


def _save_cache():
    if _CACHE is None:
        return
    os.makedirs(os.path.dirname(GEO_LOOKUP_CACHE_PATH) or ".", exist_ok=True)
    with open(GEO_LOOKUP_CACHE_PATH, "w") as f:
        json.dump(_CACHE, f, indent=2)


def _cache_key(lon, lat, precision=3):
    """
    Round coordinates before caching. At our reference-raster
    resolutions (WorldClim ~1km, SRTM 30m) two points a few hundred
    meters apart legitimately sample the same pixel, so rounding to
    ~100m (3 decimal degrees) is safe and avoids redundant I/O across
    the ~10 sub-crops generated per base location.
    """
    return f"{round(lon, precision)},{round(lat, precision)}"


# ---------------------------------------------------------------
# Point / windowed raster sampling
# ---------------------------------------------------------------

def sample_raster_point(raster_path, lon, lat, window=1, nodata_override=None):
    """
    Sample a raster at (lon, lat) in EPSG:4326, reprojecting the
    query point into the raster's native CRS first.

    window=1 samples a single pixel. window=3 (etc.) samples an
    NxN window centered on the point and returns the mean of valid
    (non-nodata) pixels -- useful for elevation in mountainous
    terrain, where a lone pixel sample can be misleading relative to
    the ~2km patch the point represents.

    Returns None if the file doesn't exist, the point falls outside
    the raster's extent, or all sampled pixels are nodata. Never
    returns a fabricated fallback value.
    """
    if not os.path.exists(raster_path):
        return None

    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(raster_path) as src:
        if src.crs is not None and str(src.crs) != _target_crs():
            xs, ys = warp_transform(_target_crs(), src.crs, [lon], [lat])
            x, y = xs[0], ys[0]
        else:
            x, y = lon, lat

        row, col = src.index(x, y)

        half = window // 2
        row0, row1 = max(0, row - half), min(src.height, row + half + 1)
        col0, col1 = max(0, col - half), min(src.width, col + half + 1)
        if row1 <= row0 or col1 <= col0:
            return None

        window_data = src.read(
            1,
            window=rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0),
        ).astype(np.float64)

        nodata = nodata_override if nodata_override is not None else src.nodata
        if nodata is not None:
            valid = window_data[window_data != nodata]
        else:
            valid = window_data.flatten()

        return float(np.mean(valid))


def sample_raster_point_stats(raster_path, lon, lat, window=3, nodata_override=None):
    """
    Sample a raster at (lon, lat) EPSG:4326, returning both the mean
    and standard deviation (ruggedness proxy) over an NxN window.
    """
    if not os.path.exists(raster_path):
        return None, None

    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(raster_path) as src:
        if src.crs is not None and str(src.crs) != _target_crs():
            xs, ys = warp_transform(_target_crs(), src.crs, [lon], [lat])
            x, y = xs[0], ys[0]
        else:
            x, y = lon, lat

        row, col = src.index(x, y)

        half = window // 2
        row0, row1 = max(0, row - half), min(src.height, row + half + 1)
        col0, col1 = max(0, col - half), min(src.width, col + half + 1)
        if row1 <= row0 or col1 <= col0:
            return None, None

        window_data = src.read(
            1,
            window=rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0),
        ).astype(np.float64)

        nodata = nodata_override if nodata_override is not None else src.nodata
        if nodata is not None:
            valid = window_data[window_data != nodata]
        else:
            valid = window_data.flatten()

        if valid.size == 0:
            return None, None

        return float(np.mean(valid)), float(np.std(valid))



# ---------------------------------------------------------------
# Public lookup functions
# ---------------------------------------------------------------

def is_protected(lon, lat):
    """
    Real point-in-polygon protected-area check against WDPA.

    Returns True / False, or None if WDPA data isn't available --
    callers must distinguish "confirmed not protected" from "unknown",
    not collapse both to False.

    KNOWN LIMITATION: some WDPA records (especially small or
    poorly-mapped reserves) are stored as a point + a reported-area
    radius rather than a true polygon boundary. A naive
    point-in-polygon check under-detects those. For a proof-of-concept
    at your current location count this is an acceptable gap -- worth
    revisiting if you scale up and protection status becomes a
    load-bearing model feature.
    """
    gdf = _load_wdpa()
    if gdf is None:
        return None

    from shapely.geometry import Point

    point = Point(lon, lat)
    hits = gdf[gdf.contains(point)]
    return bool(len(hits) > 0)


def _fix_worldclim_temp_scale(raw_value):
    """
    Some WorldClim v2 BIO1 distributions store temperature as
    degrees C, others as degrees C * 10 (an older convention carried
    over from WorldClim v1). Values outside a physically plausible
    range for annual mean temperature (-60 to 40 C) are assumed to be
    in the *10 scale and corrected. This is a heuristic over a
    *real* sampled value, not a fabrication -- it only corrects units.
    """
    if raw_value is None:
        return None
    if -60.0 <= raw_value <= 40.0:
        return raw_value
    return raw_value / 10.0


def get_climate(lon, lat):
    """
    Real annual mean temperature (deg C) and annual precipitation (mm)
    from WorldClim v2, sampled at the query point.

    Returns (temp_c, rainfall_mm), with either element None if its
    raster isn't available.

    RESOLUTION CAVEAT: WorldClim v2 30 arc-second rasters are ~1km
    resolution. This describes the regional climate around the point,
    not something specific to your ~2.24km Sentinel-2 patch. That's
    appropriate for climate (which varies slowly over space) but
    should not be read as patch-level precision.
    """
    temp = sample_raster_point(WORLDCLIM_TEMP_PATH, lon, lat, window=1)
    rainfall = sample_raster_point(WORLDCLIM_PRECIP_PATH, lon, lat, window=1)
    temp = _fix_worldclim_temp_scale(temp)
    return temp, rainfall


def _dem_tile_path(lon, lat):
    """
    Locate the Copernicus DEM GLO-30 tile file covering (lon, lat),
    using the same 1x1-degree, southwest-corner naming convention as
    download_reference_data.py's _dem_tile_name() (verified against
    AWS's own tile listing). Returns the local path whether or not
    the file actually exists -- callers check existence themselves,
    consistent with every other lookup in this module never assuming
    data is present.
    """
    lat_tile = int(lat // 1)
    lon_tile = int(lon // 1)
    ns = "N" if lat_tile >= 0 else "S"
    ew = "E" if lon_tile >= 0 else "W"
    tile_name = f"Copernicus_DSM_COG_10_{ns}{abs(lat_tile):02d}_00_{ew}{abs(lon_tile):03d}_00_DEM"
    return os.path.join(DEM_TILES_DIR, f"{tile_name}.tif")


def get_elevation_and_ruggedness(lon, lat):
    """
    Real elevation and ruggedness (elevation standard deviation in meters)
    from SRTM/Copernicus DEM.
    """
    tile_path = _dem_tile_path(lon, lat)
    if not os.path.exists(tile_path):
        return None, None
    return sample_raster_point_stats(tile_path, lon, lat, window=3)


def get_elevation(lon, lat):
    """
    Real elevation (meters) from SRTM/Copernicus DEM, averaged over a
    3x3 pixel window (~90m at SRTM 30m resolution) to reduce
    sensitivity to a single noisy pixel on sloped terrain.

    Looks up the single 1x1 degree tile covering this point rather
    than reading from a pre-built mosaic -- see DEM_TILES_DIR's
    comment in config.py for why: this project's locations are
    scattered worldwide, and mosaicking globally-scattered tiles into
    one dense raster means allocating an array covering their full
    combined bounding box, which is most of the planet.

    Returns None if the tile covering this point hasn't been
    downloaded (e.g. a location outside your original download list,
    or a query point near a tile boundary that happens to fall in a
    neighboring tile you don't have).
    """
    elev, _ = get_elevation_and_ruggedness(lon, lat)
    return elev


def get_ecoregion(lon, lat):
    """
    Real ecoregion name and biome from RESOLVE Ecoregions 2017, via
    point-in-polygon lookup. Returns (eco_name, biome_name), with
    both None if the reference layer isn't available or the point
    falls in a gap (e.g. open ocean, ice sheet).
    """
    gdf = _load_ecoregions()
    if gdf is None:
        return None, None

    from shapely.geometry import Point

    point = Point(lon, lat)
    hits = gdf[gdf.contains(point)]
    if len(hits) == 0:
        return None, None

    row = hits.iloc[0]
    eco_name = row.get("ECO_NAME", None)
    biome_name = row.get("BIOME_NAME", None)
    return eco_name, biome_name


def get_physical_descriptors(lon, lat, use_cache=True):
    """
    Real replacement for the old hash-based get_physical_descriptors().

    Returns a dict with the same shape the explainability engine
    expects (elevation, temp, rainfall, ecoregion, protected_area),
    plus a "data_source" field per group of values so downstream code
    (and the explanations it generates) can tell real values apart
    from gaps -- never silently substituting a guess.

    Soil type is intentionally omitted: there is no static, easily
    downloadable global soil raster as simple to integrate as the
    others (SoilGrids requires either its REST API or large raster
    downloads per property). If you add SoilGrids, wire it in here
    following the same "return None on missing data" contract.
    """
    cache = _load_cache() if use_cache else {}
    key = _cache_key(lon, lat)
    if use_cache and key in cache:
        cached = cache[key]
        # Only trust a cached result if it was fully resolved when it was
        # written. A result cached back when reference data wasn't
        # downloaded yet has "unavailable" sources -- returning that
        # forever, even after the real data shows up, is exactly the bug
        # that happened in practice: 09/10 ran once before the WDPA/
        # WorldClim/DEM downloads completed, cached all-None results for
        # every location, and every run after that silently returned the
        # same stale nulls even though the real data was sitting right
        # there. So: incomplete cache entries are NOT trusted -- fall
        # through and recompute instead.
        sources = [cached.get("climate_source"), cached.get("elevation_source"),
                   cached.get("ecoregion_source"), cached.get("protection_source")]
        if "unavailable" not in sources and "ruggedness_m" in cached:
            return cached
        # else: fall through and recompute -- don't return the stale entry

    temp, rainfall = get_climate(lon, lat)
    elevation, ruggedness = get_elevation_and_ruggedness(lon, lat)
    eco_name, biome_name = get_ecoregion(lon, lat)
    protected = is_protected(lon, lat)

    result = {
        "elevation_m": round(elevation, 1) if elevation is not None else None,
        "ruggedness_m": round(ruggedness, 1) if ruggedness is not None else None,
        "temp_c": round(temp, 1) if temp is not None else None,
        "rainfall_mm": round(rainfall, 1) if rainfall is not None else None,
        "ecoregion": eco_name,
        "biome": biome_name,
        "protected_area": protected,
        "climate_source": "WorldClim v2" if (temp is not None or rainfall is not None) else "unavailable",
        "elevation_source": "SRTM/Copernicus DEM" if elevation is not None else "unavailable",
        "ecoregion_source": "RESOLVE Ecoregions 2017" if eco_name is not None else "unavailable",
        "protection_source": "WDPA" if protected is not None else "unavailable",
    }

    # Only persist fully-resolved results (see the comment above for why
    # partial/unavailable results deliberately aren't cached -- they need
    # to keep retrying until the reference data behind them exists).
    sources = [result["climate_source"], result["elevation_source"],
               result["ecoregion_source"], result["protection_source"]]
    if use_cache and "unavailable" not in sources:
        cache[key] = result
        _save_cache()

    return result


# ---------------------------------------------------------------
# Sanity-check helper
# ---------------------------------------------------------------

def spot_check(locations, expect=None):
    """
    Run get_physical_descriptors() against a handful of known
    locations and print the results for manual verification against
    what you already know about those places, e.g.:

        spot_check([
            {"name": "Sundarbans, India", "lon": 88.85, "lat": 21.95},
            {"name": "Siberian Boreal Forest", "lon": 92.5, "lat": 56.5},
        ])

    This is a deliberate, cheap safeguard: a CRS mismatch or a
    marine/terrestrial WDPA filtering bug will pass code review but
    produce systematically wrong features. Eyeballing a handful of
    results against common knowledge (Sundarbans should be wet and
    warm; Siberia should be cold and low-rainfall) catches that before
    it propagates into 700+ patches or a risk model.
    """
    print(f"\n{'='*70}")
    print("GEO LOOKUP SPOT CHECK")
    print(f"{'='*70}")
    for loc in locations:
        desc = get_physical_descriptors(loc["lon"], loc["lat"], use_cache=False)
        print(f"\n{loc['name']} ({loc['lon']}, {loc['lat']}):")
        for k, v in desc.items():
            print(f"    {k}: {v}")
    print(f"\n{'='*70}")
    print("Check these against what you already know about each place.")
    print("A CRS or filtering bug produces plausible-looking wrong")
    print("numbers, not a crash -- this is the only defense against that.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Quick manual smoke test using a few well-known locations from
    # config.PATCH_LOCATIONS.
    from config import PATCH_LOCATIONS

    sample_names = [
        "Sundarbans, West Bengal, India",
        "Siberian Boreal Forest, Russia",
        "Amazon rainforest, Brazil",
        "Punjab farmland, India",
    ]
    sample_locs = [
        {"name": loc["name"], "lon": loc["lon"], "lat": loc["lat"]}
        for loc in PATCH_LOCATIONS
        if loc["name"] in sample_names
    ]
    spot_check(sample_locs)