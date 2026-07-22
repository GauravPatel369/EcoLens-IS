"""
EcoLens -- download_reference_data.py

Downloads the reference datasets geo_lookups.py needs, into the paths
config.py already expects. Run once, before 09/10/11 for real (not
null) climate/elevation/ecoregion/protection-status fields.

---------------------------------------------------------------------
HONESTY NOTE
---------------------------------------------------------------------
The URLs below were verified by directly fetching each source page at
the time this script was written -- they are real, current, working
links, not guessed. What was NOT done: actually running these
downloads end-to-end (this was written in a sandboxed environment
without network access to these hosts). Treat this as "carefully
verified, not yet executed" -- if a host has changed its layout since,
tell me the error and I'll help fix the URL rather than guess again.

    python download_reference_data.py --all
    python download_reference_data.py --ecoregions --worldclim --dem
    python download_reference_data.py --wdpa --wdpa-token YOUR_TOKEN_HERE

---------------------------------------------------------------------
WDPA IS DIFFERENT FROM THE OTHER THREE
---------------------------------------------------------------------
RESOLVE Ecoregions, WorldClim, and Copernicus DEM all have stable,
static, unauthenticated download URLs -- verified below. WDPA does
NOT: protectedplanet.net's download button generates a session-scoped
link, not a fixed URL, so it can't be hardcoded here. The only
scriptable path is WDPA's own API, which needs a free personal token:

    1. Request one at https://api.protectedplanet.net/request
       (instant, no cost, just an email/name form)
    2. Run: python download_reference_data.py --wdpa --wdpa-token <token>

If you'd rather not get a token, download the shapefile manually from
https://www.protectedplanet.net/en/thematic-areas/wdpa (choose
"Shapefile") and use merge_wdpa_parts() below to combine the 0/1/2
parts it comes in.
"""

import argparse
import os
import zipfile

import requests

from config import (
    GEO_DATA_DIR, WDPA_POLYGONS_PATH, WORLDCLIM_TEMP_PATH, WORLDCLIM_PRECIP_PATH,
    DEM_TILES_DIR, ECOREGIONS_PATH,
)

CHUNK_SIZE = 1024 * 1024  # 1MB


def _download(url, dest_path, label):
    """Stream-download a URL to dest_path with basic progress output."""
    if os.path.exists(dest_path):
        print(f"  {label}: already present at {dest_path}, skipping download.")
        return dest_path

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    print(f"  Downloading {label} from {url} ...")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        print(f"\r    {downloaded/1e6:.0f}MB / {total/1e6:.0f}MB ({pct:.0f}%)", end="")
            print()
        return dest_path
    except requests.RequestException as e:
        print(f"  FAILED to download {label}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return None


def _extract_zip(zip_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    return extract_dir


# ---------------------------------------------------------------
# 1. RESOLVE Ecoregions 2017 -- verified static URL, no auth
# ---------------------------------------------------------------

def download_ecoregions():
    print("\n=== RESOLVE Ecoregions 2017 ===")
    if os.path.exists(ECOREGIONS_PATH):
        print(f"  Already present at {ECOREGIONS_PATH}, skipping.")
        return

    zip_path = os.path.join(GEO_DATA_DIR, "Ecoregions2017.zip")
    url = "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"
    result = _download(url, zip_path, "RESOLVE Ecoregions (~150MB)")
    if result is None:
        return

    extract_dir = os.path.join(GEO_DATA_DIR, "ecoregions_extracted")
    _extract_zip(zip_path, extract_dir)

    # Find the .shp inside and move/point config at it
    for fname in os.listdir(extract_dir):
        if fname.lower().endswith(".shp"):
            src = os.path.join(extract_dir, fname)
            _copy_shapefile_set(src, ECOREGIONS_PATH)
            print(f"  Ready at {ECOREGIONS_PATH}")
            return
    print("  WARNING: no .shp file found inside the downloaded zip -- check its contents manually.")


# ---------------------------------------------------------------
# 2. WorldClim v2 bioclim 30s -- verified static URL, no auth.
#    One zip has all 19 variables; we only keep BIO1 and BIO12.
# ---------------------------------------------------------------

def download_worldclim():
    print("\n=== WorldClim v2 (BIO1 temperature, BIO12 precipitation) ===")
    if os.path.exists(WORLDCLIM_TEMP_PATH) and os.path.exists(WORLDCLIM_PRECIP_PATH):
        print(f"  Already present, skipping.")
        return

    zip_path = os.path.join(GEO_DATA_DIR, "wc2.1_30s_bio.zip")
    url = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_bio.zip"
    result = _download(url, zip_path, "WorldClim bioclim 30s (all 19 variables, ~1GB zip)")
    if result is None:
        return

    extract_dir = os.path.join(GEO_DATA_DIR, "worldclim_extracted")
    _extract_zip(zip_path, extract_dir)

    for wanted, dest in [("wc2.1_30s_bio_1.tif", WORLDCLIM_TEMP_PATH),
                         ("wc2.1_30s_bio_12.tif", WORLDCLIM_PRECIP_PATH)]:
        src = os.path.join(extract_dir, wanted)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            os.replace(src, dest)
            print(f"  Moved {wanted} -> {dest}")
        else:
            print(f"  WARNING: expected {wanted} not found in the extracted zip.")

    print(f"  Note: the other 17 BIO variables are in {extract_dir}/ if you want them; "
          f"this project only uses BIO1 and BIO12.")


# ---------------------------------------------------------------
# 3. Copernicus DEM GLO-30 -- verified S3 bucket, anonymous access
# ---------------------------------------------------------------

def _dem_tile_name(lat_deg, lon_deg):
    """
    Copernicus DEM GLO-30 tiles are 1x1 degree, named by their
    SOUTHWEST corner, e.g. Copernicus_DSM_COG_10_N00_00_E006_00_DEM
    covers 0N-1N, 6E-7E. Verified against AWS's own listing example.
    """
    ns = "N" if lat_deg >= 0 else "S"
    ew = "E" if lon_deg >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_deg):02d}_00_{ew}{abs(lon_deg):03d}_00_DEM"


def download_dem(locations, buffer_deg=0.2):
    """
    Download the Copernicus DEM tiles covering your actual patch
    locations (plus a small buffer), not the whole globe. Tiles are
    kept UNMOSAICKED in config.DEM_TILES_DIR -- geo_lookups.py's
    get_elevation() looks up the right tile per query point on
    demand. Do NOT mosaic these into one file: for worldwide-
    scattered locations, that means allocating a raster covering
    their full combined bounding box, which is most of the planet --
    this is a real failure mode (a 1.28 TiB allocation error), not a
    hypothetical one.

    locations: list of dicts with "lon"/"lat" keys (e.g. config.PATCH_LOCATIONS)
    """
    print("\n=== Copernicus DEM GLO-30 (elevation) ===")

    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
    except ImportError:
        print("  boto3 not installed -- run: pip install boto3")
        return

    s3 = boto3.client("s3", region_name="eu-central-1", config=Config(signature_version=UNSIGNED))
    bucket = "copernicus-dem-30m"

    # Collect the unique 1x1 degree tiles needed to cover every location.
    needed_tiles = set()
    for loc in locations:
        for dlat in (-buffer_deg, 0, buffer_deg):
            for dlon in (-buffer_deg, 0, buffer_deg):
                lat_tile = int((loc["lat"] + dlat) // 1)
                lon_tile = int((loc["lon"] + dlon) // 1)
                needed_tiles.add((lat_tile, lon_tile))

    os.makedirs(DEM_TILES_DIR, exist_ok=True)
    downloaded_paths = []

    for lat_tile, lon_tile in sorted(needed_tiles):
        tile_name = _dem_tile_name(lat_tile, lon_tile)
        key = f"{tile_name}/{tile_name}.tif"
        local_path = os.path.join(DEM_TILES_DIR, f"{tile_name}.tif")

        if os.path.exists(local_path):
            downloaded_paths.append(local_path)
            continue

        print(f"  Fetching DEM tile {tile_name} ...")
        try:
            s3.download_file(bucket, key, local_path)
            downloaded_paths.append(local_path)
        except Exception as e:
            print(f"  WARNING: could not fetch {key} ({e}) -- likely an ocean "
                  f"tile (no land, no DEM data) or a tile not yet public. Skipping.")

    print(f"\n  {len(downloaded_paths)} DEM tile(s) ready in {DEM_TILES_DIR}/ -- "
          f"no mosaic step needed, geo_lookups.py reads them directly.")


# ---------------------------------------------------------------
# 4. WDPA -- no stable static URL; use the real API with a token,
#    or merge a manually-downloaded zip's 0/1/2 shapefile parts.
# ---------------------------------------------------------------

def download_wdpa_via_api(token, country_iso3_list=None):
    """
    Download WDPA data via the official API (requires a free token
    from https://api.protectedplanet.net/request). Without a country
    filter this can be a very large download -- pass
    country_iso3_list to scope it to countries your patch locations
    are actually in, which is enough for this project's purposes.
    """
    print("\n=== WDPA (via API) ===")
    if os.path.exists(WDPA_POLYGONS_PATH):
        print(f"  Already present at {WDPA_POLYGONS_PATH}, skipping.")
        return

    if not country_iso3_list:
        print("  No country list given -- pass --wdpa-countries IND,BRA,COD,... to scope")
        print("  the download to countries your patch locations actually fall in.")
        print("  Downloading the FULL global dataset via the website's Shapefile button")
        print("  instead is likely faster for a global project like this one -- see")
        print("  this script's module docstring for the manual-download + merge path.")
        return

    import zipfile as zf
    parts = []
    for iso3 in country_iso3_list:
        url = f"https://api.protectedplanet.net/v3/countries/{iso3}?with_geometry=true&token={token}"
        # NOTE: the WDPA API's exact download-shapefile endpoint has changed
        # across versions -- if this 404s, check the current endpoint at
        # https://api.protectedplanet.net/documentation and adjust the URL
        # here rather than assume this one is still current.
        dest = os.path.join(GEO_DATA_DIR, f"wdpa_{iso3}.zip")
        result = _download(url, dest, f"WDPA {iso3}")
        if result:
            parts.append(result)

    if parts:
        print(f"  Downloaded {len(parts)} country file(s). Extract and use")
        print(f"  merge_wdpa_parts() below to combine them into {WDPA_POLYGONS_PATH}.")


def merge_wdpa_parts(part_shapefile_paths, output_path=WDPA_POLYGONS_PATH):
    """
    WDPA's standard "Shapefile" download comes as 3 separate shapefiles
    (named _0, _1, _2 -- split because of the dataset's size, not by
    geography). This merges them into the single layer geo_lookups.py
    expects. Use this after a manual download from
    https://www.protectedplanet.net/en/thematic-areas/wdpa, or after
    download_wdpa_via_api() above.
    """
    import geopandas as gpd
    import pandas as pd

    print(f"\nMerging {len(part_shapefile_paths)} WDPA parts -> {output_path}")
    parts = [gpd.read_file(p) for p in part_shapefile_paths]
    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged.to_file(output_path)
    print(f"  Done: {len(merged)} protected area records written to {output_path}")


def _copy_shapefile_set(src_shp, dest_shp):
    """A .shp is always accompanied by .shx/.dbf/.prj/etc siblings --
    copy the whole set, not just the .shp, or geopandas can't read it."""
    import shutil
    src_base = os.path.splitext(src_shp)[0]
    dest_base = os.path.splitext(dest_shp)[0]
    os.makedirs(os.path.dirname(dest_shp) or ".", exist_ok=True)
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        s = src_base + ext
        if os.path.exists(s):
            shutil.copy(s, dest_base + ext)


def main():
    parser = argparse.ArgumentParser(description="Download EcoLens reference datasets.")
    parser.add_argument("--all", action="store_true", help="Download ecoregions + worldclim + DEM (not WDPA -- needs a token).")
    parser.add_argument("--ecoregions", action="store_true")
    parser.add_argument("--worldclim", action="store_true")
    parser.add_argument("--dem", action="store_true")
    parser.add_argument("--wdpa", action="store_true")
    parser.add_argument("--wdpa-token", default=None, help="Free token from https://api.protectedplanet.net/request")
    parser.add_argument("--wdpa-countries", default=None, help="Comma-separated ISO3 codes, e.g. IND,BRA,COD")
    args = parser.parse_args()

    os.makedirs(GEO_DATA_DIR, exist_ok=True)

    if args.all or args.ecoregions:
        download_ecoregions()
    if args.all or args.worldclim:
        download_worldclim()
    if args.all or args.dem:
        from config import PATCH_LOCATIONS, RISK_FOREST_ECOSYSTEMS
        forest_locs = [l for l in PATCH_LOCATIONS if l["ecosystem"] in RISK_FOREST_ECOSYSTEMS]
        download_dem(forest_locs)
    if args.wdpa:
        countries = args.wdpa_countries.split(",") if args.wdpa_countries else None
        if args.wdpa_token:
            download_wdpa_via_api(args.wdpa_token, countries)
        else:
            print("\n=== WDPA ===")
            print("  Pass --wdpa-token (get one free at https://api.protectedplanet.net/request)")
            print("  or download the Shapefile manually from")
            print("  https://www.protectedplanet.net/en/thematic-areas/wdpa and call")
            print("  merge_wdpa_parts() from this script with the 3 extracted .shp paths.")


if __name__ == "__main__":
    main()