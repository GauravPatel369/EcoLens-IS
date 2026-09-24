"""
EcoLens Step 26 -- BUILD THE WDPA PROTECTED-AREA LAYER  (C11, C18)

WHY
---
`protected_area` has been the one feature in this project with no real data behind it.
`geo_lookups.py` deliberately returns None rather than False when WDPA is missing ("a
missing file is a loud, obvious gap; a fabricated number is a silent, expensive one"), so
07b's protection column falls back to a hand-curated per-location flag and the risk model's
`protected_area` column is 100 % empty -- which is why C18's Anthropogenic feature group
rests on `distance_to_prior_loss_m` alone.

The full WDPA release is public: a 4.17 GB archive containing three nested zips whose
polygon layers total ~4.4 GB of shapefile. Naively unzipping it needs ~12 GB, against ~11 GB
free on this machine, so this script never holds more than one part at a time:

    for each part:  extract nested zip -> extract polygon files only -> filter AT READ TIME
                    -> append to a GeoPackage -> delete everything extracted

Filtering happens in the OGR `where` clause, not in pandas, so the unwanted polygons are
never materialised in memory. Two filters, both matching what `geo_lookups.py` already
applies downstream:

  * STATUS in (Designated, Inscribed, Established) -- excludes merely *proposed* areas.
  * REALM <> 'Marine'  -- excludes wholly marine reserves (keeping Terrestrial and
    Coastal), so a coastal mangrove point just inside an offshore boundary is not counted
    as protected land. The older WDPA field for this was MARINE; see KEEP_COLS.

Output is a GeoPackage rather than a shapefile: shapefiles cap at 2 GB per file and store
attributes in dBASE, and the merged layer would sit near that limit. `geopandas.read_file`
reads either transparently.

Writes geo_data/WDPA_terrestrial_designated.gpkg

Run:
    python 26_build_wdpa_layer.py
    python 26_build_wdpa_layer.py --keep-archive     # don't delete the 4.17 GB source zip
"""

import argparse
import os
import shutil
import zipfile

# Anchored via config so this works from any working directory and follows the
# data/reference layout rather than re-creating the old flat geo_data/ folder.
from config import GEO_DATA_DIR, WDPA_POLYGONS_PATH
SRC_ZIP = f"{GEO_DATA_DIR}/WDPA_Sep2026_Public_shp.zip"
TMP_DIR = f"{GEO_DATA_DIR}/_wdpa_tmp"
OUT_GPKG = WDPA_POLYGONS_PATH
PARTS = [0, 1, 2]

ACCEPTED_STATUSES = ("Designated", "Inscribed", "Established")
# NOTE (6 Sep): the current WDPA schema has NO `WDPAID` and NO `MARINE` field. It uses
# SITE_ID / SITE_PID and a three-valued REALM (Terrestrial 96,105 / Coastal 4,590 /
# Marine 1,645 in part 0). geo_lookups.py still filters on `MARINE`, guarded by
# `if "MARINE" in gdf.columns` -- so on a modern download that filter silently does
# NOTHING and wholly-marine reserves are admitted as protected land. Fixed there too.
KEEP_COLS = ["SITE_ID", "SITE_PID", "NAME", "DESIG_ENG", "IUCN_CAT", "STATUS", "REALM"]

MIN_FREE_GB = 4.0        # refuse to start a part if we would risk filling the disk


def free_gb(path="."):
    total, used, free = shutil.disk_usage(path)
    return free / 2**30


def extract_part(part):
    """Pull one nested zip out of the archive, then its polygon files only."""
    nested_name = f"WDPA_Sep2026_Public_shp_{part}.zip"
    nested_path = os.path.join(TMP_DIR, nested_name)
    os.makedirs(TMP_DIR, exist_ok=True)
    if not os.path.exists(nested_path):
        with zipfile.ZipFile(SRC_ZIP) as z:
            z.extract(nested_name, TMP_DIR)

    shp = None
    with zipfile.ZipFile(nested_path) as zz:
        for name in zz.namelist():
            # polygons only -- the points layer is a different geometry type and is not
            # usable for point-in-polygon containment.
            if "-polygons." in name:
                zz.extract(name, TMP_DIR)
                if name.endswith(".shp"):
                    shp = os.path.join(TMP_DIR, name)
    os.remove(nested_path)          # the nested zip is dead weight once unpacked
    return shp


def main():
    ap = argparse.ArgumentParser(description="Build the WDPA terrestrial/designated layer")
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep the 4.17 GB source zip after building the layer")
    args = ap.parse_args()

    import geopandas as gpd
    import pyogrio

    if not os.path.exists(SRC_ZIP):
        raise SystemExit(f"{SRC_ZIP} not found -- download it first.")

    # REALM <> 'Marine' keeps Terrestrial AND Coastal, which is the intent the old
    # MARINE <> '2' encoded: a partly-marine coastal reserve still protects land.
    where = ("STATUS IN ('" + "','".join(ACCEPTED_STATUSES) + "') "
             "AND REALM <> 'Marine'")
    print(f"\n{'='*78}")
    print("BUILDING WDPA TERRESTRIAL / DESIGNATED LAYER (C11)")
    print(f"{'='*78}")
    print(f"  filter (applied at read time, not in pandas):\n    {where}\n")

    if os.path.exists(OUT_GPKG):
        os.remove(OUT_GPKG)

    total = 0
    for part in PARTS:
        if free_gb() < MIN_FREE_GB:
            raise SystemExit(f"ABORT: only {free_gb():.1f} GB free, need >{MIN_FREE_GB} GB "
                             f"headroom to unpack part {part} safely.")
        print(f"[part {part}] extracting ... ({free_gb():.1f} GB free)")
        shp = extract_part(part)
        if shp is None:
            print(f"[part {part}] no polygon layer found -- skipped")
            continue

        cols = set(pyogrio.read_info(shp)["fields"])
        keep = [c for c in KEEP_COLS if c in cols]
        print(f"[part {part}] reading with filter ...")
        gdf = pyogrio.read_dataframe(shp, columns=keep, where=where)
        print(f"[part {part}] {len(gdf):,} polygons kept")

        if gdf.crs is None:
            # Stated rather than assumed: WDPA ships EPSG:4326, but a silent wrong
            # assumption here produces wrong containment with no error.
            print(f"[part {part}] WARNING: no CRS on source; assuming EPSG:4326")
            gdf = gdf.set_crs("EPSG:4326")
        elif str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        gdf.to_file(OUT_GPKG, layer="wdpa", driver="GPKG",
                    mode="w" if total == 0 else "a")
        total += len(gdf)
        del gdf

        for f in os.listdir(TMP_DIR):
            os.remove(os.path.join(TMP_DIR, f))
        print(f"[part {part}] done, cleaned up ({free_gb():.1f} GB free)\n")

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    size_gb = os.path.getsize(OUT_GPKG) / 2**30
    print(f"  TOTAL: {total:,} terrestrial designated protected areas")
    print(f"  wrote {OUT_GPKG}  ({size_gb:.2f} GB)")

    if not args.keep_archive:
        os.remove(SRC_ZIP)
        print(f"  removed {SRC_ZIP} (regenerable: it is a public static download)")
    print(f"  {free_gb():.1f} GB free")


if __name__ == "__main__":
    main()
