"""
EcoLens Step 34 -- GENERATE ADDITIONAL FOREST RISK REGIONS  (review comment C5)

WHY THIS EXISTS
---------------
The risk model is evaluated by leave-one-region-out: hold out one forest region entirely,
train on the rest, score the held-out landscape. Statistical power therefore comes from the
NUMBER OF REGIONS, and with 17 forest locations there are only 16 usable folds. A paired
Wilcoxon over 16 folds has roughly 27% power at the effect size Clay shows, which is why
the Clay replication reads p = 0.2312 while Prithvi reads p = 0.00076. Going to 50 regions
raises that to about 57%, and 150 to about 96%.

Review comment C5 asks for the same thing from the ecological side: "increase the number of
ecosystems, geographic regions and environmental conditions ... representing different
climatic zones, forest types, disturbance regimes and conservation status."

HOW THIS DIFFERS FROM THE EXISTING 17, AND WHY THAT IS DELIBERATE
-----------------------------------------------------------------
The existing 17 are hand-picked famous forests (Amazon, Bialowieza, Redwood, Periyar).
That is a biased sample: large, intact, often protected. This script samples
SYSTEMATICALLY from RESOLVE forest biomes instead, so the expanded set includes ordinary
and fragmented forest too -- a harder and more honest generalisation test. Every generated
row carries selection="auto" so the two-method provenance is recorded in the data rather
than in somebody's memory, and the write-up can state it truthfully.

Nothing about the loss calculation differs. 10_grid_tiling_labels.py takes only lon/lat
from a config entry; tree cover, climate, elevation and protection are all looked up live
from the global layers, so a generated region is processed by exactly the same code path as
a hand-picked one.

RISK-ONLY BY DESIGN
-------------------
These regions are written to config.RISK_EXTRA_REGIONS, NOT to PATCH_LOCATIONS. Adding them
to PATCH_LOCATIONS would make stages 01/02/03 acquire imagery and embed them into the
retrieval catalogue, which would take forest from 13.5% to 31% of the catalogue and change
the meaning of every Pillar A metric -- mAP, chance precision, the spectral-baseline
contrast -- invalidating results that are already written up. Keeping them out leaves
Pillar A untouched. The cost is that they can retrieve the 126 catalogue locations as
analogs but can never BE an analog; the pool stays at 1,260 vectors.

THIS SCRIPT DOES NOT EDIT config.py.
It writes a candidate list for review plus a paste-ready block. Selection is seeded, so
re-running reproduces the same set.

Usage:
    python 34_generate_risk_regions.py                    # 33 new regions, review only
    python 34_generate_risk_regions.py --n 133            # toward 150 regions
    python 34_generate_risk_regions.py --allow-download   # permit new Hansen tiles
    python 34_generate_risk_regions.py --emit-config      # print the config block
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

from config import (PATCH_LOCATIONS, RISK_FOREST_ECOSYSTEMS, GEO_DATA_DIR,
                    HANSEN_DATA_DIR, HANSEN_TREECOVER_THRESHOLD,
                    GRID_REGION_BUFFER_KM, RESULTS_DIR)

ECOREGIONS_PATH = os.path.join(GEO_DATA_DIR, "Ecoregions2017.shp")

# RESOLVE 2017 biome names that count as forest for this purpose. Mangroves are a RESOLVE
# forest-adjacent biome but are a SEPARATE ecosystem category in this project (14 mangrove
# locations already exist in the catalogue), so including them here would mislabel them as
# ecosystem="forest". Excluded deliberately.
FOREST_BIOMES = (
    "Tropical & Subtropical Moist Broadleaf Forests",
    "Tropical & Subtropical Dry Broadleaf Forests",
    "Tropical & Subtropical Coniferous Forests",
    "Temperate Broadleaf & Mixed Forests",
    "Temperate Conifer Forests",
    "Boreal Forests/Taiga",
    "Mediterranean Forests, Woodlands & Scrub",
)

OUT_JSON = os.path.join(RESULTS_DIR, "candidate_risk_regions.json")
OUT_PY = os.path.join(RESULTS_DIR, "candidate_risk_regions.py")


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def hansen_tile_name(lat, lon):
    """Hansen tiles are 10x10 degree, named by NORTHWEST corner.

    Duplicated from 10_grid_tiling_labels rather than imported: importing 10 pulls in torch
    and a STAC client for its embedding-drift path, which this script has no use for.
    Kept byte-identical in behaviour -- if one changes, change both.
    """
    tile_lat = math.ceil(lat / 10.0) * 10
    tile_lon = math.floor(lon / 10.0) * 10
    return (f"{abs(tile_lat):02d}{'N' if tile_lat >= 0 else 'S'}_"
            f"{abs(tile_lon):03d}{'E' if tile_lon >= 0 else 'W'}")


def tiles_on_disk():
    """Tiles for which BOTH Hansen layers are already downloaded.

    Restricting candidates to these makes tree-cover verification a local read and means
    10_tiling needs no new downloads later -- Hansen tiles run 50-400 MB each, so a
    globally-scattered 33 regions could otherwise pull 5-15 GB.
    """
    have = defaultdict(set)
    if not os.path.isdir(HANSEN_DATA_DIR):
        return set()
    for f in os.listdir(HANSEN_DATA_DIR):
        if not f.endswith(".tif"):
            continue
        parts = f[:-4].split("_")
        if len(parts) >= 4 and parts[0] == "Hansen":
            have["_".join(parts[-2:])].add(parts[1])
    return {t for t, layers in have.items()
            if "treecover2000" in layers and "lossyear" in layers}


def mean_treecover(lon, lat, radius_km, tile_cache):
    """Mean treecover2000 over the region box, or None if unreadable.

    Downsampled to at most 256x256: this is a go/no-go screen, not a measurement, and the
    full-resolution box is ~1100x1100 px at 30 m.
    """
    import rasterio
    from rasterio.windows import from_bounds

    tile = hansen_tile_name(lat, lon)
    path = os.path.join(HANSEN_DATA_DIR, f"Hansen_treecover2000_{tile}.tif")
    if not os.path.exists(path):
        return None
    if tile not in tile_cache:
        try:
            tile_cache[tile] = rasterio.open(path)
        except Exception:
            tile_cache[tile] = None
    ds = tile_cache[tile]
    if ds is None:
        return None

    dlat = radius_km / 111.0
    dlon = radius_km / max(1e-6, 111.0 * math.cos(math.radians(lat)))
    try:
        win = from_bounds(lon - dlon, lat - dlat, lon + dlon, lat + dlat,
                         transform=ds.transform)
        # a point near a tile edge yields a window partly outside the raster; rasterio
        # returns only the overlapping part, which is still a fair sample of the region
        arr = ds.read(1, window=win, out_shape=(1, min(256, max(1, int(win.height))),
                                                min(256, max(1, int(win.width)))),
                      boundless=False)
    except Exception:
        return None
    if arr.size == 0:
        return None
    a = arr.astype("float32")
    a = a[a <= 100]            # Hansen treecover2000 is 0-100; anything else is nodata
    return float(a.mean()) if a.size else None


def existing_regions():
    """Forest regions already configured -- the separation constraint applies to these too."""
    out = [dict(loc) for loc in PATCH_LOCATIONS
           if loc.get("ecosystem") in RISK_FOREST_ECOSYSTEMS]
    try:                                    # present only after a previous run was accepted
        from config import RISK_EXTRA_REGIONS
        out += [dict(loc) for loc in RISK_EXTRA_REGIONS]
    except ImportError:
        pass
    return out


def load_forest_ecoregions():
    import geopandas as gpd
    if not os.path.exists(ECOREGIONS_PATH):
        sys.exit(f"ERROR: {ECOREGIONS_PATH} not found. Run download_reference_data.py first.")
    gdf = gpd.read_file(ECOREGIONS_PATH)
    gdf = gdf[gdf["BIOME_NAME"].isin(FOREST_BIOMES)].copy()
    if gdf.empty:
        sys.exit("ERROR: no forest biomes matched. Check FOREST_BIOMES against BIOME_NAME.")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    # area-weighted sampling below passes SHAPE_AREA to DataFrame.sample(weights=...), which
    # raises on NaN and on an all-zero column. Drop those rows rather than let the sampler
    # fail thousands of attempts in.
    gdf = gdf[gdf["SHAPE_AREA"].notna() & (gdf["SHAPE_AREA"] > 0)]
    if gdf.empty:
        sys.exit("ERROR: no forest polygons left after dropping empty/zero-area rows.")
    return gdf.to_crs("EPSG:4326") if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf


def sample_in_polygon(geom, rng, attempts=200):
    """Uniform point inside a polygon by rejection sampling on its bounding box."""
    from shapely.geometry import Point
    minx, miny, maxx, maxy = geom.bounds
    for _ in range(attempts):
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if geom.contains(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="EcoLens 34: generate extra forest risk regions")
    ap.add_argument("--n", type=int, default=33,
                    help="how many NEW regions to produce (33 takes 17 -> 50)")
    ap.add_argument("--min-separation-km", type=float, default=500.0,
                    help="minimum great-circle distance to every other region. Folds in "
                         "leave-one-region-out must be spatially independent; regions closer "
                         "than 2x the 15 km tiling buffer would literally share cells, and "
                         "nearby ones share drivers and disturbance regimes.")
    ap.add_argument("--min-treecover", type=float, default=float(HANSEN_TREECOVER_THRESHOLD),
                    help="reject a candidate whose mean treecover2000 over the region box is "
                         "below this. Matches the per-cell threshold used by 10_tiling, so a "
                         "region that passes here will yield a usable number of cells.")
    ap.add_argument("--allow-download", action="store_true",
                    help="allow candidates in Hansen tiles not yet on disk. Off by default: "
                         "restricting to the tiles already present keeps verification local "
                         "and means 10_tiling downloads nothing new.")
    ap.add_argument("--max-attempts", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit-config", action="store_true",
                    help="also print the RISK_EXTRA_REGIONS block to stdout")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    have = tiles_on_disk()
    print(f"Hansen tiles with both layers on disk: {len(have)}")
    if args.allow_download:
        print("  --allow-download set: candidates outside these tiles are permitted, but "
              "each new tile costs a 50-400 MB download at tiling time")

    existing = existing_regions()
    print(f"existing forest regions: {len(existing)}")

    gdf = load_forest_ecoregions()
    print(f"forest ecoregion polygons: {len(gdf)} across {gdf['BIOME_NAME'].nunique()} biomes")

    # Stratify across biomes so the expansion answers C5 (different forest types and
    # climatic zones) instead of piling more of whatever biome happens to be largest.
    by_biome = {b: g for b, g in gdf.groupby("BIOME_NAME")}
    biomes = sorted(by_biome)
    quota = {b: args.n // len(biomes) for b in biomes}
    for i in range(args.n - sum(quota.values())):      # spread the remainder
        quota[biomes[i % len(biomes)]] += 1
    print("per-biome quota: " + ", ".join(f"{b.split()[0]}={quota[b]}" for b in biomes))

    import geo_lookups as gl
    tile_cache = {}
    accepted = []
    taken = [(r["lon"], r["lat"]) for r in existing]
    rejects = Counter()
    next_id = 1 + max([int(r["id"].split("_")[-1]) for r in existing
                       if r["id"].split("_")[-1].isdigit()] or [0])

    # round-robin over biomes so a biome that is hard to place does not starve the rest
    order = [b for b in biomes for _ in range(quota[b])]
    rng.shuffle(order)
    attempts = 0
    pending = list(order)

    while pending and attempts < args.max_attempts:
        biome = pending[0]
        attempts += 1
        g = by_biome[biome]
        # weight by polygon area so large ecoregions are picked proportionally
        row = g.sample(1, random_state=rng.randrange(1 << 30), weights="SHAPE_AREA").iloc[0]
        pt = sample_in_polygon(row.geometry, rng)
        if pt is None:
            rejects["no point in polygon"] += 1
            continue
        lon, lat = float(pt.x), float(pt.y)

        tile = hansen_tile_name(lat, lon)
        if tile not in have and not args.allow_download:
            rejects["hansen tile not on disk"] += 1
            continue

        d = min((haversine_km(lon, lat, tlon, tlat) for tlon, tlat in taken),
                default=float("inf"))
        if d < args.min_separation_km:
            rejects["too close to another region"] += 1
            continue

        tc = mean_treecover(lon, lat, GRID_REGION_BUFFER_KM, tile_cache)
        if tc is None:
            rejects["treecover unreadable"] += 1
            continue
        if tc < args.min_treecover:
            rejects[f"treecover below {args.min_treecover:.0f}%"] += 1
            continue

        # enrich from the same live layers the risk model itself reads, so what is recorded
        # here is what 10_tiling will see
        try:
            protected = bool(gl.is_protected(lon, lat))
        except Exception:
            protected = None
        # get_climate returns a (temp_c, rainfall_mm) TUPLE, not a dict -- either element
        # may be None when its WorldClim raster is unavailable at that point.
        try:
            temp_c, rainfall_mm = gl.get_climate(lon, lat)
        except Exception:
            temp_c, rainfall_mm = None, None

        accepted.append({
            "id": f"forest_{next_id:03d}",
            "ecosystem": "forest",
            "lon": round(lon, 4),
            "lat": round(lat, 4),
            "name": f"{row['ECO_NAME']}, {row['REALM']}",
            "protected_area": protected,
            "climatic_region": biome,
            "selection": "auto",
            # provenance: everything needed to judge or reproduce this pick
            "_biome": biome,
            "_realm": row["REALM"],
            "_ecoregion": row["ECO_NAME"],
            "_conservation": row.get("NNH_NAME"),
            "_mean_treecover_pct": round(tc, 1),
            "_hansen_tile": tile,
            "_nearest_region_km": round(d, 1),
            "_temp_c": round(temp_c, 1) if temp_c is not None else None,
            "_rainfall_mm": round(rainfall_mm) if rainfall_mm is not None else None,
        })
        taken.append((lon, lat))
        next_id += 1
        pending.pop(0)
        print(f"  [{len(accepted):3d}/{args.n}] {accepted[-1]['id']} "
              f"({lon:8.3f},{lat:7.3f}) tc={tc:4.1f}% sep={d:6.0f}km  {biome[:34]}")

    print(f"\naccepted {len(accepted)}/{args.n} after {attempts} attempts")
    if rejects:
        print("rejections:")
        for k, v in rejects.most_common():
            print(f"  {v:6d}  {k}")
    if len(accepted) < args.n:
        print(f"\nWARNING: only {len(accepted)} of {args.n} placed. Loosen "
              f"--min-separation-km, or pass --allow-download to leave the "
              f"{len(have)} tiles already on disk.")

    print("\nspread achieved:")
    for key, label in (("_biome", "biome"), ("_realm", "realm")):
        c = Counter(r[key] for r in accepted)
        print(f"  by {label}:")
        for k, v in c.most_common():
            print(f"    {v:3d}  {k}")
    npro = sum(1 for r in accepted if r["protected_area"])
    print(f"  protected: {npro}/{len(accepted)}   unprotected: {len(accepted) - npro}")

    payload = {
        "generated_for": "review comment C5 (expand regions); risk-only, not catalogue",
        "seed": args.seed,
        "n_requested": args.n,
        "n_accepted": len(accepted),
        "min_separation_km": args.min_separation_km,
        "min_treecover_pct": args.min_treecover,
        "restricted_to_tiles_on_disk": not args.allow_download,
        "existing_regions": len(existing),
        "total_after_accept": len(existing) + len(accepted),
        "regions": accepted,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # paste-ready block, with the provenance underscore keys stripped: those belong in the
    # JSON for review, not in config
    lines = ["# Generated by 34_generate_risk_regions.py -- review before use.",
             "# Risk-only: these are tiled and forecast, but never enter the retrieval",
             "# catalogue, so all Pillar A metrics are unaffected.",
             "RISK_EXTRA_REGIONS = ["]
    for r in accepted:
        keep = {k: v for k, v in r.items() if not k.startswith("_")}
        lines.append("    {" + ", ".join(f'"{k}": {v!r}' for k, v in keep.items()) + "},")
    lines.append("]")
    block = "\n".join(lines)
    with open(OUT_PY, "w", encoding="utf-8") as f:
        f.write(block + "\n")

    for ds in tile_cache.values():
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass

    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_PY}")
    print("\nNOTHING IN config.py HAS BEEN CHANGED. Review the JSON, rename any 'name' "
          "field you like (it is display-only -- the risk model reads only lon/lat), then "
          "paste the block from the .py file into config.py.")
    if args.emit_config:
        print("\n" + block)


if __name__ == "__main__":
    main()
