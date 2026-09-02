"""
EcoLens Phase 1 — Step 1: Data acquisition

Searches Microsoft Planetary Computer's STAC catalog for low-cloud
Sentinel-2 L2A scenes covering each patch location, then extracts a
standardized 224x224, 6-band patch (resampled to 10m) ready for
Prithvi-100M.

Run:
    python 01_acquire_patches.py
"""

import argparse
import json
import os
import numpy as np
import pystac_client
import planetary_computer
import rasterio
from rasterio.warp import transform, Resampling
from rasterio.windows import from_bounds

from config import (
    PC_STAC_URL, SEARCH_DATE_RANGE, MAX_CLOUD_COVER,
    PATCH_SIZE_M, PATCH_SIZE_PX, PRITHVI_BANDS,
    PATCH_LOCATIONS, PATCHES_DIR, METADATA_CATALOG_PATH,
    MAX_WATER_FRACTION, MAX_NODATA_FRACTION, ACQUISITION_MANIFEST_PATH,
)


def assess_patch_quality(patch):
    """
    Measure what a downloaded patch actually CONTAINS, independent of
    the label attached to it.

    A cloud-free scene tells you the sky was clear, not that the
    coordinate points at the right thing. Three "mangrove" locations in
    this catalog sat offshore and produced patches that were 96-100%
    open water; they passed every existing check and fed straight into
    the retrieval results.

    patch: (6, H, W) array in PRITHVI_BANDS order
           (blue, green, red, nir, swir1, swir2)

    Returns {"water_fraction", "nodata_fraction", "ok", "reason"}.
    """
    green = patch[1].astype(np.float32)
    nir = patch[3].astype(np.float32)

    # nodata shows up as an all-zero pixel across every band
    nodata_mask = np.all(patch == 0, axis=0)
    nodata_fraction = float(nodata_mask.mean())

    valid = ~nodata_mask
    if valid.sum() == 0:
        return {"water_fraction": 1.0, "nodata_fraction": 1.0,
                "ok": False, "reason": "patch is entirely nodata"}

    # NDWI > 0 marks open water
    denom = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(denom != 0, (green - nir) / denom, 0.0)
    water_fraction = float((ndwi[valid] > 0).mean())

    reason = None
    if nodata_fraction > MAX_NODATA_FRACTION:
        reason = (f"{nodata_fraction:.0%} nodata pixels "
                  f"(limit {MAX_NODATA_FRACTION:.0%}) -- patch likely falls off the scene edge")
    elif water_fraction > MAX_WATER_FRACTION:
        reason = (f"{water_fraction:.0%} open water "
                  f"(limit {MAX_WATER_FRACTION:.0%}) -- coordinate is probably offshore")

    return {"water_fraction": round(water_fraction, 4),
            "nodata_fraction": round(nodata_fraction, 4),
            "ok": reason is None,
            "reason": reason}


def search_best_scene(catalog, lon, lat, buffer_deg=0.05):
    """
    Query the STAC catalog for the least-cloudy Sentinel-2 L2A scene
    covering the given point, within the configured date range.

    Returns the matching pystac Item, or None if nothing is found.
    """
    bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=SEARCH_DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
    )

    items = list(search.items())
    if not items:
        return None

    # Pick the clearest scene available
    return min(items, key=lambda i: i.properties["eo:cloud_cover"])


def extract_patch(item, lon, lat, patch_size_m, patch_size_px, band_map):
    """
    Extract a fixed-size, multi-band patch centered on (lon, lat).

    Bands at different native resolutions (10m / 20m) are all
    resampled to the same patch_size_px x patch_size_px grid so
    they stack cleanly into a single array.

    Returns a numpy array of shape (n_bands, patch_size_px, patch_size_px),
    band order matching the order of band_map's values.
    """
    stack = []

    for band_name, band_code in band_map.items():
        href = item.assets[band_code].href

        with rasterio.open(href) as src:
            # Reproject the center point from WGS84 into the scene's
            # native CRS (Sentinel-2 scenes are stored in UTM zones)
            cx, cy = transform("EPSG:4326", src.crs, [lon], [lat])
            cx, cy = cx[0], cy[0]

            half = patch_size_m / 2
            window = from_bounds(cx - half, cy - half, cx + half, cy + half, src.transform)

            arr = src.read(
                1,
                window=window,
                out_shape=(patch_size_px, patch_size_px),
                resampling=Resampling.bilinear,
            )
            stack.append(arr)

    return np.stack(stack).astype(np.float32)


def main(allow_low_quality=False):
    os.makedirs(PATCHES_DIR, exist_ok=True)

    catalog = pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    catalog_entries = []
    failures = []
    existing_catalog = {}
    if os.path.exists(METADATA_CATALOG_PATH):
        try:
            with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
                for entry in json.load(f):
                    existing_catalog[entry["id"]] = entry
        except Exception:
            pass

    # Provenance manifest: the exact coordinates each patch on disk was
    # actually downloaded from.
    #
    # catalog.json cannot answer this. Step 02 rewrites it with sub-crop
    # entries whose coordinates are jittered by up to ~0.012 deg, which is
    # the same order as a genuine small coordinate correction -- so any
    # tolerance wide enough to ignore the jitter is also wide enough to
    # miss a real fix. This manifest records the base coordinate exactly,
    # so staleness becomes an equality test rather than a guess.
    manifest = {}
    if os.path.exists(ACQUISITION_MANIFEST_PATH):
        try:
            with open(ACQUISITION_MANIFEST_PATH, encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    for loc in PATCH_LOCATIONS:
        out_path = f"{PATCHES_DIR}/{loc['id']}.npy"

        # A patch is reusable only if it came from the coordinates config
        # specifies today. Correcting a coordinate must invalidate imagery
        # downloaded from the old one, or the fix is silently ignored and
        # stale pixels keep feeding embeddings, retrieval and every metric.
        rec = manifest.get(loc["id"])
        is_stale = rec is not None and (
            abs(rec.get("lon", 1e9) - loc["lon"]) > 1e-6
            or abs(rec.get("lat", 1e9) - loc["lat"]) > 1e-6
        )
        if is_stale and os.path.exists(out_path):
            print(f"[{loc['id']}] Coordinates changed "
                  f"({rec['lon']:.4f}, {rec['lat']:.4f}) -> "
                  f"({loc['lon']:.4f}, {loc['lat']:.4f}); re-downloading.")

        # Never reuse a patch that failed quality control last time -- give
        # the corrected coordinate a fresh attempt.
        failed_before = rec is not None and not rec.get("quality_ok", True)

        if (os.path.exists(out_path) and rec is not None
                and not is_stale and not failed_before):
            print(f"[{loc['id']}] Patch already exists. Skipping download.")
            entry = dict(rec["catalog_entry"])
            entry["protected_area"] = loc.get("protected_area", False)
            entry["climatic_region"] = loc.get("climatic_region", "Unknown")
            catalog_entries.append(entry)
            continue

        print(f"\n[{loc['id']}] Searching scenes near {loc['name']} "
              f"({loc['lon']}, {loc['lat']})...")

        item = search_best_scene(catalog, loc["lon"], loc["lat"])

        if item is None:
            print(f"  No suitable scene found for {loc['id']} — skipping. "
                  f"Try widening SEARCH_DATE_RANGE or MAX_CLOUD_COVER.")
            failures.append((loc["id"], loc["name"],
                             f"no scene under {MAX_CLOUD_COVER}% cloud in {SEARCH_DATE_RANGE}"))
            continue

        cloud_pct = item.properties["eo:cloud_cover"]
        print(f"  Found scene {item.id} (cloud cover: {cloud_pct:.1f}%)")

        try:
            patch = extract_patch(
                item, loc["lon"], loc["lat"],
                PATCH_SIZE_M, PATCH_SIZE_PX, PRITHVI_BANDS,
            )
        except Exception as e:
            print(f"  Failed to extract patch for {loc['id']}: {e}")
            failures.append((loc["id"], loc["name"], f"extraction error: {e}"))
            continue

        quality = assess_patch_quality(patch)
        if not quality["ok"] and not allow_low_quality:
            print(f"  REJECTED {loc['id']}: {quality['reason']}")
            failures.append((loc["id"], loc["name"], quality["reason"]))
            continue
        if not quality["ok"]:
            print(f"  WARNING {loc['id']}: {quality['reason']} (kept: --allow-low-quality)")

        np.save(out_path, patch)
        print(f"  Saved patch: {out_path}  shape={patch.shape} "
              f"(water {quality['water_fraction']:.0%}, nodata {quality['nodata_fraction']:.0%})")

        catalog_entries.append({
            "id": loc["id"],
            "ecosystem": loc["ecosystem"],
            "name": loc["name"],
            "lon": loc["lon"],
            "lat": loc["lat"],
            "protected_area": loc.get("protected_area", False),
            "climatic_region": loc.get("climatic_region", "Unknown"),
            "scene_id": item.id,
            "scene_date": item.properties.get("datetime"),
            "cloud_cover_pct": cloud_pct,
            "patch_path": out_path,
            "patch_shape": list(patch.shape),
            "bands": list(PRITHVI_BANDS.keys()),
            "water_fraction": quality["water_fraction"],
            "nodata_fraction": quality["nodata_fraction"],
        })
        manifest[loc["id"]] = {
            "lon": loc["lon"],
            "lat": loc["lat"],
            "scene_id": item.id,
            "quality_ok": quality["ok"],
            "water_fraction": quality["water_fraction"],
            "nodata_fraction": quality["nodata_fraction"],
            "catalog_entry": catalog_entries[-1],
        }

    os.makedirs(os.path.dirname(METADATA_CATALOG_PATH), exist_ok=True)
    with open(METADATA_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog_entries, f, indent=2)

    os.makedirs(PATCHES_DIR, exist_ok=True)
    with open(ACQUISITION_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(catalog_entries)}/{len(PATCH_LOCATIONS)} patches acquired.")
    print(f"Catalog written to {METADATA_CATALOG_PATH}")

    # A per-location warning scrolls past in an 81-location run. Ten
    # locations were missing from this catalog for weeks because the only
    # signal was one line of output somewhere in the middle. Print the
    # roll-up at the end, where it cannot be missed.
    if failures:
        print()
        print("=" * 68)
        print(f"{len(failures)} LOCATION(S) PRODUCED NO USABLE PATCH")
        print("=" * 68)
        for pid, name, reason in failures:
            print(f"  {pid:18} {name[:38]:40} {reason}")
        print()
        print("  Fix the coordinate, widen SEARCH_DATE_RANGE / MAX_CLOUD_COVER,")
        print("  or pass --allow-low-quality to keep patches that failed the")
        print("  water/nodata check anyway.")
    else:
        print()
        print("All configured locations produced a patch that passed quality control.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire Sentinel-2 patches for every configured location.")
    parser.add_argument("--allow-low-quality", action="store_true",
                        help="Keep patches that fail the water/nodata check instead "
                             "of rejecting them. They are still flagged in the "
                             "catalog and in the end-of-run summary.")
    args = parser.parse_args()
    main(allow_low_quality=args.allow_low_quality)
