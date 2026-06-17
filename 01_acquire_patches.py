"""
EcoLens Phase 1 — Step 1: Data acquisition

Searches Microsoft Planetary Computer's STAC catalog for low-cloud
Sentinel-2 L2A scenes covering each patch location, then extracts a
standardized 224x224, 6-band patch (resampled to 10m) ready for
Prithvi-100M.

Run:
    python 01_acquire_patches.py
"""

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
)


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


def main():
    os.makedirs(PATCHES_DIR, exist_ok=True)

    catalog = pystac_client.Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    catalog_entries = []
    existing_catalog = {}
    if os.path.exists(METADATA_CATALOG_PATH):
        try:
            with open(METADATA_CATALOG_PATH) as f:
                for entry in json.load(f):
                    existing_catalog[entry["id"]] = entry
        except Exception:
            pass

    for loc in PATCH_LOCATIONS:
        out_path = f"{PATCHES_DIR}/{loc['id']}.npy"
        
        # Check if patch already downloaded
        if loc["id"] in existing_catalog and os.path.exists(out_path):
            print(f"[{loc['id']}] Patch already exists. Skipping download.")
            # Ensure the entries have the correct updated config metadata fields
            entry = existing_catalog[loc["id"]]
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
            continue

        out_path = f"{PATCHES_DIR}/{loc['id']}.npy"
        np.save(out_path, patch)
        print(f"  Saved patch: {out_path}  shape={patch.shape}")

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
        })

    os.makedirs(os.path.dirname(METADATA_CATALOG_PATH), exist_ok=True)
    with open(METADATA_CATALOG_PATH, "w") as f:
        json.dump(catalog_entries, f, indent=2)

    print(f"\nDone. {len(catalog_entries)}/{len(PATCH_LOCATIONS)} patches acquired.")
    print(f"Catalog written to {METADATA_CATALOG_PATH}")


if __name__ == "__main__":
    main()
