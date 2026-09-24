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
import time
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
    MAX_SCENE_ATTEMPTS, TROPICS_LAT, BOA_ADD_OFFSET,
    STAC_SEARCH_RETRIES, STAC_RETRY_BACKOFF_S,
    MIN_VEG_NDVI, MIN_VEG_FOREST_PCT, VEGETATED_ECOSYSTEMS, ECOSYSTEM_NDVI_BANDS,
    NORTHERN_GROWING_MONTHS, SOUTHERN_GROWING_MONTHS,
)


def save_progress(catalog_entries, manifest):
    """
    Persist the catalog and provenance manifest.

    Called after EVERY successful patch, not just at the end of the run.
    A full acquisition is ~80 network round-trips over tens of minutes,
    and writing provenance only at the end means any interruption --
    a dropped connection, Ctrl+C, a STAC timeout -- discards the record
    of everything already downloaded. The patches survive on disk, but
    with no manifest entry they look unassessed, so the next run
    re-downloads all of them. That happened once: a DNS failure two
    thirds of the way through threw away the provenance for 77 patches
    that were sitting there, complete and correct.

    Two small JSON files per patch is a negligible cost next to the
    network round-trip that just completed.
    """
    os.makedirs(os.path.dirname(METADATA_CATALOG_PATH) or ".", exist_ok=True)
    with open(METADATA_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog_entries, f, indent=2)

    os.makedirs(PATCHES_DIR, exist_ok=True)
    with open(ACQUISITION_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def in_growing_season(lat, month):
    """
    True if `month` falls in the growing season for this latitude.

    Tropical sites always return True: there is no dormant season to
    avoid, so treating one as off-season would discard usable imagery
    for nothing. See config's TROPICS_LAT block for the reasoning.
    """
    if abs(lat) < TROPICS_LAT:
        return True
    if lat >= 0:
        return month in NORTHERN_GROWING_MONTHS
    return month in SOUTHERN_GROWING_MONTHS


def assess_patch_quality(patch, ecosystem=None):
    """
    Measure what a downloaded patch actually CONTAINS, independent of
    the label attached to it.

    `ecosystem`, when given, additionally requires a patch labelled
    forest or mangrove to actually hold vegetation. This is what lets
    the caller's scene-fallback loop recover from a HAZY scene rather
    than a wrong coordinate: haze suppresses NDVI without changing
    water or nodata fractions at all, so Niger Delta -- a genuinely
    dense mangrove system -- measured NDVI 0.14 and passed every other
    check. Rejecting it here makes the loop try the next scene, which
    is usually clear. If no scene has vegetation, the coordinate really
    is wrong and the failure summary says so.

    A cloud-free scene tells you the sky was clear, not that the
    coordinate points at the right thing. Three "mangrove" locations in
    this catalog sat offshore and produced patches that were 96-100%
    open water; they passed every existing check and fed straight into
    the retrieval results.

    patch: (6, H, W) array in PRITHVI_BANDS order
           (blue, green, red, nir, swir1, swir2)

    Returns {"water_fraction", "nodata_fraction", "blue_reflectance",
             "ok", "reason"}.

    blue_reflectance is a HAZE PROXY and is reported, not enforced. Thin
    haze and sub-scene cloud raise short-wavelength reflectance while
    leaving the scene-level `eo:cloud_cover` metadata untouched, because
    that number describes a ~110 km tile and this patch is 2.24 km. Two
    patches in this catalog show it plainly -- Amazon (forest_002) at
    blue 2046 / red 2350 with NDVI median 0.17, and Bhitarkanika
    (mangrove_006) at blue 1815 -- against clean patches like Periyar
    (forest_001) at blue 1368 / red 1411.

    It is deliberately NOT a rejection criterion. Every univariate haze
    threshold tested here also fires on legitimately bright surfaces:
    desert, sand, snow and dense urban are all high-blue by nature, and
    a clear winter scene over bare montane ground (forest_014) sits in
    the same range as genuine haze. Auto-rejecting on it would silently
    delete real ecosystems to remove an artefact -- the opposite of the
    trade this gate exists to make. Recording it keeps the artefact
    visible and auditable instead; 00_validate_locations.py surfaces
    outliers for a human to judge.
    """
    blue = patch[0].astype(np.float32)
    green = patch[1].astype(np.float32)
    nir = patch[3].astype(np.float32)

    # nodata shows up as an all-zero pixel across every band
    nodata_mask = np.all(patch == 0, axis=0)
    nodata_fraction = float(nodata_mask.mean())

    valid = ~nodata_mask
    if valid.sum() == 0:
        return {"water_fraction": 1.0, "nodata_fraction": 1.0,
                "blue_reflectance": None, "ndvi_median": None,
                "forest_pct": None,
                "ok": False, "reason": "patch is entirely nodata"}

    blue_reflectance = float(np.median(blue[valid]) / 10000.0)

    # NDWI > 0 marks open water
    denom = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(denom != 0, (green - nir) / denom, 0.0)
    water_fraction = float((ndwi[valid] > 0).mean())

    red = patch[2].astype(np.float32)
    swir1 = patch[4].astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndbi = (swir1 - nir) / (swir1 + nir + 1e-8)
    ndvi_median = float(np.median(ndvi[valid]))
    forest_pct = float(((ndvi > 0.45) & (ndwi < 0.1) & (ndbi < 0.1)).mean() * 100)

    reason = None
    if nodata_fraction > MAX_NODATA_FRACTION:
        reason = (f"{nodata_fraction:.0%} nodata pixels "
                  f"(limit {MAX_NODATA_FRACTION:.0%}) -- patch likely falls off the scene edge")
    elif water_fraction > MAX_WATER_FRACTION:
        reason = (f"{water_fraction:.0%} open water "
                  f"(limit {MAX_WATER_FRACTION:.0%}) -- coordinate is probably offshore")
    elif (ecosystem in VEGETATED_ECOSYSTEMS
            and ndvi_median < MIN_VEG_NDVI and forest_pct < MIN_VEG_FOREST_PCT):
        reason = (f"almost no vegetation for a '{ecosystem}' patch: median NDVI "
                  f"{ndvi_median:.2f} (min {MIN_VEG_NDVI}), forest cover "
                  f"{forest_pct:.0f}% (min {MIN_VEG_FOREST_PCT:.0f}%) -- hazy scene, "
                  f"or the coordinate is not on the ecosystem it names")
    elif ecosystem in ECOSYSTEM_NDVI_BANDS:
        # Per-category CONTENT band. MIN_VEG_NDVI above is a single global floor and
        # cannot describe a desert (should be sparse) or tundra (low but positive),
        # so those categories previously had no content check at all -- which let a
        # Greenland icecap through as "tundra" at NDVI -0.12. See config.
        lo, hi = ECOSYSTEM_NDVI_BANDS[ecosystem]
        if not (lo <= ndvi_median <= hi):
            side = "below" if ndvi_median < lo else "above"
            reason = (f"median NDVI {ndvi_median:.2f} is {side} the expected band "
                      f"[{lo}, {hi}] for a '{ecosystem}' patch -- the surface is "
                      f"probably not the ecosystem this location names "
                      f"(water/ice/snow, wrong season, or a bad coordinate)")

    return {"water_fraction": round(water_fraction, 4),
            "nodata_fraction": round(nodata_fraction, 4),
            "blue_reflectance": round(blue_reflectance, 4),
            "ndvi_median": round(ndvi_median, 4),
            "forest_pct": round(forest_pct, 2),
            "ok": reason is None,
            "reason": reason}


def search_candidate_scenes(catalog, lon, lat, buffer_deg=0.05, max_cloud_cover=None):
    """
    Query the STAC catalog for the least-cloudy Sentinel-2 L2A scene
    covering the given point, within the configured date range.

    Returns candidate pystac Items sorted clearest-first (at most
    MAX_SCENE_ATTEMPTS of them), or an empty list if nothing is found.

    Why a LIST and not just the single clearest scene:

    A STAC search matches any scene whose footprint intersects the
    bounding box. Sentinel-2 tiles carry large nodata margins, so the
    clearest matching scene can perfectly well clip the point of
    interest -- the bbox overlaps the tile, but the 2.24 km window
    around the point lands on the tile's empty edge. The extraction
    then succeeds and returns an all-zero array.

    That is not hypothetical: mangrove_001 (Sundarbans), mangrove_015
    (Shark Bay) and agri_015 (Hokkaido) all produced entirely-nodata
    patches this way. Two all-zero patches match at cosine 0.9998, so
    they went on to dominate the retrieval results -- "Shark Bay
    Mangroves" was the top analog for "Hokkaido Potato Farms".

    Cloud cover is a property of the SCENE; whether the scene actually
    covers this POINT is not knowable until the window is read. So the
    caller has to be able to fall back to the next-clearest scene, and
    that means handing it more than one candidate. The same fallback
    also absorbs transient COG read failures, which is a separate real
    failure mode (wetland_015, mangrove_010).
    """
    bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]

    # A location may raise its own cloud ceiling (config's "max_cloud_cover"
    # key). Perpetually-cloudy equatorial sites can have no scene at all under
    # the global threshold in a whole year, and losing the location entirely
    # biases the catalog more than admitting one hazier scene does. Keeping the
    # exception per-location rather than lowering the global bar means the
    # methods section can state exactly which patches were acquired under a
    # relaxed rule.
    cloud_limit = MAX_CLOUD_COVER if max_cloud_cover is None else max_cloud_cover

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=SEARCH_DATE_RANGE,
        query={"eo:cloud_cover": {"lt": cloud_limit}},
    )

    # Retry the STAC query on transient server errors.
    #
    # Planetary Computer sits behind Azure Front Door, which returns 502
    # "OriginConnectionAborted" when its backend drops a request. That is a
    # server-side hiccup with no relation to the query, but pystac_client
    # raises APIError and -- without this -- one such blip aborts an
    # 80-location run partway through. The scene-read path already tolerates
    # flaky I/O by falling back to the next candidate; the SEARCH call had no
    # equivalent, which made it the single most fragile step in acquisition.
    #
    # Exponential backoff, because if the origin is briefly overloaded then
    # retrying immediately just adds to the load.
    items = None
    for attempt in range(STAC_SEARCH_RETRIES):
        try:
            items = list(search.items())
            break
        except Exception as exc:
            if attempt == STAC_SEARCH_RETRIES - 1:
                print(f"  STAC search failed after {STAC_SEARCH_RETRIES} attempts: {exc}")
                return []
            wait = STAC_RETRY_BACKOFF_S * (2 ** attempt)
            print(f"  STAC search failed ({type(exc).__name__}); "
                  f"retrying in {wait}s [{attempt + 1}/{STAC_SEARCH_RETRIES}]")
            time.sleep(wait)

    if not items:
        return []

    # Rank by growing season FIRST, cloud cover second.
    #
    # Ranking on cloud alone systematically selects WINTER scenes outside
    # the tropics -- clear, dry, high-pressure air is exactly when there
    # is least to see. Measured on this catalog before the fix: 44 of 79
    # patches came from Jan-Mar, and forest_014 (Jiuzhaigou, a montane
    # DECIDUOUS forest at ~3000 m) was acquired on 9 January, returning
    # 0.1% forest cover because the canopy was leafless and snow-covered.
    #
    # That is not a coordinate error, and no amount of moving the point
    # fixes it. Worse, it is a confound for the whole project: if half
    # the catalog is dormant-season and half is peak-growth, retrieval
    # similarity partly encodes ACQUISITION MONTH rather than ecosystem
    # type -- precisely the stability question the faculty review raises
    # (C10). Controlling for it is a better answer than measuring it.
    #
    # Every candidate already passes the cloud threshold, so preferring
    # growing-season scenes among them cannot admit a cloudier patch than
    # the filter allows; it only reorders within an already-acceptable
    # set. Tropical sites get no penalty -- there is no dormant season to
    # avoid, and forcing one would just discard good scenes.
    def season_penalty(item):
        if abs(lat) < TROPICS_LAT:
            return 0
        month = item.datetime.month
        if lat >= 0:
            return 0 if month in NORTHERN_GROWING_MONTHS else 1
        return 0 if month in SOUTHERN_GROWING_MONTHS else 1

    items.sort(key=lambda i: (season_penalty(i), i.properties["eo:cloud_cover"]))
    return items[:MAX_SCENE_ATTEMPTS]


def boa_add_offset(item):
    """
    The BOA_ADD_OFFSET to apply to this scene's DN values, or 0.

    Prefers the product's own declared baseline, falling back to the
    acquisition date. ESA introduced the -1000 offset with processing
    baseline 04.00 on 2022-01-25; products older than that carry no
    offset and must not be shifted.
    """
    baseline = item.properties.get("s2:processing_baseline")
    if baseline:
        try:
            return BOA_ADD_OFFSET if float(baseline) >= 4.0 else 0
        except (TypeError, ValueError):
            pass  # unparseable -- fall through to the date rule

    dt = getattr(item, "datetime", None)
    if dt is not None and (dt.year, dt.month, dt.day) >= (2022, 1, 25):
        return BOA_ADD_OFFSET
    return 0


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

    patch = np.stack(stack).astype(np.float32)

    # Apply the Sentinel-2 BOA_ADD_OFFSET.
    #
    # From ESA processing baseline 04.00 (products from 2022-01-25 onward),
    # L2A surface reflectance is stored with a -1000 DN offset:
    #
    #     reflectance = (DN + BOA_ADD_OFFSET) / 10000
    #
    # Nothing in this pipeline applied it, so every band read ~0.1
    # reflectance too bright and every normalized index was compressed
    # toward zero. The effect is NOT cosmetic, because the offset does not
    # cancel in a normalized difference -- it survives in the denominator:
    #
    #     (NIR - RED) / (NIR + RED - 2000)  != (NIR - RED) / (NIR + RED)
    #
    # Measured on this catalog: Daintree rainforest read NDVI 0.57 raw and
    # 0.91 corrected; Hinchinbrook mangrove 0.56 -> 0.91; Periyar 0.48 ->
    # 0.77. The corrected values are the ecologically plausible ones -- a
    # closed tropical canopy sits near 0.85-0.92, not 0.55. Everything
    # downstream inherited the error: 02's normalization statistics, 09's
    # NDVI/NDWI/NDBI descriptors and the forest/water/urban cover fractions
    # built on them, 07b/07c which consume those descriptors, and this
    # script's own water-fraction quality gate.
    #
    # The offset is read from the product's own metadata where the STAC item
    # exposes it, so a pre-baseline-04.00 scene is left alone rather than
    # being wrongly darkened. The date fallback covers items that omit the
    # property; every scene in this project is from 2024 and therefore
    # baseline 04.00+.
    offset = boa_add_offset(item)
    if offset:
        patch = np.clip(patch + offset, 0, None)

    return patch


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

        # A patch acquired before the growing-season rule existed carries no
        # "growing_season" key, and one acquired off-season carries False.
        # Both must be re-acquired: changing the SELECTION POLICY has to
        # invalidate imagery chosen under the old one, for exactly the reason
        # a changed coordinate does. Without this the policy silently applies
        # to new patches only, and the catalog stays half winter -- the
        # confound would survive the fix meant to remove it.
        offseason = rec is not None and not rec.get("growing_season", False)
        if offseason and not is_stale and os.path.exists(out_path):
            print(f"[{loc['id']}] Acquired outside the growing season "
                  f"(month {rec.get('scene_month', '?')}); re-downloading.")

        # Patches written before the BOA_ADD_OFFSET fix hold uncorrected DN and
        # cannot be repaired in place -- the offset must be applied to the band
        # values themselves, and a patch that has already been clipped at 0 has
        # lost the information needed to undo that. Absence of the key means
        # "written by the old code", so re-acquire.
        # Every key here was added by a quality rule that did not exist when
        # older patches were written, and a patch assessed under a weaker rule
        # has not actually been checked against the current one. Listing the
        # keys rather than a version number keeps this honest: adding a new
        # measurement to the manifest automatically invalidates everything
        # acquired before it, which is what stops a fix from applying only to
        # patches downloaded after it.
        POLICY_KEYS = ("boa_offset", "ndvi_median")
        uncorrected = rec is not None and any(k not in rec for k in POLICY_KEYS)
        if uncorrected and not is_stale and not offseason and os.path.exists(out_path):
            missing = [k for k in POLICY_KEYS if k not in rec]
            print(f"[{loc['id']}] Assessed before {', '.join(missing)} existed; "
                  f"re-downloading.")

        if (os.path.exists(out_path) and rec is not None
                and not is_stale and not failed_before
                and not offseason and not uncorrected):
            print(f"[{loc['id']}] Patch already exists. Skipping download.")
            entry = dict(rec["catalog_entry"])
            entry["protected_area"] = loc.get("protected_area", False)
            entry["climatic_region"] = loc.get("climatic_region", "Unknown")
            catalog_entries.append(entry)
            continue

        print(f"\n[{loc['id']}] Searching scenes near {loc['name']} "
              f"({loc['lon']}, {loc['lat']})...")

        cloud_limit = loc.get("max_cloud_cover", MAX_CLOUD_COVER)
        if cloud_limit != MAX_CLOUD_COVER:
            print(f"  (cloud ceiling raised to {cloud_limit}% for this location)")

        candidates = search_candidate_scenes(
            catalog, loc["lon"], loc["lat"], max_cloud_cover=cloud_limit)

        if not candidates:
            print(f"  No suitable scene found for {loc['id']} — skipping. "
                  f"Try widening SEARCH_DATE_RANGE or MAX_CLOUD_COVER.")
            failures.append((loc["id"], loc["name"],
                             f"no scene under {cloud_limit}% cloud in {SEARCH_DATE_RANGE}"))
            continue

        # Try scenes clearest-first, accepting the first that yields a patch
        # passing quality control. A scene can be cloud-free and still be the
        # wrong scene for this point -- see search_candidate_scenes().
        #
        # The best_* variables track the least-bad rejected attempt so the
        # failure message distinguishes the two causes that look identical
        # from the outside: "every scene we tried clipped this point"
        # (a scene-selection problem, retry may help) versus "every scene
        # agrees this point is open water" (a COORDINATE problem, retrying
        # is pointless and the config entry needs to move).
        item = patch = quality = None
        best_item = best_patch = best_quality = None
        last_error = None

        for attempt, cand in enumerate(candidates, start=1):
            cloud_pct = cand.properties["eo:cloud_cover"]
            print(f"  [scene {attempt}/{len(candidates)}] {cand.id} "
                  f"(cloud cover: {cloud_pct:.1f}%)")

            try:
                cand_patch = extract_patch(
                    cand, loc["lon"], loc["lat"],
                    PATCH_SIZE_M, PATCH_SIZE_PX, PRITHVI_BANDS,
                )
            except Exception as e:
                # Transient COG/network read failures are common enough that
                # giving up on the location here would be wrong.
                print(f"    read failed: {e}")
                last_error = f"extraction error: {e}"
                continue

            cand_quality = assess_patch_quality(cand_patch, loc["ecosystem"])
            if cand_quality["ok"]:
                item, patch, quality = cand, cand_patch, cand_quality
                if attempt > 1:
                    print(f"    accepted after {attempt} scenes "
                          f"(the clearest scene did not cover this point)")
                break

            print(f"    rejected: {cand_quality['reason']}")
            # Rank rejects by nodata first: a nodata-heavy patch means the
            # scene missed the point, whereas a water-heavy one means the
            # point itself is wet, and the latter is the more informative
            # thing to report back.
            if best_quality is None or (
                (cand_quality["nodata_fraction"], cand_quality["water_fraction"])
                < (best_quality["nodata_fraction"], best_quality["water_fraction"])
            ):
                best_item, best_patch, best_quality = cand, cand_patch, cand_quality

        if quality is None:
            # Nothing passed. Fall back to the least-bad attempt so
            # --allow-low-quality still has something to keep.
            item, patch, quality = best_item, best_patch, best_quality

        if quality is None:
            # Every candidate failed to READ -- no patch exists at all.
            print(f"  Failed to extract any patch for {loc['id']} "
                  f"({len(candidates)} scenes tried)")
            failures.append((loc["id"], loc["name"],
                             last_error or "extraction failed on every candidate scene"))
            continue

        if not quality["ok"] and not allow_low_quality:
            print(f"  REJECTED {loc['id']}: {quality['reason']} "
                  f"(best of {len(candidates)} scenes)")
            failures.append((loc["id"], loc["name"],
                             f"{quality['reason']} [best of {len(candidates)} scenes]"))
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
            "blue_reflectance": quality["blue_reflectance"],
            "ndvi_median": quality["ndvi_median"],
            "forest_pct": quality["forest_pct"],
            "growing_season": in_growing_season(loc["lat"], item.datetime.month),
        })
        manifest[loc["id"]] = {
            "lon": loc["lon"],
            "lat": loc["lat"],
            "scene_id": item.id,
            "quality_ok": quality["ok"],
            "water_fraction": quality["water_fraction"],
            "nodata_fraction": quality["nodata_fraction"],
            "blue_reflectance": quality["blue_reflectance"],
            # These two are what the staleness check reads to decide a patch
            # was assessed under the current quality policy. Omitting them
            # makes every location look unassessed on every run, so 01 never
            # converges -- it re-downloads the entire catalog each time.
            "ndvi_median": quality["ndvi_median"],
            "forest_pct": quality["forest_pct"],
            "scene_month": item.datetime.month,
            "growing_season": in_growing_season(loc["lat"], item.datetime.month),
            "boa_offset": boa_add_offset(item),
            "catalog_entry": catalog_entries[-1],
        }

        # Checkpoint immediately: this patch is downloaded, assessed and on
        # disk, and that fact should survive whatever happens to the next one.
        save_progress(catalog_entries, manifest)

    save_progress(catalog_entries, manifest)

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
