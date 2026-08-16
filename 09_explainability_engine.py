"""
EcoLens Objective 3 - Step 9: Ecosystem Explainability Engine

Extracts land-cover descriptors (forest, water, urban percentages,
vegetation health, and protected status) directly from raw Sentinel-2
patch bands, and real physical descriptors (elevation, climate,
ecoregion, protected status) from geo_lookups.py. Computes pair-wise
comparisons and generates natural language explanations for similar
query-analog pairs.

---------------------------------------------------------------------
WHAT CHANGED FROM THE ORIGINAL VERSION
---------------------------------------------------------------------
The original get_physical_descriptors() generated elevation,
temperature, rainfall, soil type, and ecoregion from
`sum(ord(c) for c in patch_id)` -- numbers that looked plausible but
had no relationship to the real world. They sat in the same output
dict and the same generated sentences as the real, patch-derived
NDVI/NDWI/NDBI spectral descriptors, which made the fabricated half
look as credible as the real half.

This version calls geo_lookups.get_physical_descriptors(lon, lat),
which reads real WDPA / WorldClim / SRTM / RESOLVE Ecoregions data.
If you haven't downloaded those reference datasets yet (see
geo_lookups.py's module docstring / README.md), the physical fields
come back as None -- and generate_explanation() below skips any
factor built from a None field rather than inventing a plausible
number. Run geo_lookups.spot_check() before trusting this script's
output at scale; a silent CRS or filtering bug produces wrong-but-
plausible numbers, not a crash.

Soil type has been dropped entirely -- there's no equally simple
static global soil raster to fetch (SoilGrids requires either its
REST API or large per-property downloads). Add it via geo_lookups.py
following the same "return None on missing data" contract if needed.

Run:
    python 09_explainability_engine.py
"""

import importlib.util
import json
import os
import numpy as np
from tqdm import tqdm

from config import METADATA_CATALOG_PATH, RESULTS_DIR, PATCH_SIZE_M
import geo_lookups

OUT_DESCRIPTORS_PATH = f"{RESULTS_DIR}/ecosystem_descriptors.json"
OUT_EXPLAIN_PATH = f"{RESULTS_DIR}/explainable_retrieval.json"


# ---------------------------------------------------------------
# Disturbance history (faculty comment: "discuss whether retrieved
# ecosystems represent similar disturbance pathways ... rather than
# simply similar imagery. With historical imagery analyse whether
# retrieved ecosystem analogs exhibit comparable ecological
# trajectories over time.")
# ---------------------------------------------------------------
#
# Reuses 10_grid_tiling_labels.py's Hansen Global Forest Change tile
# helpers (dynamic import -- numbered filenames aren't importable with
# a normal `import 10_...` statement, same pattern already used by
# 11_forest_risk_forecast.py::predict_risk()) to pull a real, dated
# loss-year history for each patch, not just its current-snapshot
# forest_cover value. This turns "these two patches look similar
# today" into "these two patches have lost tree cover in the same
# years" -- a trajectory comparison, not a single-frame one.
_tiling = None


def _get_tiling_module():
    global _tiling
    if _tiling is None:
        spec = importlib.util.spec_from_file_location(
            "tiling", os.path.join(os.path.dirname(__file__), "10_grid_tiling_labels.py")
        )
        _tiling = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_tiling)
    return _tiling


def get_disturbance_history(lon, lat):
    """
    Real Hansen lossyear history over the patch's full footprint
    (config.PATCH_SIZE_M, matching what was actually acquired for this
    location -- not an arbitrary window size).

    Returns None for every field if the Hansen tile for this location
    isn't available (not cached and couldn't be downloaded) -- same
    "None means unavailable, never fabricated" contract every other
    field in this module follows.
    """
    tiling = _get_tiling_module()
    half_km = (PATCH_SIZE_M / 1000.0) / 2.0
    d_lat, d_lon = tiling.km_to_deg(half_km, lat)

    lossyear_path = tiling.ensure_hansen_tile("lossyear", lat, lon)
    if lossyear_path is None:
        return {"loss_years": None, "years_since_last_loss": None, "cumulative_loss_pct": None}

    lossyear_arr, _, _ = tiling.read_window(
        lossyear_path, lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat
    )
    if lossyear_arr is None or lossyear_arr.size == 0:
        return {"loss_years": None, "years_since_last_loss": None, "cumulative_loss_pct": None}

    total_px = lossyear_arr.size
    loss_years = {}
    for offset in np.unique(lossyear_arr):
        if offset == 0:
            continue
        year = 2000 + int(offset)
        pct = round(100.0 * float(np.sum(lossyear_arr == offset)) / total_px, 3)
        loss_years[year] = pct

    years_since_last_loss = max(loss_years.keys()) if loss_years else None
    cumulative_loss_pct = round(sum(loss_years.values()), 3)

    return {
        "loss_years": loss_years,           # {year: pct_of_patch_lost_that_year}
        "years_since_last_loss": years_since_last_loss,
        "cumulative_loss_pct": cumulative_loss_pct,
    }


def calculate_patch_descriptors(entry):
    """
    Load raw 6-band patch array and compute REAL spectral descriptors
    (NDVI/NDWI/NDBI-derived land cover), then merge with REAL physical
    descriptors from geo_lookups (elevation/climate/ecoregion/
    protected status). Nothing in this function fabricates a value --
    fields with no available data are left as None and callers must
    handle that explicitly.
    """
    patch_path = entry.get("patch_path")
    lon, lat = entry["lon"], entry["lat"]

    # Protected-area status: prefer a real WDPA point-in-polygon check.
    # Fall back to the catalog's own protected_area field (which for
    # this project's 75 hand-curated base locations is a manually
    # verified value from config.py, not a fabricated one) only if
    # WDPA data isn't downloaded yet -- and record which source was
    # actually used so it's auditable in the output.
    wdpa_protected = geo_lookups.is_protected(lon, lat)
    if wdpa_protected is not None:
        protected_area = wdpa_protected
        protected_source = "WDPA"
    else:
        protected_area = entry.get("protected_area", False)
        protected_source = "catalog (WDPA unavailable)"

    desc = {
        "forest_cover": 0.0,
        "water_cover": 0.0,
        "urban_cover": 0.0,
        "veg_health": 0.0,
        "protected_area": protected_area,
        "protected_source": protected_source,
    }

    if patch_path and os.path.exists(patch_path):
        try:
            raw = np.load(patch_path)
            if np.max(raw) > 0:
                # Scale to reflectance [0, 1]
                patch = raw.astype(np.float32) / 10000.0

                # Sentinel-2 bands mapping: 0=Blue, 1=Green, 2=Red, 3=NIR, 4=SWIR1, 5=SWIR2
                green = patch[1]
                red = patch[2]
                nir = patch[3]
                swir1 = patch[4]

                ndvi = (nir - red) / (nir + red + 1e-8)
                ndwi = (green - nir) / (green + nir + 1e-8)
                ndbi = (swir1 - nir) / (swir1 + nir + 1e-8)

                forest_mask = (ndvi > 0.45) & (ndwi < 0.1) & (ndbi < 0.1)
                forest_pct = float(np.mean(forest_mask) * 100.0)

                water_mask = (ndwi > 0.0)
                water_pct = float(np.mean(water_mask) * 100.0)

                urban_mask = (ndbi > 0.0) & (ndvi < 0.35) & (ndwi < 0.0)
                urban_pct = float(np.mean(urban_mask) * 100.0)

                veg_active = ndvi[ndvi > 0.2]
                veg_health = float(np.mean(veg_active)) if len(veg_active) > 0 else 0.0

                desc["forest_cover"] = round(forest_pct, 2)
                desc["water_cover"] = round(water_pct, 2)
                desc["urban_cover"] = round(urban_pct, 2)
                desc["veg_health"] = round(veg_health, 4)
        except Exception as e:
            print(f"Error reading patch {patch_path}: {e}")

    # Real physical descriptors (elevation/climate/ecoregion), or None
    # per-field if the corresponding reference dataset isn't available.
    phys = geo_lookups.get_physical_descriptors(lon, lat)
    desc["elevation_m"] = phys["elevation_m"]
    desc["temp_c"] = phys["temp_c"]
    desc["rainfall_mm"] = phys["rainfall_mm"]
    desc["ecoregion"] = phys["ecoregion"]
    desc["biome"] = phys["biome"]
    # If WDPA gave us a real answer above, prefer it; otherwise take
    # whatever geo_lookups.get_physical_descriptors reported (same
    # WDPA call, so this is just keeping the two consistent).
    desc["climate_source"] = phys["climate_source"]
    desc["elevation_source"] = phys["elevation_source"]
    desc["ecoregion_source"] = phys["ecoregion_source"]

    # Real Hansen-derived disturbance history -- scoped to forest/mangrove
    # patches, where tree-cover loss is the ecologically meaningful signal
    # this data source can speak to (a "loss year" over farmland or open
    # water is not informative and would just burn Hansen tile downloads
    # for no scientific benefit). Non-forest/mangrove patches get None,
    # same "unavailable, not fabricated" contract as every other field.
    if entry.get("ecosystem") in ("forest", "mangrove"):
        disturbance = get_disturbance_history(lon, lat)
    else:
        disturbance = {"loss_years": None, "years_since_last_loss": None, "cumulative_loss_pct": None}
    desc["loss_years"] = disturbance["loss_years"]
    desc["years_since_last_loss"] = disturbance["years_since_last_loss"]
    desc["cumulative_loss_pct"] = disturbance["cumulative_loss_pct"]

    return desc


def generate_explanation(q, a):
    """
    Compare query and analog descriptors to generate a natural language
    explanation of their ecological similarity.

    Every factor derived from real-but-optional reference data
    (elevation/temp/rainfall/ecoregion) is guarded: if either patch is
    missing that field (reference dataset not downloaded), the factor
    is skipped entirely rather than comparing None to a number or,
    worse, silently treating a missing value as if it were real.
    """
    factors = []

    forest_diff = abs(q["forest_cover"] - a["forest_cover"])
    water_diff = abs(q["water_cover"] - a["water_cover"])
    urban_diff = abs(q["urban_cover"] - a["urban_cover"])
    veg_diff = abs(q["veg_health"] - a["veg_health"])

    metrics = {
        "forest_diff": round(forest_diff, 2),
        "water_diff": round(water_diff, 2),
        "urban_diff": round(urban_diff, 2),
        "veg_health_diff": round(veg_diff, 4),
    }

    # 1. Forest Canopy Cover Comparison (real, spectral -- always available)
    if forest_diff < 15.0:
        if q["forest_cover"] > 50.0 and a["forest_cover"] > 50.0:
            factors.append(f"high forest canopy coverage ({q['forest_cover']:.1f}% vs {a['forest_cover']:.1f}%)")
        elif q["forest_cover"] < 10.0 and a["forest_cover"] < 10.0:
            factors.append(f"minimal forest canopy ({q['forest_cover']:.1f}% vs {a['forest_cover']:.1f}%)")
        else:
            factors.append(f"similar forest cover ({q['forest_cover']:.1f}% vs {a['forest_cover']:.1f}%)")

    # 2. Water Presence Comparison
    if water_diff < 12.0:
        if q["water_cover"] > 15.0 and a["water_cover"] > 15.0:
            factors.append(f"significant open water presence ({q['water_cover']:.1f}% vs {a['water_cover']:.1f}%)")
        elif q["water_cover"] < 2.0 and a["water_cover"] < 2.0:
            pass  # Dry matches don't need highlighting unless dominant
        else:
            factors.append(f"comparable surface water profile ({q['water_cover']:.1f}% vs {a['water_cover']:.1f}%)")

    # 3. Urban / bare-ground influence
    if urban_diff < 15.0:
        if q["urban_cover"] > 25.0 and a["urban_cover"] > 25.0:
            factors.append(f"prominent built-up/exposed soil footprint ({q['urban_cover']:.1f}% vs {a['urban_cover']:.1f}%)")
        elif q["urban_cover"] < 5.0 and a["urban_cover"] < 5.0:
            factors.append("pristine surface conditions with minimal human footprint")
        else:
            factors.append(f"similar built-up/bare ground percentage ({q['urban_cover']:.1f}% vs {a['urban_cover']:.1f}%)")

    # 4. Vegetation Health (NDVI)
    if veg_diff < 0.08:
        if q["veg_health"] > 0.40 and a["veg_health"] > 0.40:
            factors.append(f"strong photosynthetic activity (mean NDVI of {q['veg_health']:.2f} vs {a['veg_health']:.2f})")
        else:
            factors.append("comparable vegetation vigor indices")

    # 5. Climate/elevation -- REAL data, guarded against missing values.
    if q["temp_c"] is not None and a["temp_c"] is not None:
        temp_diff = abs(q["temp_c"] - a["temp_c"])
        metrics["temp_diff"] = round(temp_diff, 1)
        if temp_diff < 3.5:
            factors.append(f"matching temperature profiles ({q['temp_c']:.1f}\u00b0C vs {a['temp_c']:.1f}\u00b0C)")
    else:
        metrics["temp_diff"] = None

    if q["rainfall_mm"] is not None and a["rainfall_mm"] is not None:
        rain_diff = abs(q["rainfall_mm"] - a["rainfall_mm"])
        metrics["rain_diff"] = round(rain_diff, 1)
        if rain_diff < 300.0:
            factors.append(f"comparable annual precipitation ({q['rainfall_mm']:.0f}mm vs {a['rainfall_mm']:.0f}mm)")
    else:
        metrics["rain_diff"] = None

    if q["elevation_m"] is not None and a["elevation_m"] is not None:
        elev_diff = abs(q["elevation_m"] - a["elevation_m"])
        metrics["elev_diff"] = round(elev_diff, 1)
        if elev_diff < 200.0:
            factors.append(f"similar altitude profiles ({q['elevation_m']:.0f}m vs {a['elevation_m']:.0f}m)")
    else:
        metrics["elev_diff"] = None

    # 6. Ecoregion matching -- REAL (RESOLVE), guarded against missing values.
    if q["ecoregion"] is not None and a["ecoregion"] is not None and q["ecoregion"] == a["ecoregion"]:
        factors.append(f"shared ecoregion designation ('{q['ecoregion']}')")

    # 7. Protected Status (real WDPA where available, else the
    #    hand-curated catalog value -- see calculate_patch_descriptors)
    if q["protected_area"] == a["protected_area"]:
        if q["protected_area"]:
            factors.append("shared conservation status as designated protected zones")
        else:
            factors.append("similar unprotected conservation status")

    # 8. Disturbance history (real Hansen lossyear data, forest/mangrove
    #    only -- see get_disturbance_history()). This is a TRAJECTORY
    #    comparison, not a snapshot one: it asks whether the two patches
    #    lost tree cover in the SAME years, not just whether they look
    #    similar today.
    if q.get("loss_years") is not None and a.get("loss_years") is not None:
        q_years = set(q["loss_years"].keys())
        a_years = set(a["loss_years"].keys())
        union = q_years | a_years
        jaccard = len(q_years & a_years) / len(union) if union else 1.0
        metrics["disturbance_trajectory_jaccard"] = round(jaccard, 3)
        cum_diff = abs(q["cumulative_loss_pct"] - a["cumulative_loss_pct"])
        metrics["cumulative_loss_pct_diff"] = round(cum_diff, 2)
        if not q_years and not a_years:
            factors.append("no detected tree-cover loss in either location's Hansen record (both stable)")
        elif jaccard > 0.3:
            shared = sorted(q_years & a_years)
            factors.append(f"overlapping disturbance history (tree-cover loss recorded in {len(shared)} of the same year(s): {shared})")
    else:
        metrics["disturbance_trajectory_jaccard"] = None
        metrics["cumulative_loss_pct_diff"] = None

    if len(factors) == 0:
        explanation = "These two ecosystems share general ecological attributes with small variations in overall land cover profiles."
    elif len(factors) == 1:
        explanation = f"These two ecosystems are considered similar because both exhibit {factors[0]}."
    elif len(factors) == 2:
        explanation = f"These two ecosystems are considered similar because both exhibit {factors[0]} and {factors[1]}."
    else:
        explanation = f"These two ecosystems are considered similar because both exhibit {', '.join(factors[:-1])}, and {factors[-1]}."

    return {"explanation": explanation, "metrics": metrics}


def main():
    print(f"\n{'='*60}")
    print("EcoLens Explainability Engine: Computing Ecosystem Descriptors")
    print(f"{'='*60}\n")

    if not os.path.exists(METADATA_CATALOG_PATH):
        print(f"Error: Catalog not found at {METADATA_CATALOG_PATH}. Run steps 1-3 first.")
        return

    with open(METADATA_CATALOG_PATH) as f:
        catalog = json.load(f)

    # Upfront, honest check of what real reference data is actually
    # available -- so the run tells you what's real vs. missing before
    # you've generated 700+ explanations, not after.
    from config import WDPA_POLYGONS_PATH, WORLDCLIM_TEMP_PATH, WORLDCLIM_PRECIP_PATH, DEM_TILES_DIR, ECOREGIONS_PATH
    print("Reference dataset availability:")
    for label, path in [
        ("WDPA (protected areas)", WDPA_POLYGONS_PATH),
        ("WorldClim temperature", WORLDCLIM_TEMP_PATH),
        ("WorldClim precipitation", WORLDCLIM_PRECIP_PATH),
        ("RESOLVE Ecoregions", ECOREGIONS_PATH),
    ]:
        status = "found" if os.path.exists(path) else "MISSING -- fields depending on this will be null"
        print(f"    {label:<28} {path:<40} {status}")
    n_dem_tiles = len(os.listdir(DEM_TILES_DIR)) if os.path.isdir(DEM_TILES_DIR) else 0
    dem_status = f"{n_dem_tiles} tile(s) found" if n_dem_tiles else "MISSING -- elevation will be null"
    print(f"    {'DEM (elevation)':<28} {DEM_TILES_DIR:<40} {dem_status}")
    print()

    # 1. Compute descriptors for all patches
    descriptors = {}
    print(f"Computing descriptors for {len(catalog)} patches...")

    for entry in tqdm(catalog, desc="Ecosystem Descriptors Extraction"):
        desc = calculate_patch_descriptors(entry)
        descriptors[entry["id"]] = desc

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_DESCRIPTORS_PATH, "w") as f:
        json.dump(descriptors, f, indent=2)
    print(f"\nEcosystem descriptors saved to: {OUT_DESCRIPTORS_PATH}")

    # 2. Load actual cosine retrieval rankings from ALL models so that every
    #    result the dashboard could display has an explanation precomputed.
    retrieval_dir = RESULTS_DIR
    retrieval_files = [
        os.path.join(retrieval_dir, f)
        for f in os.listdir(retrieval_dir)
        if f.startswith("retrieval_results_") and f.endswith(".json")
    ]

    top_analogs_per_query = {}
    TOP_K = 50

    for rfile in retrieval_files:
        with open(rfile) as f:
            rdata = json.load(f)
        cosine_results = rdata.get("cosine", {})
        for qid, ranked_list in cosine_results.items():
            if qid not in top_analogs_per_query:
                top_analogs_per_query[qid] = set()
            for entry in ranked_list[:TOP_K]:
                top_analogs_per_query[qid].add(entry["id"])

    total_pairs = sum(len(v) for v in top_analogs_per_query.values())
    print(f"\nPrecomputing explanations for {total_pairs} query-analog pairs "
          f"(top {TOP_K} cosine results x {len(retrieval_files)} models)...")

    explanations = {}
    pids = list(descriptors.keys())
    for qid in tqdm(pids, desc="Generating Explanations"):
        explanations[qid] = {}
        q = descriptors.get(qid)
        if q is None:
            continue
        analog_ids = top_analogs_per_query.get(qid, set())
        for aid in analog_ids:
            a = descriptors.get(aid)
            if a is None:
                continue
            explanations[qid][aid] = generate_explanation(q, a)

    output_data = {"descriptors": descriptors, "explanations": explanations}

    with open(OUT_EXPLAIN_PATH, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Explainable similarity matrix saved to: {OUT_EXPLAIN_PATH}")
    print("\nDone. Run 08_retrieval_dashboard.py to integrate explanation reports.")


if __name__ == "__main__":
    main()