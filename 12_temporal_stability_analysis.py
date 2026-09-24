"""
EcoLens Objective 2 - Step 12: Temporal Stability Analysis

Evaluates whether the foundation model embeddings are temporally robust.
Downloads a summer patch (July) and a winter patch (January) for a subset
of locations, extracts their embeddings, and computes cosine similarity.
High similarity indicates the model captures stable ecosystem traits
rather than transient seasonal artifacts.
"""

import os
import sys
import json
import numpy as np
import pystac_client
import planetary_computer
import torch
import warnings
from scipy.spatial.distance import cosine

# Suppress noisy STAC/Rasterio warnings
warnings.filterwarnings("ignore")

from config import (
    PC_STAC_URL, MAX_CLOUD_COVER, PATCH_SIZE_M, PATCH_SIZE_PX, 
    PRITHVI_BANDS, PATCH_LOCATIONS, SUPPORTED_MODELS, DEFAULT_MODEL
)
from importlib import import_module
acq = import_module("01_acquire_patches")
ext = import_module("03_extract_embeddings")

# 13 owns the single definition of "embed exactly the way 03 built the catalog"
# (crop geometry + per-model normalisation). 12 defers to it rather than keeping a
# second copy that can drift -- see _embed() below for what that drift already cost.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_a13", "13_analog_risk_features.py")
a13 = _ilu.module_from_spec(_spec)
sys.modules["_a13"] = a13
_spec.loader.exec_module(a13)

# Define the seasons
SEASONS = {
    "summer": "2023-07-01/2023-08-31",
    "winter": "2023-01-01/2023-02-28"
}

# ACQUISITION-DATE AXIS (C10: "...and acquisition date").
# The same growing-season window two years apart. Season is held fixed and haze is
# controlled by taking the clearest scene on both sides, so what remains is the ordinary
# year-to-year variation a real catalog will contain -- different phenological timing
# within the window, different crop rotation, different antecedent rainfall.
#
# This is the axis that matters MOST operationally: our catalog is acquired once, and every
# future query image will come from a different year. If the embedding cannot recognise a
# place across two Julys, the catalog goes stale.
SEASONS["summer_prev"] = "2021-07-01/2021-08-31"

def search_scene(catalog, lon, lat, date_range, buffer_deg=0.05, attempts=4,
                 cloud_lt=MAX_CLOUD_COVER, cloud_gte=None, pick="clearest"):
    """One scene in the window, chosen by cloud cover, with retry.

    A single Azure `OriginConnectionAborted` killed this script's Satlas run on
    4 Sep after four models had already completed. That is the same transient
    failure implementation.md 3.1 records as having "aborted a whole run" of 01 --
    01 grew a retry for it; this script never did. Backoff added so one bad response
    costs one scene, not the whole job.

    `pick` exists for the ATMOSPHERIC axis: the seasonal test always wants the clearest
    available scene, but the atmospheric test deliberately wants a HAZY one, so the
    same query has to be able to return either end of the cloud range.
    """
    import time
    bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
    q = {"lt": cloud_lt}
    if cloud_gte is not None:
        q["gte"] = cloud_gte
    delay = 3.0
    for i in range(attempts):
        try:
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=date_range,
                query={"eo:cloud_cover": q},
            )
            items = list(search.items())
            break
        except Exception as e:
            if i == attempts - 1:
                print(f"  [Warn] STAC search failed after {attempts} attempts: "
                      f"{type(e).__name__}")
                return None
            time.sleep(delay)
            delay *= 2
    if not items:
        return None
    chooser = max if pick == "cloudiest" else min
    return chooser(items, key=lambda i: i.properties["eo:cloud_cover"])


def search_seasonal_scene(catalog, lon, lat, date_range, buffer_deg=0.05, attempts=4):
    """Least-cloudy scene in the window (the seasonal axis)."""
    return search_scene(catalog, lon, lat, date_range, buffer_deg, attempts)


# ---------------------------------------------------------------------------
# ATMOSPHERIC AXIS (C10: "...and atmospheric conditions")
#
# The seasonal axis asks "does the embedding survive a different time of year?".
# This asks the operationally harder question: "if the only image available over a
# site is hazy, is its embedding still the same place?" Both scenes come from the
# SAME summer window, so season is held fixed and haze is the only thing varying.
#
# CLEAR is capped well below the catalog's own MAX_CLOUD_COVER=15 so the reference
# side is genuinely clean; HAZY is floored at 25% so the contrast is real rather
# than two near-identical scenes a few percent apart.
# ---------------------------------------------------------------------------
ATMOS_CLEAR_MAX = 3.0
ATMOS_HAZY_MIN = 25.0
ATMOS_HAZY_MAX = 70.0

# Minimum patch-mean blue increase (scaled reflectance) for a pair to count as an
# actual haze test. Below this the two patches are atmospherically indistinguishable.
MIN_BLUE_DELTA = 150.0


# ---------------------------------------------------------------------------
# PATCH CACHE -- shared across models.
#
# This script re-downloads the same patches for every model. Running all five costs
# 5 x 20 = 100 COG reads where 20 would do, and on 4 Sep that wasted bandwidth was
# directly slowing the concurrent 13_analog_risk_features.py Clay run (which is
# network-bound, not compute-bound). The scene choice is deterministic -- same window,
# same cloud filter, min/max by cloud cover -- so the same location always resolves to
# the same scene, and a patch cached by the first model is valid for every later one.
#
# Keyed by mode + side + coordinate. Deleting the directory is always safe.
# ---------------------------------------------------------------------------
def _patch_cache_path(mode, side, lon, lat):
    from config import RESULTS_DIR
    d = os.path.join(RESULTS_DIR, "_stability_patch_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{mode}_{side}_{lon:.4f}_{lat:.4f}.npy")


def cached_extract(item, lon, lat, mode, side):
    """extract_patch(), memoised on disk across model runs."""
    p = _patch_cache_path(mode, side, lon, lat)
    if os.path.exists(p):
        try:
            return np.load(p)
        except Exception:
            pass          # corrupt/partial write -- just re-fetch
    patch = acq.extract_patch(item, lon, lat, PATCH_SIZE_M, PATCH_SIZE_PX, PRITHVI_BANDS)
    try:
        np.save(p, patch)
    except Exception:
        pass              # caching is an optimisation, never a correctness requirement
    return patch


def patch_haze_proxy(patch):
    """Mean blue reflectance of the 2240 m patch itself, as a haze proxy.

    WHY THIS IS NECESSARY. `eo:cloud_cover` is a **scene-level** property -- it describes
    a ~110 km tile, while the patch is 2.24 km. A scene can be 68 % cloudy and the 2.24 km
    window still be perfectly clear, in which case a "clear vs hazy" comparison is
    silently comparing two clear patches and the resulting high cosine means nothing.
    The first Serengeti run scored 0.9953 against a 68 %-cloud scene, which is exactly
    what that artifact looks like.

    Blue (B02) is the band haze and thin cloud lift most -- Rayleigh and aerosol
    scattering are strongest at short wavelengths -- so its patch mean separates a
    genuinely hazy window from a clear one. Reporting it lets each pair be checked:
    if the two sides have near-identical blue, the pair did not actually test haze and
    must be excluded rather than averaged in.

    PRITHVI_BANDS order is (B02, B03, B04, B8A, B11, B12), so index 0 is blue.
    """
    return float(np.mean(np.asarray(patch, dtype=np.float64)[0]))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Temporal Stability Analysis")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=list(SUPPORTED_MODELS.keys()))
    parser.add_argument("--mode", choices=["seasonal", "atmospheric", "interannual"],
                        default="seasonal",
                        help="seasonal: July vs January. atmospheric: clear vs hazy, both "
                             "from the SAME July window. interannual: July 2023 vs July "
                             "2021, same season different year (C10 acquisition date).")
    args = parser.parse_args()
    model_key = args.model
    atmos = args.mode == "atmospheric"
    inter = args.mode == "interannual"

    print(f"\n{'='*70}")
    if atmos:
        print(f"EcoLens ATMOSPHERIC Stability Analysis ({SUPPORTED_MODELS[model_key]['label']})")
        print(f"{'='*70}")
        print(f"Same summer window for both sides; only haze differs "
              f"(clear <{ATMOS_CLEAR_MAX:.0f}% vs hazy {ATMOS_HAZY_MIN:.0f}-{ATMOS_HAZY_MAX:.0f}%).\n")
    else:
        print(f"EcoLens Temporal Stability Analysis ({SUPPORTED_MODELS[model_key]['label']})")
        print(f"{'='*70}")
        print("Testing if embeddings remain consistent across Winter and Summer seasons.\n")

    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)
    
    # Clay and Satlas have no `timm_name`, so the old else-branch passed None into
    # timm.create_model and died with "'NoneType' object has no attribute 'startswith'".
    # Same defect as 10._get_drift_model. Each family needs its own loader.
    if model_key == "prithvi":
        model = ext.load_prithvi_model()
    elif model_key == "clay":
        model = ext.load_clay_model()
    elif model_key == "satlas":
        model = ext.load_satlas_model()
    else:
        model = ext.load_timm_model(SUPPORTED_MODELS[model_key]["timm_name"])
    
    # Catalog normalisation stats -- needed by 13.embed_cell for the Prithvi path.
    means, stds, _stats_meta = a13.load_norm_stats()

    # We will test on a subset of locations to save time (1 from each ecosystem)
    tested_ecosystems = set()
    test_locations = []
    for loc in PATCH_LOCATIONS:
        if loc["ecosystem"] not in tested_ecosystems:
            test_locations.append(loc)
            tested_ecosystems.add(loc["ecosystem"])
            
    print(f"Testing temporal stability on {len(test_locations)} diverse locations...")
    
    results = []
    
    for loc in test_locations:
        print(f"\nLocation: {loc['name']} ({loc['ecosystem']})")
        
        # 1. Fetch the reference scene -- clearest summer scene in both modes, but
        #    atmospheric mode caps it much tighter so the "clear" side is truly clean.
        item_summer = search_scene(catalog, loc["lon"], loc["lat"], SEASONS["summer"],
                                   cloud_lt=ATMOS_CLEAR_MAX) if atmos else \
                      search_seasonal_scene(catalog, loc["lon"], loc["lat"], SEASONS["summer"])
        if not item_summer:
            print(f"  [Skip] No {'clear' if atmos else 'clear summer'} scene found.")
            continue

        # 2. Fetch the second scene: hazy same-window (atmospheric), the same July two
        #    years earlier (interannual), or January (seasonal).
        #
        # Written as an explicit if/elif rather than a chained ternary. The chained form
        # is what produced the 5 Sep interannual bug: an edit updated the printed LABELS
        # to "July 2021" but failed to update this expression, so the run reported an
        # acquisition-date test while actually re-running the January comparison. The
        # numbers came out byte-identical to the seasonal run across all five models,
        # which is the only reason it was caught.
        if atmos:
            item_winter = search_scene(catalog, loc["lon"], loc["lat"], SEASONS["summer"],
                                       cloud_lt=ATMOS_HAZY_MAX, cloud_gte=ATMOS_HAZY_MIN,
                                       pick="cloudiest")
        elif inter:
            item_winter = search_seasonal_scene(catalog, loc["lon"], loc["lat"],
                                                SEASONS["summer_prev"])
        else:
            item_winter = search_seasonal_scene(catalog, loc["lon"], loc["lat"],
                                                SEASONS["winter"])
        if not item_winter:
            print(f"  [Skip] No "
                  f"{'hazy' if atmos else 'clear July-2021' if inter else 'clear winter'}"
                  f" scene found.")
            continue

        # ASSERT THE SCENE MATCHES THE WINDOW WE CLAIM.
        # The label and the query drifted apart once (see above); a printed label is not
        # evidence of what was fetched. Check the item's own datetime against the window.
        want = (SEASONS["summer"] if atmos
                else SEASONS["summer_prev"] if inter
                else SEASONS["winter"])
        got = str(item_winter.properties.get("datetime", ""))[:10]
        lo, hi = want.split("/")
        if not (lo <= got <= hi):
            raise AssertionError(
                f"scene/window mismatch for {loc['name']}: mode={args.mode} expected a "
                f"scene in {want} but got {got}. The comparison would be mislabelled.")

        a_lbl, b_lbl = (("Clear", "Hazy") if atmos
                        else ("July 2023", "July 2021") if inter
                        else ("Summer", "Winter"))
        print(f"  {a_lbl} Cloud Cover: {item_summer.properties['eo:cloud_cover']:.1f}%")
        print(f"  {b_lbl} Cloud Cover: {item_winter.properties['eo:cloud_cover']:.1f}%")

        try:
            # 3. Extract Patches
            patch_summer = cached_extract(item_summer, loc["lon"], loc["lat"], args.mode, "a")
            patch_winter = cached_extract(item_winter, loc["lon"], loc["lat"], args.mode, "b")
            
            # 4. Extract Embeddings
            # Clay and Satlas were unreachable here: the else-branch assumed a timm
            # model, so --model clay/satlas crashed the same way 10._get_drift_model
            # did. Each model needs its own input preparation.
            #
            # NOTE ON NORMALIZATION: raw patches are used for both seasons. That is
            # correct HERE because this metric compares a location to ITSELF across
            # two dates -- both sides live in the same space, so the cosine is
            # meaningful. It would NOT be valid to compare these vectors against the
            # catalog (see 13_analog_risk_features.py's docstring for what that costs).
            def _embed(patch, item):
                # Delegate to 13.embed_cell -- ONE definition of "the catalog's
                # preprocessing path", shared by both scripts.
                #
                # This used to be a private copy, and it was WRONG FOR PRITHVI: it fed the
                # RAW patch, while 03 builds Prithvi's catalog vectors from the 02
                # Z-SCORED patch. Same-place-two-dates cosines were still internally
                # consistent, so the bug was invisible in 12's own output -- but the moment
                # 21_stability_normalized.py compared them against the catalog's background
                # distribution, Prithvi scored 2.5 sigma BELOW background and looked
                # catastrophically unstable. That was a preprocessing mismatch, not a
                # property of the model. Every other model already matched, which is
                # exactly why Prithvi alone was the outlier.
                date = str(item.properties.get("datetime", ""))[:10] or None
                return a13.embed_cell(model, model_key, patch,
                                      loc["lon"], loc["lat"], date, means, stds)

            emb_summer = _embed(patch_summer, item_summer)
            emb_winter = _embed(patch_winter, item_winter)
            emb_summer = np.asarray(emb_summer, dtype=np.float32)
            emb_winter = np.asarray(emb_winter, dtype=np.float32)
                
            # Normalize embeddings to match downstream FAISS logic
            emb_summer /= np.linalg.norm(emb_summer) + 1e-8
            emb_winter /= np.linalg.norm(emb_winter) + 1e-8
                
            # 5. Compute Similarity
            sim = 1.0 - cosine(emb_summer, emb_winter)
            axis_lbl = ("Clear-vs-Hazy" if atmos
                        else "Across-Years" if inter else "Temporal")
            print(f"  --> {axis_lbl} Cosine Similarity: {sim:.4f}")

            results.append({
                "id": loc["id"],
                "name": loc["name"],
                "ecosystem": loc["ecosystem"],
                "similarity": float(sim),
                "summer_cloud_pct": float(item_summer.properties["eo:cloud_cover"]),
                "winter_cloud_pct": float(item_winter.properties["eo:cloud_cover"]),
                # patch-level haze, not scene-level -- see patch_haze_proxy()
                "summer_patch_blue": patch_haze_proxy(patch_summer),
                "winter_patch_blue": patch_haze_proxy(patch_winter),
            })
            if atmos:
                d_blue = (results[-1]["winter_patch_blue"]
                          - results[-1]["summer_patch_blue"])
                print(f"      patch blue: clear {results[-1]['summer_patch_blue']:.0f} "
                      f"-> hazy {results[-1]['winter_patch_blue']:.0f}  "
                      f"(delta {d_blue:+.0f})"
                      + ("   [!] no real haze contrast in the patch"
                         if d_blue < MIN_BLUE_DELTA else ""))
        except Exception as e:
            print(f"  [Error] Failed to process patches: {e}")

    print(f"\n{'='*70}")
    print(f"{'ATMOSPHERIC' if atmos else 'ACQUISITION-DATE' if inter else 'TEMPORAL'} "
          f"STABILITY SUMMARY")
    print(f"{'='*70}")
    if not results:
        print("No valid pairs found for testing.")
        return

    avg_sim = np.mean([r["similarity"] for r in results])
    print(f"Overall Average "
          f"{'Clear-vs-Hazy' if atmos else 'Across-Years' if inter else 'Seasonal'} "
          f"Similarity: {avg_sim:.4f}")

    for eco in sorted(set([r["ecosystem"] for r in results])):
        eco_sims = [r["similarity"] for r in results if r["ecosystem"] == eco]
        print(f"  {eco:<15}: {np.mean(eco_sims):.4f}")

    if atmos:
        # Split on MEASURED patch haze, not on the scene-level cloud percentage. A pair
        # whose two patches have the same blue reflectance never tested haze at all, and
        # averaging it in would inflate the apparent robustness.
        real = [r for r in results
                if r["winter_patch_blue"] - r["summer_patch_blue"] >= MIN_BLUE_DELTA]
        fake = [r for r in results if r not in real]
        print(f"\n  Pairs with genuine patch-level haze contrast "
              f"(blue delta >= {MIN_BLUE_DELTA}): {len(real)}/{len(results)}")
        if real:
            print(f"    mean similarity on those: {np.mean([r['similarity'] for r in real]):.4f}")
            for r in sorted(real, key=lambda r: r["similarity"]):
                print(f"      {r['name'][:38]:<38} {r['similarity']:.4f}  "
                      f"(blue {r['summer_patch_blue']:.0f} -> {r['winter_patch_blue']:.0f})")
        if fake:
            print(f"    EXCLUDED -- the hazy scene's patch was not actually hazy: "
                  f"{', '.join(r['name'][:28] for r in fake)}")

    # PERSIST THE RESULT. This script has existed since before 3 Sep and had never
    # been run -- partly because it printed to stdout and saved nothing, so there was
    # never an artifact for anything to report (implementation.md C10: "12 exists but
    # has never been run and its output is not reported anywhere"). One shared file,
    # keyed by model, so all five runs accumulate instead of overwriting.
    from config import RESULTS_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = (f"{RESULTS_DIR}/atmospheric_stability.json" if atmos
                else f"{RESULTS_DIR}/interannual_stability.json" if inter
                else f"{RESULTS_DIR}/temporal_stability.json")
    payload = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {}
    payload[model_key] = {
        "label": SUPPORTED_MODELS[model_key]["label"],
        "mean_seasonal_cosine": float(avg_sim),
        "n_locations": len(results),
        "mode": args.mode,
        "summer_window": SEASONS["summer"],
        "winter_window": (SEASONS["summer"] if atmos
                          else SEASONS["summer_prev"] if inter
                          else SEASONS["winter"]),
        "cloud_bands": ({"clear_lt": ATMOS_CLEAR_MAX,
                         "hazy_gte": ATMOS_HAZY_MIN, "hazy_lt": ATMOS_HAZY_MAX}
                        if atmos else {"both_lt": MAX_CLOUD_COVER}),
        "per_ecosystem": {
            eco: float(np.mean([r["similarity"] for r in results if r["ecosystem"] == eco]))
            for eco in sorted({r["ecosystem"] for r in results})
        },
        "per_location": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_path}  (models recorded: {', '.join(sorted(payload))})")


if __name__ == "__main__":
    main()
