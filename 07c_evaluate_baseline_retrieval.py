"""
EcoLens Objective 2 - Step 7c: Handcrafted-Feature Retrieval Baseline

---------------------------------------------------------------------
WHY THIS SCRIPT EXISTS
---------------------------------------------------------------------
Faculty review comment: "Explain why geospatial foundation models are
necessary. Demonstrate how semantic representations capture ecosystem
characteristics that conventional spectral indices or handcrafted
features cannot."

Up to this point the project asserted that foundation-model embeddings
are useful for retrieval, but never measured that against the obvious
alternative: rank by a handcrafted feature vector built from the same
spectral indices 09_explainability_engine.py already computes (NDVI/
NDWI/NDBI-derived forest/water/urban cover + vegetation health). This
script builds exactly that baseline, evaluates it with the SAME
leave-one-location-out (grouped) methodology used for every embedding
model in 07_evaluate_retrieval.py, and reports it side by side.

---------------------------------------------------------------------
WHAT THE BASELINE IS
---------------------------------------------------------------------
For each patch, a 4-dimensional feature vector:
    [forest_cover_pct, water_cover_pct, urban_cover_pct, veg_health]
(from results/ecosystem_descriptors.json, produced by
09_explainability_engine.py -- run that first). Vectors are z-score
standardized across the dataset, then ranked by Euclidean distance --
no learned representation, no pretraining, just the four numbers a
domain expert would compute by hand from Sentinel-2 bands.

---------------------------------------------------------------------
OUTPUT
---------------------------------------------------------------------
Writes results/retrieval_results_spectral_baseline.json in the exact
schema 06_retrieval_engine.py writes for every other model. This means
the NEXT time you run 07_evaluate_retrieval.py, the spectral baseline
shows up automatically in the cross-model comparison table -- no
changes needed there.

Run:
    python 07c_evaluate_baseline_retrieval.py
"""

import importlib.util
import json
import os

import numpy as np

from config import RESULTS_DIR, METADATA_CATALOG_PATH, SUPPORTED_MODELS

DESCRIPTORS_PATH = f"{RESULTS_DIR}/ecosystem_descriptors.json"
OUT_PATH = f"{RESULTS_DIR}/retrieval_results_spectral_baseline.json"
FEATURE_KEYS = ["forest_cover", "water_cover", "urban_cover", "veg_health"]


def _load_module(numbered_filename, module_name):
    """Numbered script filenames (e.g. '07_evaluate_retrieval.py') can't be
    imported with a normal `import 07_evaluate_retrieval` statement --
    load them by path instead. Same pattern already used by
    11_forest_risk_forecast.py::predict_risk() to reuse 10's helpers."""
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(os.path.dirname(__file__), numbered_filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_spectral_ranking(descriptors, catalog_lookup):
    """
    Rank every patch against every other patch by Euclidean distance in
    standardized [forest_cover, water_cover, urban_cover, veg_health]
    space. Returns a dict keyed "euclidean" (matching the naming
    06_retrieval_engine.py uses for its distance-based method) mapping
    query_id -> ranked list of candidate dicts, same shape as
    06_retrieval_engine.py's search_all() output.
    """
    ids = [pid for pid in descriptors if pid in catalog_lookup]
    matrix = np.array([[descriptors[pid][k] for k in FEATURE_KEYS] for pid in ids], dtype=np.float64)

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    standardized = (matrix - mean) / std

    n = len(ids)
    results = {"euclidean": {}}

    for i in range(n):
        diffs = standardized - standardized[i]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        order = np.argsort(dists)

        ranked = []
        rank = 0
        for j in order:
            if j == i:
                continue
            entry = catalog_lookup[ids[j]]
            rank += 1
            ranked.append({
                "id": ids[j],
                "ecosystem": entry["ecosystem"],
                "name": entry.get("name", ""),
                "lon": entry.get("lon"),
                "lat": entry.get("lat"),
                "protected_area": entry.get("protected_area", False),
                "climatic_region": entry.get("climatic_region", "Unknown"),
                "score": float(1.0 / (1.0 + dists[j])),
                "rank": rank,
            })
        results["euclidean"][ids[i]] = ranked

    return results


def main():
    print(f"\n{'='*70}")
    print("EcoLens Step 7c: Handcrafted-Feature (Spectral Index) Retrieval Baseline")
    print(f"{'='*70}")
    print("Answers: do foundation-model embeddings retrieve better than ranking")
    print("by NDVI/NDWI/NDBI-derived forest/water/urban/veg-health alone?\n")

    if not os.path.exists(DESCRIPTORS_PATH):
        print(f"Error: {DESCRIPTORS_PATH} not found. Run 09_explainability_engine.py first.")
        return
    if not os.path.exists(METADATA_CATALOG_PATH):
        print(f"Error: {METADATA_CATALOG_PATH} not found. Run 01-04 first.")
        return

    with open(DESCRIPTORS_PATH, encoding="utf-8") as f:
        descriptors = json.load(f)
    with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    catalog_lookup = {e["id"]: e for e in catalog}

    print(f"Building spectral-feature ranking for {len(descriptors)} patches "
          f"using features: {FEATURE_KEYS}")
    baseline_results = build_spectral_ranking(descriptors, catalog_lookup)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)
    print(f"Saved: {OUT_PATH}")

    # ---------------------------------------------------------------
    # Evaluate with the exact same grouped methodology as every
    # embedding model, and print a direct comparison.
    # ---------------------------------------------------------------
    ev = _load_module("07_evaluate_retrieval.py", "ev07")

    full_catalog_lookup = ev.build_catalog_lookup({"euclidean": baseline_results["euclidean"]})
    grouped = ev.evaluate_method(
        baseline_results["euclidean"], full_catalog_lookup, "euclidean",
        ev.EVALUATION_K_VALUES, group_aware=True,
    )
    baseline_map = grouped["overall"]["mAP"]
    baseline_p1 = grouped["overall"]["P@1"]
    print(f"\nSpectral baseline (grouped, leave-one-location-out):")
    print(f"  mAP = {baseline_map:.4f}   P@1 = {baseline_p1:.4f}   "
          f"MRR = {grouped['overall']['MRR']:.4f}")

    print(f"\n{'='*70}")
    print("COMPARISON: embedding models vs. handcrafted spectral baseline")
    print(f"{'='*70}")
    print(f"{'Method':<22} {'mAP (grouped)':<16} {'vs. spectral baseline'}")
    print("-" * 70)
    print(f"{'Spectral baseline':<22} {baseline_map:<16.4f} {'--'}")

    any_model_found = False
    for model_key, model_cfg in SUPPORTED_MODELS.items():
        rpath = f"{RESULTS_DIR}/retrieval_results_{model_key}.json"
        if not os.path.exists(rpath):
            continue
        any_model_found = True
        with open(rpath, encoding="utf-8") as f:
            model_results = json.load(f)
        if "cosine" not in model_results:
            continue
        model_catalog_lookup = ev.build_catalog_lookup({"cosine": model_results["cosine"]})
        model_grouped = ev.evaluate_method(
            model_results["cosine"], model_catalog_lookup, "cosine",
            ev.EVALUATION_K_VALUES, group_aware=True,
        )
        model_map = model_grouped["overall"]["mAP"]
        delta = model_map - baseline_map
        pct = (delta / baseline_map * 100.0) if baseline_map > 0 else float("nan")
        verdict = "beats baseline" if delta > 0 else "below baseline"
        print(f"{model_cfg['label']:<22} {model_map:<16.4f} "
              f"{delta:+.4f} ({pct:+.1f}%, {verdict})")

    if not any_model_found:
        print("\n(No embedding model retrieval_results_<model>.json files found yet --")
        print(" run 06_retrieval_engine.py --model <model> for at least one model")
        print(" to see the comparison populated.)")

    print(f"\nDone. Re-run 07_evaluate_retrieval.py to see 'spectral_baseline' folded")
    print(f"into the standard cross-model comparison tables and evaluation_report.json.")


if __name__ == "__main__":
    main()
