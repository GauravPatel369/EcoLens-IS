"""
EcoLens 07b -- do retrieved analogs share real ecological characteristics?
(faculty comment C11)

GROUP-AWARE. Read this before changing the exclusion rule.
--------------------------------------------------------------------
02_preprocess_patches.py expands each base location into 10 overlapping
sub-crops. They share a coordinate, so they share their climate cell,
their DEM cell, their ecoregion polygon, their curated protection flag
and their Hansen loss-year vector -- every quantity this script measures.

The original version of this script excluded only the query patch
itself (`aid == qid`). Measured 3 Sep: **100 % of the top-5 analog pairs
it scored were sub-crops of the query's own base location** (3850 of
3850). It was therefore reporting Temperature MAE 0.01 C, Forest Cover
MAE 0.00 % and 100 % protection agreement -- comparing each place to
itself. That is the LEAKED evaluation mode that implementation.md 9
forbids quoting, arrived at silently.

This version excludes the whole base location, matching
07_evaluate_retrieval.py's GROUPED protocol. The numbers get much worse
and much more meaningful.

RANDOM-ANALOG CONTROL
--------------------------------------------------------------------
An MAE in isolation cannot be judged: "rainfall MAE 400 mm" is only
good or bad relative to what you would get by picking analogs at
random from the same pool. Every metric is therefore reported twice,
retrieved vs random, over the same candidate pool and the same
exclusion rule. The gap between them is the actual C11 result.

Run:
    python 07b_evaluate_ecological_similarity.py
"""

import json
import os

import numpy as np

from config import RESULTS_DIR, SUPPORTED_MODELS, METADATA_CATALOG_PATH

TOP_K = 5
RANDOM_SEED = 0


def base_of(patch_id, base_map):
    """Base location for a sub-crop id, preferring the catalog's explicit
    base_id over re-deriving it from the string (an ecosystem name could
    itself contain '_p')."""
    if patch_id in base_map:
        return base_map[patch_id]
    return patch_id.rsplit("_p", 1)[0]


def pair_metrics(q_desc, a_desc, acc):
    """Accumulate one query/analog comparison into acc."""
    for key, bucket in (("temp_c", "temp"), ("rainfall_mm", "rain"),
                        ("elevation_m", "elev"), ("forest_cover", "forest")):
        qv, av = q_desc.get(key), a_desc.get(key)
        if qv is not None and av is not None:
            acc[bucket].append(abs(qv - av))

    qp, ap = q_desc.get("protected_area"), a_desc.get("protected_area")
    if qp is not None and ap is not None:
        acc["protected"].append(1.0 if qp == ap else 0.0)

    ql, al = q_desc.get("loss_years"), a_desc.get("loss_years")
    if ql is not None and al is not None:
        q_years, a_years = set(ql.keys()), set(al.keys())
        union = q_years | a_years
        acc["jaccard"].append(len(q_years & a_years) / len(union) if union else 1.0)


def new_acc():
    return {k: [] for k in ("temp", "rain", "elev", "forest", "protected", "jaccard")}


def summarise(acc):
    def m(key):
        return float(np.mean(acc[key])) if acc[key] else None
    return {
        "temp_mae_c": m("temp"), "temp_n": len(acc["temp"]),
        "rainfall_mae_mm": m("rain"), "rainfall_n": len(acc["rain"]),
        "elevation_mae_m": m("elev"), "elevation_n": len(acc["elev"]),
        "forest_cover_mae_pct": m("forest"), "forest_cover_n": len(acc["forest"]),
        "protection_agreement": m("protected"), "protection_n": len(acc["protected"]),
        "trajectory_jaccard": m("jaccard"), "trajectory_n": len(acc["jaccard"]),
    }


def fmt(v, unit="", nd=2):
    return "N/A" if v is None else f"{v:.{nd}f}{unit}"


def evaluate_ecological_similarity():
    print(f"\n{'='*78}")
    print("EcoLens 07b: Ecological Characteristic Similarity  [GROUP-AWARE]")
    print(f"{'='*78}")

    desc_path = f"{RESULTS_DIR}/ecosystem_descriptors.json"
    if not os.path.exists(desc_path):
        print(f"Error: Descriptors not found at {desc_path}. Run 09_explainability_engine.py first.")
        return
    with open(desc_path, encoding="utf-8") as f:
        descriptors = json.load(f)

    base_map = {}
    if os.path.exists(METADATA_CATALOG_PATH):
        with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
            for e in json.load(f):
                base_map[e["id"]] = e.get("base_id", e["id"].rsplit("_p", 1)[0])

    retrieval_files = [
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("retrieval_results_") and f.endswith(".json")
    ]
    if not retrieval_files:
        print(f"No retrieval result files found in {RESULTS_DIR}.")
        return

    print(f"Top-{TOP_K} analogs per query, EXCLUDING every sub-crop of the query's own")
    print("base location. Each metric is shown against a random-analog control drawn")
    print("from the same pool under the same exclusion, so the gap is interpretable.\n")

    report = {}
    for rfile in sorted(retrieval_files):
        model_key = rfile.replace("retrieval_results_", "").replace(".json", "")
        label = SUPPORTED_MODELS.get(model_key, {}).get("label", model_key)

        with open(os.path.join(RESULTS_DIR, rfile), encoding="utf-8") as f:
            payload = json.load(f)
        # The embedding models write a "cosine" block; 07c's spectral baseline ranks by
        # EUCLIDEAN distance over 4 handcrafted indices and writes only "euclidean".
        # Hardcoding "cosine" silently produced `queries=0` and a table of N/A for the
        # baseline -- it looked like the baseline had no ecological agreement to measure,
        # when in fact nothing had been read. Prefer cosine, else take whatever method
        # the file actually contains.
        method = "cosine" if "cosine" in payload else next(iter(payload), None)
        if method is None:
            print(f"[{label}] no ranked results in {rfile}")
            continue
        if method != "cosine":
            print(f"[{label}] no cosine block; using '{method}'")
        cosine_res = payload.get(method, {})

        all_ids = [q for q in cosine_res if q in descriptors]
        rng = np.random.default_rng(RANDOM_SEED)

        retrieved, random_ctl = new_acc(), new_acc()
        kept, dropped_all_same = 0, 0

        for qid in all_ids:
            q_desc = descriptors[qid]
            q_base = base_of(qid, base_map)

            picked = []
            for analog in cosine_res[qid]:
                aid = analog["id"]
                if base_of(aid, base_map) == q_base:
                    continue          # <-- the fix: whole location, not just the patch
                if aid not in descriptors:
                    continue
                picked.append(aid)
                if len(picked) == TOP_K:
                    break
            if not picked:
                dropped_all_same += 1
                continue
            kept += 1
            for aid in picked:
                pair_metrics(q_desc, descriptors[aid], retrieved)

            pool = [i for i in all_ids if base_of(i, base_map) != q_base]
            if pool:
                for aid in rng.choice(pool, size=min(TOP_K, len(pool)), replace=False):
                    pair_metrics(q_desc, descriptors[str(aid)], random_ctl)

        r, c = summarise(retrieved), summarise(random_ctl)
        report[model_key] = {"label": label, "queries": kept,
                             "retrieved": r, "random_control": c}

        print(f"[{label}]  queries={kept}"
              + (f"  (skipped {dropped_all_same} with no cross-location analog)" if dropped_all_same else ""))
        print(f"  {'metric':<34}{'retrieved':>14}{'random':>14}   verdict")
        rows = [
            ("Temperature MAE (lower better)", "temp_mae_c", " C", 2, False),
            ("Rainfall MAE (lower better)", "rainfall_mae_mm", " mm", 1, False),
            ("Elevation MAE (lower better)", "elevation_mae_m", " m", 1, False),
            ("Forest-cover MAE (lower better)", "forest_cover_mae_pct", " %", 2, False),
            ("Protection agreement (higher better)", "protection_agreement", "", 3, True),
            ("Disturbance Jaccard (higher better)", "trajectory_jaccard", "", 3, True),
        ]
        for name, key, unit, nd, higher_better in rows:
            rv, cv = r[key], c[key]
            if rv is None or cv is None:
                verdict = "n/a"
            elif higher_better:
                verdict = "better than random" if rv > cv else ("no better" if rv < cv else "tied")
            else:
                verdict = "better than random" if rv < cv else ("no better" if rv > cv else "tied")
            print(f"  {name:<34}{fmt(rv, unit, nd):>14}{fmt(cv, unit, nd):>14}   {verdict}")
        print(f"  (n: climate {r['temp_n']}, elevation {r['elevation_n']}, "
              f"trajectory {r['trajectory_n']} -- elevation is forest-only, "
              f"trajectory is forest/mangrove-only)")
        print()

    out = f"{RESULTS_DIR}/ecological_similarity.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved to {out}")
    print("\nNOTE: protection status comes from the curated config.py flag, not WDPA")
    print("(09 reports protected_source='catalog (WDPA unavailable)'), so protection")
    print("agreement partly measures that curation rather than independent data.")


if __name__ == "__main__":
    evaluate_ecological_similarity()
