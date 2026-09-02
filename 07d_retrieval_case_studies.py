"""
EcoLens Objective 2 - Step 7d: Retrieval Case Studies (Successes, Partial
Successes, and Failures)

---------------------------------------------------------------------
WHY THIS SCRIPT EXISTS
---------------------------------------------------------------------
Faculty review comments:
  "Provide visual examples illustrating successful retrievals, partially
   successful retrievals, and retrieval failures, together with
   ecological interpretations."
  "Investigate situations where visually similar ecosystems are
   ecologically different and discuss possible reasons for these
   retrieval failures."

Up to this point the dashboard let a user click around and eyeball
individual results, but nothing in the project systematically flagged
which retrievals actually succeed, partially succeed, or fail --
against a real ecological yardstick, not just "same category label".

---------------------------------------------------------------------
WHAT "SUCCESS" AND "FAILURE" MEAN HERE
---------------------------------------------------------------------
For every query's top-1 cosine analog (grouped -- i.e. a genuinely
different base location, not a near-duplicate sub-crop of the same
place; see 07_evaluate_retrieval.py's leakage discussion), this script
cross-references TWO independent signals that are already computed
elsewhere in the pipeline and never combined until now:

  1. Embedding similarity (cosine score from 06_retrieval_engine.py)
     -- "does the model THINK these look alike?"
  2. Ecological descriptor agreement (from
     results/ecosystem_descriptors.json, via 09_explainability_engine.py
     -- forest/water/urban cover, veg health, temp/rainfall/elevation
     where available, protected status) -- "ARE these actually alike,
     by real spectral/physical measurements?"

Categories:
  - SUCCESS: same ecosystem category AND low descriptor disagreement
    (top ecological-agreement tercile).
  - FAILURE ("visually similar, ecologically different"): high
    embedding similarity but high descriptor disagreement (bottom
    ecological-agreement tercile) -- this is exactly the case the
    faculty comment asks to be investigated, regardless of whether the
    category label matches.
  - PARTIAL: everything else (same category but middling descriptor
    agreement, or different category but still descriptor-similar).

Run:
    python 07d_retrieval_case_studies.py [--model prithvi] [--top-n 5]
"""

import argparse
import json
import os

import numpy as np

from config import RESULTS_DIR, METADATA_CATALOG_PATH, SUPPORTED_MODELS, DEFAULT_MODEL

OUT_PATH = f"{RESULTS_DIR}/retrieval_case_studies.json"

DESCRIPTOR_FIELDS = ["forest_cover", "water_cover", "urban_cover", "veg_health"]


def descriptor_disagreement(q_desc, a_desc):
    """
    A single scalar "how ecologically different are these two patches"
    score: mean absolute difference across the always-available spectral
    descriptors, each min-max-ish scaled by a fixed reasonable range so
    no single field (e.g. forest_cover, 0-100) dominates over another
    (e.g. veg_health, roughly 0-1).
    """
    ranges = {"forest_cover": 100.0, "water_cover": 100.0, "urban_cover": 100.0, "veg_health": 1.0}
    diffs = []
    for field in DESCRIPTOR_FIELDS:
        qv, av = q_desc.get(field), a_desc.get(field)
        if qv is None or av is None:
            continue
        diffs.append(abs(qv - av) / ranges[field])
    return float(np.mean(diffs)) if diffs else None


def build_case_studies(retrieval_results, descriptors, catalog_lookup, top_n):
    """
    retrieval_results: the 'cosine' block of retrieval_results_<model>.json
    (query_id -> ranked list of candidates, ALL candidates, so we can find
    the first genuinely different-location hit ourselves).
    """
    records = []
    for qid, ranked in retrieval_results.items():
        q_desc = descriptors.get(qid)
        if q_desc is None:
            continue
        q_base = catalog_lookup.get(qid, {}).get("base_id", qid.split("_p")[0])
        q_eco = catalog_lookup.get(qid, {}).get("ecosystem")

        # First candidate from a genuinely different base location --
        # same grouped/leave-one-location-out logic as 07_evaluate_retrieval.py,
        # so this reflects the honest (de-leaked) top-1, not a near-duplicate
        # sub-crop of the query's own location.
        top1 = None
        for cand in ranked:
            cand_base = catalog_lookup.get(cand["id"], {}).get("base_id", cand["id"].split("_p")[0])
            if cand_base != q_base:
                top1 = cand
                break
        if top1 is None:
            continue

        a_desc = descriptors.get(top1["id"])
        if a_desc is None:
            continue

        disagreement = descriptor_disagreement(q_desc, a_desc)
        if disagreement is None:
            continue

        records.append({
            "query_id": qid,
            "query_ecosystem": q_eco,
            "query_name": catalog_lookup.get(qid, {}).get("name", ""),
            "analog_id": top1["id"],
            "analog_ecosystem": top1["ecosystem"],
            "analog_name": top1.get("name", ""),
            "cosine_score": top1["score"],
            "same_category": q_eco == top1["ecosystem"],
            "descriptor_disagreement": disagreement,
        })

    if not records:
        return {"success": [], "partial": [], "failure": []}

    disagreements = np.array([r["descriptor_disagreement"] for r in records])
    low_thresh = np.percentile(disagreements, 33)
    high_thresh = np.percentile(disagreements, 67)

    success, partial, failure = [], [], []
    for r in records:
        d = r["descriptor_disagreement"]
        if r["same_category"] and d <= low_thresh:
            success.append(r)
        elif d >= high_thresh:
            # High disagreement regardless of category label -- this is
            # the "visually/embedding-similar but ecologically different"
            # case the faculty comment asks to be investigated, whether
            # or not the category label happens to match.
            failure.append(r)
        else:
            partial.append(r)

    success.sort(key=lambda r: r["cosine_score"], reverse=True)
    failure.sort(key=lambda r: r["cosine_score"], reverse=True)
    partial.sort(key=lambda r: r["cosine_score"], reverse=True)

    return {
        "thresholds": {"low_disagreement": float(low_thresh), "high_disagreement": float(high_thresh)},
        "success": success[:top_n],
        "partial": partial[:top_n],
        "failure": failure[:top_n],
        "counts": {"success": len(success), "partial": len(partial), "failure": len(failure), "total": len(records)},
    }


def print_case(r, explanations):
    print(f"  Query:  {r['query_id']:<20} ({r['query_ecosystem']:<12}) {r['query_name']}")
    print(f"  Analog: {r['analog_id']:<20} ({r['analog_ecosystem']:<12}) {r['analog_name']}")
    print(f"  Cosine similarity: {r['cosine_score']:.4f}   Descriptor disagreement: {r['descriptor_disagreement']:.4f}   Same category: {r['same_category']}")
    expl = explanations.get(r["query_id"], {}).get(r["analog_id"])
    if expl:
        print(f"  Explanation: {expl['explanation']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="EcoLens Step 7d: Retrieval case studies")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=list(SUPPORTED_MODELS.keys()))
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    desc_path = f"{RESULTS_DIR}/ecosystem_descriptors.json"
    explain_path = f"{RESULTS_DIR}/explainable_retrieval.json"
    retrieval_path = f"{RESULTS_DIR}/retrieval_results_{args.model}.json"

    if not os.path.exists(desc_path):
        print(f"Error: {desc_path} not found. Run 09_explainability_engine.py first.")
        return
    if not os.path.exists(retrieval_path):
        print(f"Error: {retrieval_path} not found. Run 06_retrieval_engine.py --model {args.model} first.")
        return

    with open(desc_path, encoding="utf-8") as f:
        descriptors = json.load(f)
    with open(retrieval_path, encoding="utf-8") as f:
        retrieval_results = json.load(f)
    with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    catalog_lookup = {e["id"]: e for e in catalog}

    explanations = {}
    if os.path.exists(explain_path):
        with open(explain_path, encoding="utf-8") as f:
            explanations = json.load(f).get("explanations", {})

    label = SUPPORTED_MODELS.get(args.model, {}).get("label", args.model)
    print(f"\n{'='*70}")
    print(f"EcoLens Step 7d: Retrieval Case Studies -- {label}")
    print(f"{'='*70}")
    print("Cross-references embedding similarity against REAL ecological")
    print("descriptor agreement (not just category label) to find genuine")
    print("successes, partial successes, and failures.\n")

    cases = build_case_studies(retrieval_results.get("cosine", {}), descriptors, catalog_lookup, args.top_n)

    counts = cases["counts"]
    print(f"Top-1 (grouped) analogs classified: {counts['total']} total")
    print(f"  SUCCESS (same category, low descriptor disagreement): {counts['success']}")
    print(f"  PARTIAL (mixed signal):                                {counts['partial']}")
    print(f"  FAILURE (high descriptor disagreement, any category):  {counts['failure']}")

    print(f"\n{'-'*70}")
    print(f"SUCCESSFUL RETRIEVALS (top {args.top_n})")
    print(f"{'-'*70}")
    for r in cases["success"]:
        print_case(r, explanations)

    print(f"{'-'*70}")
    print(f"PARTIALLY SUCCESSFUL RETRIEVALS (top {args.top_n})")
    print(f"{'-'*70}")
    for r in cases["partial"]:
        print_case(r, explanations)

    print(f"{'-'*70}")
    print(f"RETRIEVAL FAILURES -- embedding-similar but ecologically different (top {args.top_n})")
    print(f"{'-'*70}")
    for r in cases["failure"]:
        print_case(r, explanations)
        same_cat = "same category label" if r["same_category"] else "DIFFERENT category label"
        print(f"  -> Why this likely failed: high embedding similarity ({r['cosine_score']:.3f}) despite "
              f"{same_cat} and descriptor disagreement of {r['descriptor_disagreement']:.3f}. "
              f"The model is picking up shared low-level visual texture/color (e.g. canopy density, "
              f"water sheen, soil tone) that doesn't track the real spectral/physical differences "
              f"09_explainability_engine.py measured -- exactly the risk of judging ecological "
              f"similarity from imagery alone.\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({args.model: cases}, f, indent=2)
    print(f"Saved case studies to: {OUT_PATH}")


if __name__ == "__main__":
    main()
