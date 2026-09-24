"""
EcoLens Step 17 -- CROSS-REGION RETRIEVAL EVALUATION  (Phase 3; C7, feeds C14)

C7: "Is retrieval consistent across geographic regions, not just nearby ecosystems?"
07_evaluate_retrieval.py contains no notion of region at all, so this has never
been answered. This script adds the two protocols implementation.md Phase 3 asks
for, using biogeographic REALM derived from RESOLVE (16_derive_location_geography.py):

  A. STRATIFIED  -- ordinary grouped retrieval, mAP reported per realm. Answers
     "does the system work as well in the Afrotropics as the Palearctic?"

  B. LEAVE-ONE-REALM-OUT -- query from realm X with the candidate pool restricted
     to everywhere EXCEPT X. This is the retrieval-side mirror of
     11 --spatial-holdout and the honest test of "across geographically distinct
     landscapes": the model must find an ecological match on another continent,
     with no same-realm shortcut available.

  C. SAME- vs CROSS-REALM analog rate per model.

Every protocol keeps 07's GROUPED rule (a patch's own base location is never a
candidate); realm exclusion is applied ON TOP of that, never instead of it.

Writes results/cross_region_retrieval.json

Run:
    python 17_cross_region_retrieval.py
"""

import json
import os
from collections import defaultdict

import numpy as np

from config import METADATA_CATALOG_PATH, METADATA_DIR, RESULTS_DIR, SUPPORTED_MODELS

GEO_PATH = f"{METADATA_DIR}/location_geography.json"
OUT_PATH = f"{RESULTS_DIR}/cross_region_retrieval.json"
K = 5


def load_geo():
    if not os.path.exists(GEO_PATH):
        raise SystemExit(f"{GEO_PATH} not found -- run 16_derive_location_geography.py")
    with open(GEO_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_model(model_key):
    with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    key = "prithvi_embedding" if model_key == "prithvi" else f"{model_key}_embedding"
    ids, bases, ecos, vecs = [], [], [], []
    for e in catalog:
        p = e.get(key)
        if not p or not os.path.exists(p):
            continue
        v = np.load(p).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        ids.append(e["id"])
        bases.append(e.get("base_id", e["id"].rsplit("_p", 1)[0]))
        ecos.append(e["ecosystem"])
        vecs.append(v)
    return np.array(bases), np.array(ecos), np.stack(vecs)


def average_precision(hits):
    """AP over a ranked boolean list."""
    n_rel, score = 0, 0.0
    for i, h in enumerate(hits, 1):
        if h:
            n_rel += 1
            score += n_rel / i
    return score / n_rel if n_rel else 0.0


def evaluate(bases, ecos, X, realms, restrict_cross_realm):
    """Grouped retrieval. If restrict_cross_realm, candidates must be in a
    DIFFERENT realm from the query."""
    S = X @ X.T
    np.fill_diagonal(S, -2.0)
    per_realm = defaultdict(list)
    same_realm_hits, cross_realm_hits = 0, 0
    overall = []
    for i in range(len(X)):
        qr = realms[i]
        if qr is None:
            continue
        mask = bases != bases[i]                      # 07's GROUPED rule
        if restrict_cross_realm:
            mask &= np.array([r != qr for r in realms])
        if not mask.any():
            continue
        cand = np.where(mask)[0]
        order = cand[np.argsort(-S[i, cand])][:K]
        hits = [ecos[j] == ecos[i] for j in order]
        ap = average_precision(hits)
        per_realm[qr].append(ap)
        overall.append(ap)
        if not restrict_cross_realm:
            for j in order:
                if realms[j] == qr:
                    same_realm_hits += 1
                else:
                    cross_realm_hits += 1
    return {
        "overall_map_at_k": float(np.mean(overall)) if overall else None,
        "n_queries": len(overall),
        "per_realm": {r: {"map_at_k": float(np.mean(v)), "n": len(v)}
                      for r, v in sorted(per_realm.items())},
        "same_realm_analogs": same_realm_hits,
        "cross_realm_analogs": cross_realm_hits,
    }


def main():
    geo = load_geo()
    report = {}
    print(f"\n{'='*78}")
    print("EcoLens 17: cross-region retrieval (C7)")
    print(f"{'='*78}")
    print(f"Grouped (own base location excluded) + realm protocols, mAP@{K}\n")

    for mk in SUPPORTED_MODELS:
        try:
            bases, ecos, X = load_model(mk)
        except Exception as e:
            print(f"[{mk}] skipped ({type(e).__name__})")
            continue
        realms = [geo.get(b, {}).get("realm") for b in bases]
        n_norealm = sum(1 for r in realms if r is None)

        within = evaluate(bases, ecos, X, realms, restrict_cross_realm=False)
        cross = evaluate(bases, ecos, X, realms, restrict_cross_realm=True)
        report[mk] = {"label": SUPPORTED_MODELS[mk]["label"],
                      "stratified": within, "leave_one_realm_out": cross,
                      "patches_without_realm": n_norealm}

        lab = SUPPORTED_MODELS[mk]["label"]
        drop = cross["overall_map_at_k"] - within["overall_map_at_k"]
        print(f"[{lab}]  (patches with no realm: {n_norealm})")
        print(f"  normal grouped            mAP@{K} = {within['overall_map_at_k']:.4f}")
        print(f"  LEAVE-ONE-REALM-OUT       mAP@{K} = {cross['overall_map_at_k']:.4f}"
              f"   ({drop:+.4f}, {100*drop/within['overall_map_at_k']:+.1f}%)")
        tot = within["same_realm_analogs"] + within["cross_realm_analogs"]
        if tot:
            print(f"  analogs drawn from the query's OWN realm: "
                  f"{100*within['same_realm_analogs']/tot:.1f}%")
        print(f"  {'realm':<16}{'normal':>10}{'cross-realm':>13}{'delta':>10}{'n':>7}")
        for r in sorted(within["per_realm"]):
            a = within["per_realm"][r]["map_at_k"]
            b = cross["per_realm"].get(r, {}).get("map_at_k")
            n = within["per_realm"][r]["n"]
            bs = f"{b:.4f}" if b is not None else "  n/a"
            ds = f"{b-a:+.4f}" if b is not None else "     -"
            print(f"  {r:<16}{a:>10.4f}{bs:>13}{ds:>10}{n:>7}")
        print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"saved {OUT_PATH}")
    print("\nHOW TO READ THIS (C7): the leave-one-realm-out column is the honest")
    print("cross-landscape number. A large drop means retrieval was leaning on")
    print("same-realm neighbours; a small drop means the embedding generalises")
    print("across biogeographic boundaries, which is what the research question asks.")


if __name__ == "__main__":
    main()
