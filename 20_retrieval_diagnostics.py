"""
EcoLens Step 20 -- SIMILARITY-THRESHOLD SWEEP + CLUSTER QUALITY  (C22, C10)

Two things the review asked for that need no new data, only the embeddings already
on disk.

A. SIMILARITY-THRESHOLD SWEEP (C22: "...and similarity thresholds")
   Retrieval so far always returns top-K, which forces an answer even when nothing
   in the database is a real analog. In practice a user needs to know: at what
   cosine score is a match worth trusting? This sweeps the cut-off and reports
   precision / recall / F1 at each, so an operating point can be CHOSEN and
   JUSTIFIED rather than assumed.

B. CLUSTER QUALITY (C10: "...demonstrate whether ecologically similar ecosystems
   naturally cluster in the learned feature space")
   t-SNE already shows this visually in 08, but a picture is not evidence. Two
   standard numbers make it checkable:
     * silhouette score -- do points sit closer to their own ecosystem than to the
       nearest other one? (-1 worst, +1 best; ~0 means no structure)
     * adjusted Rand index -- if the space is clustered blindly into k groups
       (k = number of ecosystems), how well does that agree with the true labels?
       Adjusted so that random agreement scores 0.

Both are computed on the GROUPED footing used everywhere else: a patch's own base
location never counts as a match, so sub-crop near-duplicates cannot inflate anything.

Writes results/retrieval_diagnostics.json

Run:
    python 20_retrieval_diagnostics.py
"""

import json
import os

import numpy as np

from config import METADATA_CATALOG_PATH, RESULTS_DIR, SUPPORTED_MODELS

OUT_PATH = f"{RESULTS_DIR}/retrieval_diagnostics.json"
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]


def load_model(model_key):
    with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    key = "prithvi_embedding" if model_key == "prithvi" else f"{model_key}_embedding"
    bases, ecos, vecs = [], [], []
    for e in catalog:
        p = e.get(key)
        if not p or not os.path.exists(p):
            continue
        v = np.load(p).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        bases.append(e.get("base_id", e["id"].rsplit("_p", 1)[0]))
        ecos.append(e["ecosystem"])
        vecs.append(v)
    return np.array(bases), np.array(ecos), np.stack(vecs)


def threshold_sweep(bases, ecos, X):
    """For each cut-off: of the pairs we ACCEPT as analogs, how many share the
    query's ecosystem (precision), and what share of all same-ecosystem pairs do we
    catch (recall)?"""
    S = X @ X.T
    n = len(X)
    iu = np.triu_indices(n, k=1)
    sims = S[iu]
    same_eco = (ecos[iu[0]] == ecos[iu[1]])
    diff_base = (bases[iu[0]] != bases[iu[1]])     # GROUPED: drop own-location pairs
    sims, same_eco = sims[diff_base], same_eco[diff_base]

    total_rel = int(same_eco.sum())
    rows = []
    for t in THRESHOLDS:
        acc = sims >= t
        n_acc = int(acc.sum())
        if n_acc == 0:
            rows.append({"threshold": t, "accepted": 0, "precision": None,
                         "recall": 0.0, "f1": None})
            continue
        tp = int((acc & same_eco).sum())
        prec = tp / n_acc
        rec = tp / total_rel if total_rel else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"threshold": t, "accepted": n_acc, "precision": prec,
                     "recall": rec, "f1": f1})
    return rows, float(same_eco.mean()), total_rel


def cluster_quality(ecos, X):
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    labels = np.unique(ecos)
    y = np.searchsorted(labels, ecos)
    sil = float(silhouette_score(X, y, metric="cosine"))
    km = KMeans(n_clusters=len(labels), random_state=0, n_init=10).fit(X)
    ari = float(adjusted_rand_score(y, km.labels_))
    return {"silhouette_cosine": sil, "adjusted_rand_index": ari,
            "n_ecosystems": int(len(labels))}


def main():
    report = {}
    print(f"\n{'='*78}")
    print("EcoLens 20: similarity-threshold sweep (C22) + cluster quality (C10)")
    print(f"{'='*78}")

    for mk in SUPPORTED_MODELS:
        try:
            bases, ecos, X = load_model(mk)
        except Exception as e:
            print(f"[{mk}] skipped ({type(e).__name__})")
            continue
        rows, base_rate, total_rel = threshold_sweep(bases, ecos, X)
        cq = cluster_quality(ecos, X)
        best = max((r for r in rows if r["f1"] is not None), key=lambda r: r["f1"],
                   default=None)
        report[mk] = {"label": SUPPORTED_MODELS[mk]["label"], "sweep": rows,
                      "chance_precision": base_rate, "cluster": cq,
                      "best_f1_threshold": best["threshold"] if best else None}

        print(f"\n[{SUPPORTED_MODELS[mk]['label']}]")
        print(f"  cluster quality: silhouette {cq['silhouette_cosine']:+.4f}   "
              f"adjusted Rand {cq['adjusted_rand_index']:.4f}   "
              f"({cq['n_ecosystems']} ecosystems)")
        print(f"  chance precision (share of all cross-location pairs that happen to "
              f"share an ecosystem): {base_rate:.4f}")
        print(f"  {'cut-off':>8}{'accepted':>11}{'precision':>11}{'recall':>9}{'F1':>8}")
        for r in rows:
            if r["precision"] is None:
                print(f"  {r['threshold']:>8.2f}{r['accepted']:>11}{'  -':>11}{'  -':>9}{'  -':>8}")
                continue
            mark = "  <-- best F1" if best and r["threshold"] == best["threshold"] else ""
            print(f"  {r['threshold']:>8.2f}{r['accepted']:>11,}{r['precision']:>11.4f}"
                  f"{r['recall']:>9.4f}{r['f1']:>8.4f}{mark}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {OUT_PATH}")
    print("\nHOW TO READ THIS:")
    print("  * precision at a cut-off vs 'chance precision' says whether that score")
    print("    actually means anything. A model whose precision never rises far above")
    print("    chance has no usable threshold, however good its top-K ranking looks.")
    print("  * silhouette near 0 means ecosystems do NOT form separated clusters, even")
    print("    if a t-SNE picture looks organised -- t-SNE will always draw clumps.")


if __name__ == "__main__":
    main()
