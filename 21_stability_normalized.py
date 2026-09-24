"""
EcoLens Step 21 -- SCALE-NORMALISED STABILITY  (C10, C24)

WHY THIS EXISTS
---------------
`12` reports raw cosine similarity between two acquisitions of the same place, and it is
tempting to read those numbers across models: Prithvi 0.9059 vs Satlas 0.4485 on the
atmospheric axis looks like Prithvi is twice as haze-robust.

**That comparison is invalid.** Each model has its own similarity scale. Prithvi's
embeddings are strongly anisotropic -- its cosines between UNRELATED catalog patches sit
around 0.98-0.998 (implementation.md C22/C24) -- so for Prithvi even a *wrong* match scores
high, and 0.9059 may be a poor score on its own scale. Satlas spreads its scores far more
widely, so 0.4485 may be a good one.

WHAT THIS COMPUTES
------------------
For each model, the background distribution of cosine similarity between patches of
DIFFERENT base locations, drawn from the same catalog `07`/`20` evaluate on. Then each
stability score is placed on that scale:

    z = (stability_cosine - background_mean) / background_std

z answers the question that actually matters: *when the same place is re-imaged in another
season or through haze, does it still look more like itself than two random places look
like each other -- and by how many standard deviations?* That IS comparable across models.

Writes results/stability_normalized.json

Run:
    python 21_stability_normalized.py
"""

import json
import os

import numpy as np

from config import METADATA_CATALOG_PATH, RESULTS_DIR, SUPPORTED_MODELS

OUT_PATH = f"{RESULTS_DIR}/stability_normalized.json"

# seasonal  = July vs January       (does season break it?)
# atmos     = clear vs hazy, same window (does haze break it?)
# interann  = July 2023 vs July 2021    (does a different YEAR break it? -- the one that
#                                        decides whether a once-acquired catalog stays valid)
AXES = ("seasonal", "atmospheric", "interannual")


def background(model_key, rng, n_pairs=200_000):
    """Mean/std cosine between patches of DIFFERENT base locations."""
    with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    key = "prithvi_embedding" if model_key == "prithvi" else f"{model_key}_embedding"
    bases, vecs = [], []
    for e in catalog:
        p = e.get(key)
        if not p or not os.path.exists(p):
            continue
        v = np.load(p).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        bases.append(e.get("base_id", e["id"].rsplit("_p", 1)[0]))
        vecs.append(v)
    if len(vecs) < 50:
        return None
    X = np.stack(vecs)
    b = np.array(bases)
    i = rng.integers(0, len(X), n_pairs)
    j = rng.integers(0, len(X), n_pairs)
    keep = b[i] != b[j]                      # GROUPED: never a patch against its own site
    i, j = i[keep], j[keep]
    sims = np.einsum("ij,ij->i", X[i], X[j])
    # Keep the raw samples: z assumes a Gaussian background and these distributions are
    # visibly not Gaussian (Prithvi's is crushed against 1.0). The percentile below is
    # distribution-free and is the number to quote.
    return {"mean": float(sims.mean()), "std": float(sims.std()),
            "p95": float(np.percentile(sims, 95)), "n_pairs": int(len(sims)),
            "n_patches": int(len(X)), "_samples": sims}


def main():
    rng = np.random.default_rng(0)
    stab = {}
    for name, path in (("seasonal", f"{RESULTS_DIR}/temporal_stability.json"),
                       ("atmospheric", f"{RESULTS_DIR}/atmospheric_stability.json"),
                       ("interannual", f"{RESULTS_DIR}/interannual_stability.json")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                stab[name] = json.load(f)

    report = {}
    print(f"\n{'='*84}")
    print("SCALE-NORMALISED STABILITY -- raw cosines are NOT comparable across models")
    print(f"{'='*84}")
    print(f"  {'model':<14}{'bg mean':>9}{'bg std':>8}"
          f"{'  |':>3}{'seasonal':>10}{'pctile':>8}"
          f"{'  |':>3}{'atmos':>9}{'pctile':>8}"
          f"{'  |':>3}{'interann':>10}{'pctile':>8}")

    for mk in SUPPORTED_MODELS:
        bg = background(mk, rng)
        if bg is None:
            continue
        samples = bg.pop("_samples")
        row = {"label": SUPPORTED_MODELS[mk]["label"], "background": bg}
        cells = []
        for axis in AXES:
            d = stab.get(axis, {}).get(mk)
            if not d:
                row[axis] = None
                cells.append((None, None, None))
                continue
            # Atmospheric: use only pairs with genuine PATCH-level haze (see 12).
            recs = d["per_location"]
            if axis == "atmospheric":
                recs = [r for r in recs
                        if r.get("winter_patch_blue", 0) - r.get("summer_patch_blue", 0) >= 150.0]
            if not recs:
                row[axis] = None
                cells.append((None, None, None))
                continue
            m = float(np.mean([r["similarity"] for r in recs]))
            z = (m - bg["mean"]) / (bg["std"] + 1e-12)
            # Distribution-free: share of background pairs the re-imaged location beats.
            # 0.50 = the same place across dates is no more similar to itself than two
            # random places are to each other. 0.99 = it is clearly still recognisable.
            pct = float((samples < m).mean())
            row[axis] = {"mean_cosine": m, "n": len(recs),
                         "z_above_background": float(z), "percentile": pct}
            cells.append((m, z, pct))
        report[mk] = row

        def fmt(c):
            return (f"{c[0]:>9.4f}{c[2]*100:>7.1f}%" if c[0] is not None
                    else f"{'  -':>9}{'  -':>8}")
        print(f"  {SUPPORTED_MODELS[mk]['label']:<14}{bg['mean']:>9.4f}{bg['std']:>8.4f}"
              + "".join(f"{'  |':>3}{fmt(c)}" for c in cells))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {OUT_PATH}")
    print("\nHOW TO READ THIS:")
    print("  z = how many standard deviations above the model's OWN background a re-imaged")
    print("  location scores. Higher z = the model still recognises the place. A model with")
    print("  a high raw cosine but a low z is simply one whose cosines are all high --")
    print("  it has not demonstrated stability, only anisotropy.")
    print("  pctile is the same idea without assuming a Gaussian background, and is the")
    print("  number to quote: 50% means a place re-imaged on another date is no more like")
    print("  itself than two random places are like each other.")


if __name__ == "__main__":
    main()
