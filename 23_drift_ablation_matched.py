"""
EcoLens Step 23 -- MATCHED-SUBSET EMBEDDING-DRIFT ABLATION  (C16)

WHY THIS EXISTS
---------------
`11 --ablation` compares "drivers only" against "drivers + embedding_drift" across the
WHOLE features table. That table has 143,000 rows and `embedding_drift` is filled for
**4,572 of them (3.2 %, 275 cells)** -- computing it costs two STAC searches and two model
forward passes per cell, so it is deliberately sampled, not exhaustive.

Comparing the two arms over all rows therefore asks a nearly meaningless question: the
gradient-boosted models handle NaN natively, so on 96.8 % of rows the "with drift" arm is
simply the "without drift" arm. Any real effect is diluted ~30x before it is measured. The
whole-table run reported -0.0008, which is indistinguishable from zero by construction.

`14_analog_ablation.py` already got this right for the analog features -- it retrains the
baseline **on the same subset** so the comparison isolates the feature rather than the
sample. This script applies the same discipline to `embedding_drift`.

WHAT IT DOES
------------
1. Keep only rows where `embedding_drift` is present.
2. Train both arms on exactly those rows, same temporal split as `11`
   (train obs_year <= cutoff, test after), same model family.
3. Report PR-AUC with bootstrap CIs, and a per-region breakdown so a single region cannot
   carry the result.

A null result here is still a real answer, and an informative one: it would say a cell's
OWN temporal drift carries little, while the disturbance history of ecologically similar
places ELSEWHERE carries a lot (+11.0 %, C15) -- which is an argument for analog transfer
specifically, not for "embeddings help" in general.

Writes risk_model/drift_ablation_matched.json

Run:
    python 23_drift_ablation_matched.py
"""

import json
import os

import numpy as np
import pandas as pd

from config import RISK_FEATURES_PATH, RISK_MODEL_DIR

OUT_PATH = f"{RISK_MODEL_DIR}/drift_ablation_matched.json"

DRIVERS = ["baseline_treecover_pct", "distance_to_prior_loss_m", "protected_area",
           "temp_c", "rainfall_mm", "elevation_m", "ruggedness_m", "obs_year"]
LABEL = "label_loss_within_horizon"
DRIFT = "embedding_drift"


def fit_score(Xtr, ytr, Xte, yte, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    m = HistGradientBoostingClassifier(random_state=seed)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return float(average_precision_score(yte, p)), p


def boot_ci(y, p, n=500, seed=0):
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(average_precision_score(y[idx], p[idx]))
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    df = pd.read_csv(RISK_FEATURES_PATH)
    total_rows = len(df)
    sub = df[df[DRIFT].notna()].copy()
    n_cells = sub.groupby(["cell_lon", "cell_lat"]).ngroups

    print(f"\n{'='*74}")
    print("MATCHED-SUBSET EMBEDDING-DRIFT ABLATION (C16)")
    print(f"{'='*74}")
    print(f"  full table         : {total_rows:,} rows")
    print(f"  rows WITH drift    : {len(sub):,} ({100*len(sub)/total_rows:.1f} %), "
          f"{n_cells} cells, {sub.region_id.nunique()} regions")
    print("  Both arms are trained on THESE rows only -- that is the whole point;")
    print("  the whole-table comparison is diluted ~30x by missing values.\n")

    years = sorted(sub.obs_year.unique())
    cutoff = years[int(len(years) * 0.8)] if len(years) > 1 else years[0]
    tr, te = sub[sub.obs_year <= cutoff], sub[sub.obs_year > cutoff]
    print(f"  temporal split at obs_year {cutoff}: "
          f"train {len(tr):,} ({tr[LABEL].mean()*100:.1f} % positive), "
          f"test {len(te):,} ({te[LABEL].mean()*100:.1f} % positive)")

    if len(te) < 50 or te[LABEL].nunique() < 2:
        print("\n  Test fold is too small or single-class -- cannot evaluate. "
              "Compute drift for more cells (10 --with-embedding-drift "
              "--drift-cells-per-location N).")
        return

    # Drop features that are constant or entirely missing IN THIS SUBSET. `11` does the
    # same ("Skipped constant/uninformative feature(s)"); without it sklearn's histogram
    # binner raises "window shape cannot be larger than input array shape". Here
    # `protected_area` is 100 % NaN on these rows -- WDPA is unavailable, so it is only
    # ever filled from the curated per-location flag, which does not reach grid cells.
    usable = [c for c in DRIVERS if tr[c].nunique(dropna=True) >= 2]
    dropped = [c for c in DRIVERS if c not in usable]
    if dropped:
        print(f"  skipping constant/empty feature(s) in this subset: {dropped}")

    ytr, yte = tr[LABEL].to_numpy(), te[LABEL].to_numpy()
    arms, preds = {}, {}
    for name, feats in (("drivers_only", usable), ("drivers_plus_drift", usable + [DRIFT])):
        ap, p = fit_score(tr[feats].to_numpy(), ytr, te[feats].to_numpy(), yte)
        preds[name] = p          # reuse for the per-region table; refitting per region
                                 # would train 2 new models per region on the same data
        lo, hi = boot_ci(yte, p)
        arms[name] = {"features": feats, "pr_auc": ap, "ci95": [lo, hi]}
        print(f"\n  {name:<20} PR-AUC {ap:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    delta = arms["drivers_plus_drift"]["pr_auc"] - arms["drivers_only"]["pr_auc"]
    pct = 100.0 * delta / arms["drivers_only"]["pr_auc"]
    print(f"\n  delta {delta:+.4f} ({pct:+.1f} %)")

    lo_d, hi_d = arms["drivers_only"]["ci95"]
    overlaps = not (arms["drivers_plus_drift"]["pr_auc"] > hi_d
                    or arms["drivers_plus_drift"]["pr_auc"] < lo_d)
    verdict = ("NO DETECTABLE BENEFIT -- the drift arm sits inside the driver-only CI"
               if overlaps else
               ("DRIFT HELPS" if delta > 0 else "DRIFT HURTS"))
    print(f"  VERDICT: {verdict}")

    # per-region, so one region cannot carry the result
    per_region = {}
    p_only, p_drift = preds['drivers_only'], preds['drivers_plus_drift']
    print(f"\n  {'region':<14}{'n_test':>8}{'drivers':>10}{'+drift':>10}{'delta':>9}")
    for rid, g in te.groupby("region_id"):
        if len(g) < 30 or g[LABEL].nunique() < 2:
            continue
        m = te.region_id == rid
        from sklearn.metrics import average_precision_score
        a = average_precision_score(g[LABEL], p_only[m.to_numpy()])
        b = average_precision_score(g[LABEL], p_drift[m.to_numpy()])
        per_region[rid] = {"n_test": int(len(g)), "drivers_only": float(a),
                           "drivers_plus_drift": float(b), "delta": float(b - a)}
        print(f"  {rid:<14}{len(g):>8}{a:>10.4f}{b:>10.4f}{b-a:>+9.4f}")

    improved = sum(1 for v in per_region.values() if v["delta"] > 0)
    if per_region:
        print(f"\n  regions improved: {improved}/{len(per_region)}")

    out = {"total_rows": int(total_rows), "rows_with_drift": int(len(sub)),
           "cells_with_drift": int(n_cells), "regions": int(sub.region_id.nunique()),
           "cutoff_year": int(cutoff), "arms": arms, "delta": float(delta),
           "delta_pct": float(pct), "verdict": verdict, "per_region": per_region,
           "regions_improved": improved, "n_regions_scored": len(per_region)}
    os.makedirs(RISK_MODEL_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
