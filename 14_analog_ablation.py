"""
EcoLens Step 14 -- DOES ANALOG TRANSFER IMPROVE FOREST-LOSS FORECASTING?
(Phase 4; faculty comments C1, C15, C16)

This is the test the re-framed research question reduces to:

    Can geospatial foundation models identify ecological analogs whose historical
    disturbance trajectories improve forest-loss risk forecasting ACROSS
    GEOGRAPHICALLY DISTINCT LANDSCAPES?

Arms compared, all on IDENTICAL rows, split and protocol:
    1. drivers only                 -- the conventional baseline
    2. drivers + analog-transferred -- what 13_analog_risk_features.py produced

TWO METHODOLOGICAL POINTS THAT DECIDE WHETHER THIS MEANS ANYTHING
-----------------------------------------------------------------
1. THE BASELINE IS RECOMPUTED ON THE SAME SUBSET.
   13 embeds a sample of cells (~317 of 8,469), so the analog arm covers only
   those. Comparing it against the full-grid driver baseline from 11 would be
   comparing two different datasets and calling the difference an effect. Every
   arm here is trained and tested on exactly the merged rows.

2. THE SPATIAL-HOLDOUT NUMBER IS THE RESULT; THE TEMPORAL ONE IS NOT.
   implementation.md 3.4d: on a temporal split the driver-only model already
   scores PR-AUC 0.6923, but under leave-one-location-out it collapses to a mean
   0.3172 with 4 of 17 regions BELOW random. It memorises regions rather than
   learning a transferable mechanism. The hypothesis is specifically about
   transfer ACROSS landscapes, so a gain on the temporal split would prove almost
   nothing while a gain under spatial holdout is the actual claim.

REPORT THE RESULT EITHER WAY (implementation.md 9). "Analog features did not
help" answers the research question; tuning until they help does not.

Run:
    python 14_analog_ablation.py
    python 14_analog_ablation.py --spatial-holdout
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

from config import RISK_FEATURES_PATH, RESULTS_DIR, RISK_MODEL_DIR

# Input and outputs are BOTH model-suffixed. 13 already learned this lesson the hard
# way -- its Clay run overwrote Prithvi's CSV because only one path existed -- so the
# ablation must not repeat it. Prithvi keeps the unsuffixed names so existing results
# and every reference to them stay valid.
def _paths(model_key):
    sfx = "" if model_key == "prithvi" else f"_{model_key}"
    return (f"{RESULTS_DIR}/analog_cell_features{sfx}.csv",
            f"{RISK_MODEL_DIR}/analog_ablation_temporal{sfx}.json",
            f"{RISK_MODEL_DIR}/analog_ablation_spatial{sfx}.json")


# separate files per mode: the two runs answer different questions and an
# earlier version had the temporal run silently overwrite the spatial one.
ANALOG_PATH, OUT_JSON_TEMPORAL, OUT_JSON_SPATIAL = _paths("prithvi")

# ONLY features that depend on the ANALOGS. This is the whole point of the test:
# if the analog arm won because it was handed an extra LOCAL feature, the result
# would be about that feature, not about analog transfer.
ANALOG_FEATURES = [
    "analog_prior_loss_rate",
    "analog_loss_rate_at_horizon",
    "analog_frac_with_loss_at_horizon",
    "analog_trajectory_jaccard",
    "analog_fragmentation_delta",
    "analog_mean_similarity",
]

# `cell_edge_density` is a purely LOCAL fragmentation measure of the query cell --
# it says nothing about any analog. It is given to BOTH arms so the comparison
# isolates analog transfer rather than crediting it with a local signal the
# baseline was denied. (analog_fragmentation_delta is cell_edge minus the analogs'
# mean edge, so it does carry a local component; holding cell_edge_density fixed in
# both arms is what makes that difference interpretable as an analog contribution.)
SHARED_EXTRA_FEATURES = ["cell_edge_density"]


def rf():
    if "_rf" not in sys.modules:
        spec = importlib.util.spec_from_file_location("_rf", "11_forest_risk_forecast.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["_rf"] = m
        spec.loader.exec_module(m)
    return sys.modules["_rf"]


def merge():
    import pandas as pd
    if not os.path.exists(ANALOG_PATH):
        raise FileNotFoundError(
            f"{ANALOG_PATH} not found -- run 13_analog_risk_features.py first.")
    drivers = pd.read_csv(RISK_FEATURES_PATH)
    analog = pd.read_csv(ANALOG_PATH)
    analog = analog.drop(columns=[c for c in ("region_id", "analog_top_ecosystems",
                                              "analog_n", "analog_similarity_spread")
                                  if c in analog.columns])
    # analog_similarity_spread is dropped deliberately: measured std 0.0002 on
    # Prithvi -- a constant column cannot inform a model and only dilutes the
    # permutation-importance table.
    for df in (drivers, analog):
        df["cell_lon"] = df["cell_lon"].round(5)
        df["cell_lat"] = df["cell_lat"].round(5)
    merged = drivers.merge(analog, on=["cell_lon", "cell_lat", "obs_year"], how="inner")
    return merged


def spatial_folds(rows, feature_names, r):
    """Leave-one-region-out, returning per-region AP so the two arms can be paired.

    Mirrors 11.run_spatial_holdout's fold construction exactly (same model, same
    seed, same constant-feature filter) -- only the return value differs.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    regions = sorted({row["region_id"] for row in rows})
    out = {}
    for held in regions:
        tr = [x for x in rows if x["region_id"] != held]
        te = [x for x in rows if x["region_id"] == held]
        Xtr, ytr, _ = r.build_matrix(tr, feature_names)
        Xte, yte, _ = r.build_matrix(te, feature_names)
        keep = [j for j in range(len(feature_names))
                if len(np.unique(Xtr[:, j][~np.isnan(Xtr[:, j])])) >= 2]
        if not keep or len(np.unique(yte)) < 2:
            continue
        m = HistGradientBoostingClassifier(random_state=42, max_iter=100)
        m.fit(Xtr[:, keep], ytr)
        s = m.predict_proba(Xte[:, keep])[:, 1]
        try:
            roc = roc_auc_score(yte, s)
        except ValueError:
            roc = float("nan")
        out[held] = {"ap": float(average_precision_score(yte, s)),
                     "roc_auc": float(roc), "n": int(len(yte)), "pos": int(yte.sum())}
    return out


def arm(rows, feature_names, label, X_all, y, obs_years, cutoff):
    r = rf()
    idx = [DRIVER_ALL.index(f) for f in feature_names]
    X = X_all[:, idx]
    train = obs_years <= cutoff
    test = obs_years > cutoff
    if train.sum() == 0 or test.sum() == 0 or len(np.unique(y[test])) < 2:
        print(f"  [{label}] not enough data on both sides of the split")
        return None
    return r.train_and_evaluate(X[train], y[train], X[test], y[test],
                                feature_names, label)


def main():
    ap_ = argparse.ArgumentParser(description="EcoLens 14: analog-transfer ablation")
    ap_.add_argument("--spatial-holdout", action="store_true",
                     help="leave-one-region-out instead of the temporal split. THIS IS "
                          "THE ARM THAT TESTS THE HYPOTHESIS.")
    ap_.add_argument("--model", default="prithvi",
                     help="which model's analog features to test; reads "
                          "results/analog_cell_features[_MODEL].csv and writes "
                          "model-suffixed JSON so arms never overwrite each other.")
    args = ap_.parse_args()

    global ANALOG_PATH, OUT_JSON_TEMPORAL, OUT_JSON_SPATIAL
    ANALOG_PATH, OUT_JSON_TEMPORAL, OUT_JSON_SPATIAL = _paths(args.model)
    print(f"[model: {args.model}]  features <- {ANALOG_PATH}")

    import pandas as pd
    r = rf()

    merged = merge()
    cells = merged[["cell_lon", "cell_lat"]].drop_duplicates().shape[0]
    print(f"\n{'='*74}")
    print("EcoLens 14: does analog transfer improve forest-loss forecasting?")
    print(f"{'='*74}")
    print(f"merged rows      : {len(merged):,}  ({cells} cells with analog features)")
    print(f"regions          : {merged.region_id.nunique()}")
    print(f"positive rate    : {merged[r.TARGET_COLUMN].mean():.4f}")
    print(f"analog features  : {len(ANALOG_FEATURES)}")
    print("\nNOTE: the driver-only arm below is retrained on THESE rows, not taken from")
    print("11's full-grid run -- otherwise the comparison would be across two datasets.")

    global DRIVER_ALL
    BASE_FEATURES = r.DRIVER_FEATURES + SHARED_EXTRA_FEATURES
    DRIVER_ALL = BASE_FEATURES + ANALOG_FEATURES
    rows = merged.to_dict("records")

    if args.spatial_holdout:
        print(f"\n{'='*74}")
        print("SPATIAL HOLDOUT (leave-one-region-out) -- THE HYPOTHESIS TEST")
        print(f"{'='*74}")
        # 11.run_spatial_holdout prints but returns None, so the fold loop is
        # reimplemented here: comparing two arms by their MEANS alone is not a test
        # when the fold std is ~0.19 across 16 regions. What matters is whether the
        # SAME region improves when analog features are added, so the folds are paired.
        per_arm = {}
        for label, feats in (("drivers_only", BASE_FEATURES),
                             ("drivers_plus_analog", DRIVER_ALL)):
            per_arm[label] = spatial_folds(rows, feats, r)

        regions = sorted(per_arm["drivers_only"].keys())
        base = np.array([per_arm["drivers_only"][g]["ap"] for g in regions])
        aug = np.array([per_arm["drivers_plus_analog"][g]["ap"] for g in regions])
        diff = aug - base

        print(f"\n{'='*74}")
        print("PAIRED PER-REGION COMPARISON")
        print(f"{'='*74}")
        print(f"  {'region':<16}{'drivers':>10}{'+analog':>10}{'delta':>10}   n_pos")
        for g, b, a in sorted(zip(regions, base, aug), key=lambda t: t[2] - t[1]):
            n = per_arm['drivers_only'][g]['pos']
            print(f"  {g:<16}{b:>10.4f}{a:>10.4f}{a-b:>+10.4f}   {n}")

        wins = int((diff > 0).sum())
        print(f"\n  mean drivers only        : {base.mean():.4f}")
        print(f"  mean drivers + analog    : {aug.mean():.4f}")
        print(f"  mean delta               : {diff.mean():+.4f}  "
              f"({100*diff.mean()/base.mean():+.1f}%)")
        print(f"  regions improved         : {wins}/{len(regions)}")

        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(aug, base)
            print(f"  paired Wilcoxon          : p = {p:.4f}")
        except Exception as e:
            p = None
            print(f"  paired Wilcoxon unavailable ({type(e).__name__})")

        verdict = ("ANALOG FEATURES HELP across landscapes"
                   if p is not None and p < 0.05 and diff.mean() > 0 else
                   "NOT SIGNIFICANT -- report as a null result (implementation.md 9)")
        print(f"\n  VERDICT: {verdict}")

        os.makedirs(RISK_MODEL_DIR, exist_ok=True)
        with open(OUT_JSON_SPATIAL, "w", encoding="utf-8") as f:
            json.dump({"mode": "spatial_holdout", "cells": cells, "rows": len(merged),
                       "regions": regions,
                       "drivers_only_ap": base.tolist(),
                       "drivers_plus_analog_ap": aug.tolist(),
                       "mean_drivers_only": float(base.mean()),
                       "mean_drivers_plus_analog": float(aug.mean()),
                       "mean_delta": float(diff.mean()),
                       "regions_improved": wins,
                       "n_regions": len(regions),
                       "wilcoxon_p": (float(p) if p is not None else None),
                       "verdict": verdict}, f, indent=2)
        print(f"  saved {OUT_JSON_SPATIAL}")
        return

    X_all, y, obs_years = r.build_matrix(rows, DRIVER_ALL)
    all_years = sorted(set(int(v) for v in obs_years))
    cutoff = all_years[int(len(all_years) * 0.8)] if len(all_years) > 1 else all_years[0]
    print(f"temporal cutoff  : {cutoff} (train <= {cutoff}, test > {cutoff})")

    out = {}
    base = arm(rows, BASE_FEATURES, "ARM 1: drivers only (+ local edge density)", X_all, y, obs_years, cutoff)
    both = arm(rows, DRIVER_ALL, "ARM 2: drivers + analog-transferred", X_all, y, obs_years, cutoff)

    print(f"\n{'='*74}")
    print("RESULT -- temporal split")
    print(f"{'='*74}")
    if base and both:
        d = both["ap"] - base["ap"]
        print(f"  drivers only            PR-AUC {base['ap']:.4f}")
        print(f"  drivers + analog        PR-AUC {both['ap']:.4f}")
        print(f"  delta                          {d:+.4f}  ({100*d/base['ap']:+.1f}%)")
        print()
        print("  Read this against the spatial-holdout run (--spatial-holdout), which is")
        print("  the arm the hypothesis is actually about. A temporal-split gain mostly")
        print("  shows the feature is informative WITHIN regions the model already knows.")
        out = {"mode": "temporal", "cutoff": cutoff, "cells": cells, "rows": len(merged),
               "drivers_only_ap": base["ap"], "drivers_plus_analog_ap": both["ap"],
               "delta_ap": d}
        os.makedirs(RISK_MODEL_DIR, exist_ok=True)
        with open(OUT_JSON_TEMPORAL, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved {OUT_JSON_TEMPORAL}")


if __name__ == "__main__":
    main()
