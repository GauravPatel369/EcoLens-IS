"""
EcoLens Step 32 -- WHAT THE BRIDGE BETWEEN THE TWO PILLARS IS ACTUALLY WORTH

WHY THIS EXISTS
---------------
14_analog_ablation.py compares two arms: drivers alone, and drivers plus the eight analog
columns. On the spatial holdout the second wins by +11.5% (PR-AUC 0.3542 -> 0.3949,
p = 0.00076). That result is the centre of the project, and as it stands it has an obvious
rival explanation that nothing in the pipeline rules out:

    A gradient-boosted model given eight MORE COLUMNS of almost any kind has more capacity
    to fit. Maybe retrieval is irrelevant and any eight extra numbers would have helped.

This script settles that, and answers two further questions the professor's review actually
asked but which no existing arm addresses.

    ARM A  drivers only                  the baseline
    ARM B  drivers + REAL analogs        the current headline result
    ARM C  drivers + RANDOM analogs      <-- the control that decides whether the claim holds
    ARM D  retrieval only, no drivers    can retrieval forecast BY ITSELF?
    ARM E  persistence heuristic         does 0.6974 beat a one-line rule of thumb?
    ARM F  neighbourhood heuristic       ditto, using distance to prior loss

ARM C is produced by 13_analog_risk_features.py --random-analogs, which runs the identical
pipeline -- same leakage guards, same top-k, same eight columns, same horizon windows -- and
replaces similarity with noise. If B beats C, similarity is doing the work. If B and C tie,
the +11.5% is an artifact of feature count and the project's central claim does not hold.
Either outcome is reported; that is the point of a control.

ARM D is the direct answer to "can ecosystem analog retrieval contribute to forecasting?"
It uses NO driver features at all: a cell's risk is the similarity-weighted mean of what
happened to its retrieved analogs. It is a forecast made purely by the search engine.

ARMS E and F matter because PR-AUC 0.6974 is currently only ever compared against the base
rate. Beating chance is a weak claim. Beating "there was loss nearby recently, so expect
more" is the claim a reviewer will want.

Every arm is scored on BOTH protocols, on the SAME rows, so the numbers are comparable to
each other in a way the existing ablations are not.

Usage:
    python 32_pillar_bridge.py                   # both protocols, prithvi
    python 32_pillar_bridge.py --model clay
    python 32_pillar_bridge.py --spatial-only
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from config import RESULTS_DIR, RISK_MODEL_DIR, RISK_FEATURES_PATH, RISK_HORIZON_YEARS

OUT_PATH = os.path.join(RESULTS_DIR, "pillar_bridge.json")

DRIVERS = ["baseline_treecover_pct", "distance_to_prior_loss_m", "protected_area",
           "temp_c", "rainfall_mm", "elevation_m", "ruggedness_m"]

ANALOG_COLS = ["analog_prior_loss_rate", "analog_loss_rate_at_horizon",
               "analog_frac_with_loss_at_horizon", "analog_n",
               "analog_trajectory_jaccard", "analog_fragmentation_delta",
               "cell_edge_density", "analog_mean_similarity"]

LABEL = f"label_loss_H{RISK_HORIZON_YEARS}"
KEYS = ["region_id", "cell_lon", "cell_lat", "obs_year"]


def analog_path(model, random_ctrl=False):
    base = (os.path.join(RESULTS_DIR, "analog_cell_features.csv") if model == "prithvi"
            else os.path.join(RESULTS_DIR, f"analog_cell_features_{model}.csv"))
    return base.replace(".csv", "_randctrl.csv") if random_ctrl else base


def fit_score(X_tr, y_tr, X_te, y_te):
    """One learner, no selection. PR-AUC on the held-out side, or None if unscoreable.

    HistGradientBoosting only: this script compares FEATURE SETS, so the learner must be
    held fixed. Running a bake-off inside each arm would confound "arm B is better" with
    "arm B got luckier with model choice".
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None
    m = HistGradientBoostingClassifier(class_weight="balanced", max_depth=6, random_state=42)
    m.fit(X_tr, y_tr)
    return float(average_precision_score(y_te, m.predict_proba(X_te)[:, 1]))


def score_ranking(y_true, scores):
    """PR-AUC for an arm that produces a score directly, with no model to fit."""
    from sklearn.metrics import average_precision_score
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, scores))


def arms_for_split(df, tr_mask, te_mask, has_ctrl):
    """Score every arm on one train/test division of the SAME dataframe."""
    out = {}
    y_tr = df.loc[tr_mask, LABEL].to_numpy(dtype=int)
    y_te = df.loc[te_mask, LABEL].to_numpy(dtype=int)

    def mat(cols, mask):
        return df.loc[mask, cols].to_numpy(dtype="float64")

    out["A_drivers_only"] = fit_score(mat(DRIVERS, tr_mask), y_tr,
                                     mat(DRIVERS, te_mask), y_te)

    real = [c for c in ANALOG_COLS if c in df.columns]
    out["B_drivers_plus_real_analogs"] = fit_score(
        mat(DRIVERS + real, tr_mask), y_tr, mat(DRIVERS + real, te_mask), y_te)

    if has_ctrl:
        ctrl = [c + "__ctrl" for c in real if c + "__ctrl" in df.columns]
        out["C_drivers_plus_random_analogs"] = fit_score(
            mat(DRIVERS + ctrl, tr_mask), y_tr, mat(DRIVERS + ctrl, te_mask), y_te)
    else:
        out["C_drivers_plus_random_analogs"] = None

    # ARM D: retrieval only. No drivers at all -- the forecast IS the analogs' history.
    out["D_retrieval_only"] = fit_score(mat(real, tr_mask), y_tr,
                                       mat(real, te_mask), y_te)
    # and the same idea with no model whatsoever: rank cells directly by what happened to
    # their analogs. If THIS beats the base rate, retrieval alone carries risk signal.
    if "analog_loss_rate_at_horizon" in df.columns:
        out["D2_analog_rate_as_score"] = score_ranking(
            y_te, df.loc[te_mask, "analog_loss_rate_at_horizon"].fillna(0).to_numpy())
    else:
        out["D2_analog_rate_as_score"] = None

    # ARM E: persistence. "This cell was disturbed before, so expect it again."
    # distance_to_prior_loss_m is small when prior loss is close, so negate it to make
    # "closer to past loss" = "higher risk".
    if "distance_to_prior_loss_m" in df.columns:
        d = df.loc[te_mask, "distance_to_prior_loss_m"].to_numpy(dtype="float64")
        d = np.where(np.isnan(d), np.nanmax(d) if np.isfinite(np.nanmax(d)) else 0.0, d)
        out["E_persistence_near_prior_loss"] = score_ranking(y_te, -d)
    else:
        out["E_persistence_near_prior_loss"] = None

    # ARM F: baseline tree cover alone -- the single strongest driver, as a one-column rule.
    if "baseline_treecover_pct" in df.columns:
        out["F_treecover_only"] = score_ranking(
            y_te, df.loc[te_mask, "baseline_treecover_pct"].fillna(0).to_numpy())
    else:
        out["F_treecover_only"] = None

    out["_base_rate"] = float(y_te.mean()) if len(y_te) else None
    out["_n_train"] = int(tr_mask.sum())
    out["_n_test"] = int(te_mask.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description="EcoLens 32: what the pillar bridge is worth")
    ap.add_argument("--model", default="prithvi")
    ap.add_argument("--spatial-only", action="store_true")
    ap.add_argument("--temporal-only", action="store_true")
    args = ap.parse_args()

    feats = pd.read_csv(RISK_FEATURES_PATH)
    real_p = analog_path(args.model)
    if not os.path.exists(real_p):
        sys.exit(f"ERROR: {real_p} not found. Run 13_analog_risk_features.py first.")
    real = pd.read_csv(real_p)

    ctrl_p = analog_path(args.model, random_ctrl=True)
    has_ctrl = os.path.exists(ctrl_p)
    if not has_ctrl:
        print(f"NOTE: no control file at {ctrl_p}.")
        print("      ARM C will be null. Produce it with:")
        print(f"      python 13_analog_risk_features.py --model {args.model} --random-analogs")

    df = feats.merge(real, on=KEYS, how="inner").dropna(subset=[LABEL])
    if has_ctrl:
        ctrl = pd.read_csv(ctrl_p)
        keep = KEYS + [c for c in ANALOG_COLS if c in ctrl.columns]
        ctrl = ctrl[keep].rename(columns={c: c + "__ctrl" for c in keep if c not in KEYS})
        before = len(df)
        df = df.merge(ctrl, on=KEYS, how="inner")
        print(f"control merged: {len(df):,} rows (from {before:,}); "
              f"the control must cover the SAME cells or the comparison is not paired")

    print(f"\n{'=' * 78}")
    print(f"PILLAR BRIDGE -- model {args.model}")
    print(f"{'=' * 78}")
    print(f"  rows {len(df):,}   cells {df.groupby(['cell_lon', 'cell_lat']).ngroups:,}   "
          f"regions {df.region_id.nunique()}   label {LABEL}")

    result = {"generated": datetime.now().isoformat(timespec="seconds"),
              "model": args.model, "label": LABEL,
              "has_random_control": has_ctrl,
              "n_rows": int(len(df)), "n_regions": int(df.region_id.nunique()),
              "protocols": {}}

    # ---- TEMPORAL -------------------------------------------------------------------
    if not args.spatial_only:
        years = sorted(df.obs_year.unique())
        cutoff = years[int(len(years) * 0.8)]
        # same embargo logic as 11: a label at Y covers Y+1..Y+H, so training years within
        # H of the test start would carry test-period outcomes
        tr = df.obs_year <= cutoff - RISK_HORIZON_YEARS
        te = df.obs_year > cutoff
        print(f"\nTEMPORAL: train obs_year <= {cutoff - RISK_HORIZON_YEARS}, "
              f"embargo {cutoff - RISK_HORIZON_YEARS + 1}-{cutoff}, test > {cutoff}")
        r = arms_for_split(df, tr, te, has_ctrl)
        result["protocols"]["temporal"] = r
        report(r)

    # ---- SPATIAL (leave-one-region-out) ---------------------------------------------
    if not args.temporal_only:
        regions = sorted(df.region_id.unique())
        print(f"\nSPATIAL: leave-one-region-out over {len(regions)} regions")
        per_fold = {}
        for rid in regions:
            te = df.region_id == rid
            tr = ~te
            r = arms_for_split(df, tr, te, has_ctrl)
            per_fold[rid] = r
            arms = [k for k in r if not k.startswith("_")]
            print(f"  {rid:12s} " + "  ".join(
                f"{k.split('_')[0]}={r[k]:.4f}" if r[k] is not None else f"{k.split('_')[0]}=--"
                for k in arms))
        means = {}
        for k in next(iter(per_fold.values())):
            if k.startswith("_"):
                continue
            vals = [v[k] for v in per_fold.values() if v[k] is not None]
            means[k] = float(np.mean(vals)) if vals else None
        result["protocols"]["spatial"] = {"per_fold": per_fold, "means": means,
                                          "n_folds": len(regions)}
        print("\n  MEAN ACROSS FOLDS:")
        report(means)

        # the paired test that the headline claim rests on, plus the SAME test against the
        # control -- which is the comparison that decides whether similarity matters
        try:
            from scipy.stats import wilcoxon
            def paired(a, b):
                pa = [(per_fold[r][a], per_fold[r][b]) for r in regions
                      if per_fold[r][a] is not None and per_fold[r][b] is not None]
                if len(pa) < 5:
                    return None
                d = np.array([y - x for x, y in pa])
                return {"n": len(pa), "mean_delta": float(d.mean()),
                        "improved": int((d > 0).sum()),
                        "wilcoxon_p": float(wilcoxon(d).pvalue)}
            tests = {
                "B_vs_A_real_analogs_help": paired("A_drivers_only",
                                                   "B_drivers_plus_real_analogs"),
                "C_vs_A_random_analogs_help": paired("A_drivers_only",
                                                     "C_drivers_plus_random_analogs"),
                "B_vs_C_similarity_matters": paired("C_drivers_plus_random_analogs",
                                                    "B_drivers_plus_real_analogs"),
            }
            result["protocols"]["spatial"]["paired_tests"] = tests
            print("\n  PAIRED TESTS (Wilcoxon over folds):")
            for k, v in tests.items():
                if v is None:
                    print(f"    {k:34s} not computable")
                else:
                    print(f"    {k:34s} delta {v['mean_delta']:+.4f}  "
                          f"{v['improved']}/{v['n']} folds  p={v['wilcoxon_p']:.5f}")
            verdict(tests)
        except ImportError:
            print("  (scipy missing -- paired tests skipped)")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nwrote {OUT_PATH}")


def report(r):
    names = {
        "A_drivers_only": "A  drivers only",
        "B_drivers_plus_real_analogs": "B  drivers + REAL analogs",
        "C_drivers_plus_random_analogs": "C  drivers + RANDOM analogs",
        "D_retrieval_only": "D  retrieval only (no drivers)",
        "D2_analog_rate_as_score": "D2 analog rate as raw score",
        "E_persistence_near_prior_loss": "E  persistence heuristic",
        "F_treecover_only": "F  tree cover only",
    }
    base = r.get("_base_rate")
    for k, label in names.items():
        v = r.get(k)
        if v is None:
            print(f"    {label:34s}      --")
        elif base:
            print(f"    {label:34s} {v:.4f}   lift {v / base:.2f}x")
        else:
            print(f"    {label:34s} {v:.4f}")
    if base:
        print(f"    {'(base rate)':34s} {base:.4f}")


def verdict(tests):
    """State plainly what the control implies, in both directions."""
    bc = tests.get("B_vs_C_similarity_matters")
    ca = tests.get("C_vs_A_random_analogs_help")
    print("\n  VERDICT:")
    if bc is None:
        print("    B vs C not computable -- no verdict on whether similarity matters.")
        return
    if bc["wilcoxon_p"] < 0.05 and bc["mean_delta"] > 0:
        print("    Real analogs beat RANDOM analogs. The gain comes from ecological")
        print("    SIMILARITY, not from having eight more columns. The central claim holds.")
    else:
        print("    Real analogs do NOT beat random analogs at this sample size.")
        print("    The +11.5% cannot be attributed to similarity -- eight columns of noise")
        print("    do about as well. The central claim is NOT supported by this control and")
        print("    must not be stated as a retrieval result.")
    if ca and ca["mean_delta"] > 0 and ca["wilcoxon_p"] < 0.05:
        print("    NOTE: random analogs ALSO beat drivers alone, which means part of the")
        print("    original +11.5% was model capacity rather than transferred information.")


if __name__ == "__main__":
    main()
