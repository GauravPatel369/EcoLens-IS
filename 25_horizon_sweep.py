"""
EcoLens Step 25 -- RISK HORIZON / TEMPORAL WINDOW SENSITIVITY  (C22)

WHAT C22 ASKED AND WHY THIS WAS RECORDED AS BLOCKED
---------------------------------------------------
C22 wants sensitivity to the **temporal window**. The risk model labels a cell positive if
it loses forest within `RISK_HORIZON_YEARS = 2` of the observation year, and that 2 was
never justified. Sweeping it looked expensive because the label is produced inside
`10_grid_tiling_labels.py`, a ~75-minute run per setting.

It is not expensive. The label only depends on **which years a cell lost forest**, and that
history can be read once and re-thresholded for any horizon:

    label(H) = any(obs_year < loss_year <= obs_year + H)

So one pass over Hansen gives every horizon at once.

THE VALIDATION THAT MAKES THIS TRUSTWORTHY
------------------------------------------
The first version of this script tried the cheap route: rebuild the labels from
`09.get_disturbance_history` rather than re-running `10`. It included a self-check comparing
the reconstructed H=2 label against the table's own column, and **that check refused the
result** -- 75.9 % agreement, and a reconstructed positive rate of 49.7 % against the real
25.6 %. That reader uses a wider footprint than 10's cell window, so the sweep would have
described a model that was never trained.

The labels are therefore produced where they belong, inside `10`, which now emits
`label_loss_H1/H2/H3/H5` alongside the native column in a single pass -- the horizon is only
a threshold on a `lossyear` array already in memory, so extra horizons are nearly free.
`label_loss_H2` reproduces `label_loss_within_horizon` exactly (both 25.6 % positive), which
is the check that the alternate labels are computed the same way as the real one.

This script now reads those columns directly.

Writes results/horizon_sweep.json

Run:
    python 25_horizon_sweep.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from config import RISK_FEATURES_PATH, RESULTS_DIR

OUT_PATH = f"{RESULTS_DIR}/horizon_sweep.json"
HIST_CACHE = f"{RESULTS_DIR}/cell_loss_history.json"
HORIZONS = [1, 2, 3, 5]
NATIVE_HORIZON = 2
MIN_AGREEMENT = 0.90        # below this the shortcut is not valid; abort rather than mislead

DRIVERS = ["baseline_treecover_pct", "distance_to_prior_loss_m", "temp_c",
           "rainfall_mm", "elevation_m", "ruggedness_m", "obs_year"]


def load_history(cells):
    if os.path.exists(HIST_CACHE):
        with open(HIST_CACHE, encoding="utf-8") as f:
            return {tuple(map(float, k.split("_"))): v for k, v in json.load(f).items()}
    import importlib.util
    spec = importlib.util.spec_from_file_location("_t09", "09_explainability_engine.py")
    t09 = importlib.util.module_from_spec(spec)
    sys.modules["_t09"] = t09
    spec.loader.exec_module(t09)

    from tqdm import tqdm
    hist = {}
    for lon, lat in tqdm(cells, total=len(cells), desc="hansen history"):
        try:
            h = t09.get_disturbance_history(lon, lat)
            hist[(round(lon, 5), round(lat, 5))] = sorted(h.get("loss_years") or []) if h else []
        except Exception:
            hist[(round(lon, 5), round(lat, 5))] = []
    with open(HIST_CACHE, "w", encoding="utf-8") as f:
        json.dump({f"{k[0]}_{k[1]}": v for k, v in hist.items()}, f)
    return hist


def label_for(hist, lon, lat, obs_year, H):
    ys = hist.get((round(lon, 5), round(lat, 5)), [])
    return int(any(obs_year < y <= obs_year + H for y in ys))


def evaluate(df, label_col):
    """Temporal split, same cutoff rule as 11; PR-AUC of the best-of-one model."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    years = sorted(df.obs_year.unique())
    cutoff = years[int(len(years) * 0.8)]
    tr, te = df[df.obs_year <= cutoff], df[df.obs_year > cutoff]
    if te[label_col].nunique() < 2 or tr[label_col].nunique() < 2:
        return None, None, None
    feats = [c for c in DRIVERS if tr[c].nunique(dropna=True) >= 2]
    m = HistGradientBoostingClassifier(random_state=0)
    m.fit(tr[feats].to_numpy(), tr[label_col].to_numpy())
    p = m.predict_proba(te[feats].to_numpy())[:, 1]
    return (float(average_precision_score(te[label_col].to_numpy(), p)),
            float(te[label_col].mean()), int(len(te)))


def main():
    df = pd.read_csv(RISK_FEATURES_PATH)
    print(f"\n{'='*78}")
    print("RISK HORIZON / TEMPORAL WINDOW SENSITIVITY (C22)")
    print(f"{'='*78}")
    print(f"  {len(df):,} rows, {df.groupby(['cell_lon','cell_lat']).ngroups:,} cells")

    cols = {H: f"label_loss_H{H}" for H in HORIZONS if f"label_loss_H{H}" in df.columns}
    if not cols:
        raise SystemExit("No label_loss_H* columns. Re-run 10_grid_tiling_labels.py -- it "
                         "emits them alongside the native label.")

    # Consistency check: the H=2 column must reproduce the native label exactly, since both
    # come from the same threshold on the same array. If they diverge, the alternate labels
    # were not computed the way the model's own label was.
    if NATIVE_HORIZON in cols:
        same = (df[cols[NATIVE_HORIZON]] == df.label_loss_within_horizon).mean()
        print(f"  check: label_loss_H{NATIVE_HORIZON} vs label_loss_within_horizon "
              f"-> {same*100:.2f} % identical")
        if same < 0.999:
            raise SystemExit("H2 column does not reproduce the native label; aborting "
                             "rather than sweeping labels of unknown provenance.")

    rows = {}
    print(f"\n{'horizon':>8}{'positive rate':>16}{'test rows':>11}{'PR-AUC':>10}{'lift':>9}")
    for H, col in cols.items():
        ap, posrate, n_te = evaluate(df, col)
        if ap is None:
            print(f"  {H:>8}{'  single-class':>16}")
            continue
        lift = ap / posrate if posrate else None
        rows[str(H)] = {"horizon_years": H, "positive_rate": posrate, "n_test": n_te,
                        "pr_auc": ap, "lift_over_base_rate": lift,
                        "is_current_default": H == NATIVE_HORIZON}
        mark = "   <-- current default" if H == NATIVE_HORIZON else ""
        print(f"  {H:>8}{posrate*100:>15.1f}%{n_te:>11,}{ap:>10.4f}{lift:>9.2f}{mark}")

    out = {"source": "label_loss_H* columns emitted by 10_grid_tiling_labels.py",
           "horizons": rows}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT_PATH}")
    print("\nHOW TO READ THIS:")
    print("  PR-AUC is NOT comparable across horizons directly -- a longer horizon has a")
    print("  higher positive rate, and PR-AUC's own baseline IS the positive rate. The")
    print("  LIFT column (PR-AUC / positive rate) is the comparable number: how many times")
    print("  better than guessing the base rate the model does at that horizon.")


if __name__ == "__main__":
    main()
