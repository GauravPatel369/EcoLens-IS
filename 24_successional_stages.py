"""
EcoLens Step 24 -- SUCCESSIONAL STAGES AND RECOVERY  (C20)

WHAT C20 STILL WANTED
---------------------
C20 asks for disturbance pathways, fragmentation, **successional stages** and **recovery**.
The first two shipped (15_analog_trajectory_figure.py, and the fragmentation features in
13). Succession and recovery were recorded as blocked on "a regrowth product Hansen does
not provide".

That was half right, and the half that is wrong is worth having.

Hansen's `gain` layer (which we do not hold, and which only covers 2000-2012 anyway) is not
the only route. `lossyear` already encodes, for every disturbed pixel, WHEN it was
disturbed -- so for any observation year it gives **stand age since disturbance**, which IS
the standard field definition of successional stage. What Hansen cannot tell us is whether
regrowth actually happened; it can only tell us how long a stand has been undisturbed. So:

  * SUCCESSIONAL STAGE  -> derivable, and derived here.
  * RECOVERY (biomass / canopy return) -> genuinely NOT derivable from loss data.
    Reported as a limitation, with the *proxy* we can defend: the survival rate of
    disturbed pixels, i.e. how often a pixel that lost forest goes on to lose it again.

WHAT THIS COMPUTES
------------------
For every grid cell x observation year already in the risk table, using the same Hansen
tiles `10` reads:

  1. `years_since_disturbance` -- observation year minus the most recent loss year at or
     before it. Undisturbed pixels are censored, not zero-filled (censoring matters: a
     never-disturbed stand is not "age 0").
  2. Successional CLASS, on the standard forestry breaks:
        0-5 y   early / pioneer
        6-15 y  mid
        16+ y   late
        never   mature-undisturbed (censored)
  3. `reburn_rate` -- of pixels disturbed in year Y, the share disturbed AGAIN by 2021.
     This is the recovery proxy: high re-disturbance means the stand is not recovering into
     a stable state.

Then it tests the ecological question directly: **does successional stage predict future
loss?** If early-successional stands are disproportionately re-disturbed, that is a real
pathway finding and a usable risk feature.

Writes results/successional_stages.json

Run:
    python 24_successional_stages.py
"""

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from config import RISK_FEATURES_PATH, RESULTS_DIR

OUT_PATH = f"{RESULTS_DIR}/successional_stages.json"

# Standard forestry breaks for time since stand-replacing disturbance.
STAGE_BREAKS = [(0, 5, "early (0-5y)"), (6, 15, "mid (6-15y)"), (16, 99, "late (16+y)")]
HANSEN_EPOCH = 2000          # lossyear stores year-2000 (1 = 2001 ... 21 = 2021)
LAST_LOSS_YEAR = 2021


def stage_of(age):
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "undisturbed (censored)"
    for lo, hi, name in STAGE_BREAKS:
        if lo <= age <= hi:
            return name
    return "late (16+y)"


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_t10", "10_grid_tiling_labels.py")
    t10 = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["_t10"] = t10
    spec.loader.exec_module(t10)

    df = pd.read_csv(RISK_FEATURES_PATH)
    print(f"\n{'='*78}")
    print("SUCCESSIONAL STAGES AND RECOVERY (C20)")
    print(f"{'='*78}")
    print(f"  {len(df):,} cell-year rows, {df.groupby(['cell_lon','cell_lat']).ngroups:,} cells")

    # Disturbance history per cell, read once from the Hansen tiles 10 already caches.
    exp = importlib.util.spec_from_file_location("_t09", "09_explainability_engine.py")
    t09 = importlib.util.module_from_spec(exp)
    sys.modules["_t09"] = t09
    exp.loader.exec_module(t09)

    cells = df[["cell_lon", "cell_lat"]].drop_duplicates()
    print(f"  reading Hansen loss history for {len(cells):,} cells ...")

    hist = {}
    from tqdm import tqdm
    for lon, lat in tqdm(cells.itertuples(index=False), total=len(cells), desc="hansen"):
        try:
            h = t09.get_disturbance_history(lon, lat)
            hist[(round(lon, 5), round(lat, 5))] = sorted(h.get("loss_years") or []) if h else []
        except Exception:
            hist[(round(lon, 5), round(lat, 5))] = []

    n_with = sum(1 for v in hist.values() if v)
    print(f"  cells with any recorded loss: {n_with:,} / {len(hist):,} "
          f"({100*n_with/max(len(hist),1):.1f} %)")

    # ---- 1 & 2: stand age and successional class, per cell-year -------------------
    ages, stages = [], []
    for r in df.itertuples(index=False):
        ys = hist.get((round(r.cell_lon, 5), round(r.cell_lat, 5)), [])
        prior = [y for y in ys if y <= r.obs_year]
        if prior:
            age = r.obs_year - max(prior)
            ages.append(age)
            stages.append(stage_of(age))
        else:
            ages.append(np.nan)                 # CENSORED, not zero -- see docstring
            stages.append("undisturbed (censored)")
    df["years_since_disturbance"] = ages
    df["successional_stage"] = stages

    # ---- 3: does successional stage predict future loss? -------------------------
    print(f"\n  {'stage':<26}{'n rows':>10}{'future-loss rate':>20}")
    order = ["early (0-5y)", "mid (6-15y)", "late (16+y)", "undisturbed (censored)"]
    by_stage = {}
    base = float(df.label_loss_within_horizon.mean())
    for st in order:
        g = df[df.successional_stage == st]
        if len(g) == 0:
            continue
        rate = float(g.label_loss_within_horizon.mean())
        by_stage[st] = {"n_rows": int(len(g)), "future_loss_rate": rate,
                        "lift_vs_base": rate / base if base else None}
        print(f"  {st:<26}{len(g):>10,}{rate*100:>19.1f}%")
    print(f"  {'ALL':<26}{len(df):>10,}{base*100:>19.1f}%  <- base rate")

    # ---- recovery proxy: re-disturbance ------------------------------------------
    print(f"\n  RECOVERY PROXY -- of pixels disturbed in year Y, share disturbed AGAIN by {LAST_LOSS_YEAR}:")
    reburn = {}
    for y in range(2001, LAST_LOSS_YEAR):
        disturbed = [k for k, v in hist.items() if y in v]
        if len(disturbed) < 20:
            continue
        again = [k for k in disturbed if any(yy > y for yy in hist[k])]
        reburn[y] = {"n_cells_disturbed": len(disturbed),
                     "n_disturbed_again": len(again),
                     "rate": len(again) / len(disturbed)}
    for y, v in sorted(reburn.items()):
        print(f"    {y}  n={v['n_cells_disturbed']:>5}  re-disturbed {v['rate']*100:>5.1f} %")

    out = {
        "n_rows": int(len(df)),
        "n_cells": int(len(hist)),
        "cells_with_loss": int(n_with),
        "stage_breaks": [[lo, hi, name] for lo, hi, name in STAGE_BREAKS],
        "base_future_loss_rate": base,
        "by_stage": by_stage,
        "reburn_by_year": reburn,
        "limitation": ("Hansen records LOSS only. Stand age since disturbance is derivable "
                       "and is the standard definition of successional stage; actual "
                       "biomass/canopy RECOVERY is not derivable and is proxied here by "
                       "re-disturbance rate."),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
