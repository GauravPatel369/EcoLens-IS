"""
EcoLens Step 29 -- PER-REGION RISK GRIDS FOR THE DASHBOARD MAP

WHY
---
27_location_forecasts.py scores ONE cell per location -- the 1 km square at the site's
centre point. That is a sample, not the forest, and a single number cannot show the thing
that actually matters: whether risk is spread evenly across a region or concentrated along
one logging front.

The model was always capable of more. `10` tiles a 15 km radius around every forest region
into ~500-700 cells and every one of them already carries its driver features. This scores
all of them so the dashboard can draw the region instead of describing it.

WHAT IT PRODUCES
----------------
For each forest region: one row per 1 km cell with (lat, lon, calibrated risk), plus a
summary the UI can state in words -- how many cells, how many above the high-risk cut-off,
and the expected NUMBER of cells that will see loss (the sum of probabilities, which is the
expected count of a sum of Bernoulli trials). A count is what a forest manager can act on;
a mean probability is not.

HONESTY CONSTRAINTS BUILT IN
----------------------------
* Uses the CALIBRATED model. Raw probabilities run 2-3x high in the mid-range, and a map
  shaded by them would be uniformly alarming.
* `in_sample: true` is recorded in the payload and shown in the UI. The model was trained
  on these same 17 regions, so this map illustrates what the model believes -- it is not
  evidence of accuracy. The out-of-sample evidence is the leave-one-region-out result and
  the 2021 holdout, and the map must not be mistaken for either.
* Cells whose features are missing are emitted with risk = null and drawn grey, never as
  low risk.

Writes outputs/results/region_risk_grids.json

Run:
    python 29_region_risk_map.py
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

from config import (RISK_FEATURES_PATH, RISK_MODEL_DIR, RESULTS_DIR,
                    RISK_HORIZON_YEARS, METADATA_CATALOG_PATH)

OUT_PATH = f"{RESULTS_DIR}/region_risk_grids.json"
HIGH_RISK = 0.50          # the cut-off the summary line counts against
LABEL = "label_loss_within_horizon"


def main():
    import joblib

    # the calibrated wrapper class must be importable for joblib to unpickle it
    spec = importlib.util.spec_from_file_location("_c28", "28_calibrate_risk_model.py")
    c28 = importlib.util.module_from_spec(spec)
    sys.modules["_c28"] = c28
    spec.loader.exec_module(c28)
    sys.modules["__main__"].CalibratedRiskModel = c28.CalibratedRiskModel

    cal_path = f"{RISK_MODEL_DIR}/forest_loss_risk_model_calibrated.joblib"
    if not os.path.exists(cal_path):
        raise SystemExit("No calibrated model. Run 28_calibrate_risk_model.py first -- an "
                         "uncalibrated map would be uniformly alarming and wrong.")
    bundle = joblib.load(cal_path)
    model, feats = bundle["model"], bundle["feature_names"]

    df = pd.read_csv(RISK_FEATURES_PATH)
    year = int(df.obs_year.max())
    cur = df[df.obs_year == year].copy()

    print(f"\n{'='*78}")
    print(f"PER-REGION RISK GRIDS  (obs_year {year}, horizon {RISK_HORIZON_YEARS} y)")
    print(f"{'='*78}")
    print(f"  {len(cur):,} cells across {cur.region_id.nunique()} regions, calibrated model\n")

    X = c28.build_X(cur, feats)
    cur["risk"] = model.predict_proba(X)[:, 1]

    # map region_id -> a human name from the catalog
    names = {}
    if os.path.exists(METADATA_CATALOG_PATH):
        for e in json.load(open(METADATA_CATALOG_PATH, encoding="utf-8")):
            b = e.get("base_id", e["id"].rsplit("_p", 1)[0])
            names.setdefault(b, e.get("name", b))

    regions = {}
    print(f"  {'region':<13}{'cells':>7}{'>=50%':>8}{'mean':>8}{'expected':>10}  name")
    for rid, g in cur.groupby("region_id"):
        r = g["risk"].to_numpy()
        expected = float(np.nansum(r))          # E[# cells with loss] = sum of probabilities
        n_high = int((r >= HIGH_RISK).sum())
        regions[rid] = {
            "region_id": rid,
            "name": names.get(rid, rid),
            "n_cells": int(len(g)),
            "n_high_risk": n_high,
            "high_risk_cutoff": HIGH_RISK,
            "mean_risk": float(np.nanmean(r)),
            "expected_cells_with_loss": expected,
            "center": [float(g.cell_lat.mean()), float(g.cell_lon.mean())],
            # compact triples: [lat, lon, risk]; rounded because the UI cannot show more
            "cells": [[round(float(a), 4), round(float(b), 4),
                       (None if np.isnan(c) else round(float(c), 3))]
                      for a, b, c in zip(g.cell_lat, g.cell_lon, g.risk)],
        }
        print(f"  {rid:<13}{len(g):>7}{n_high:>8}{np.nanmean(r)*100:>7.1f}%"
              f"{expected:>10.0f}  {names.get(rid, '')[:34]}")

    out = {
        "meta": {
            "obs_year": year,
            "horizon_years": RISK_HORIZON_YEARS,
            "cell_size_m": 1000,
            "calibrated": True,
            "high_risk_cutoff": HIGH_RISK,
            # Stated in the payload so the UI cannot render this map without the caveat.
            "in_sample": True,
            "in_sample_note": (
                "The model was trained on these same regions, so this map shows what the "
                "model believes about places it has seen. It is an illustration, not a "
                "measure of accuracy. Out-of-sample evidence: leave-one-region-out "
                "(+11.0 % from analogs) and the 2021 holdout (ROC-AUC 0.90)."),
            "reading": (
                "Each square is 1 km. Its shade is the probability that ANY tree-cover "
                "pixel inside it is lost within the horizon -- not how much forest is "
                "lost. Where loss does occur, typical canopy lost is a few percent."),
        },
        "regions": regions,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"\n  saved {OUT_PATH}  ({os.path.getsize(OUT_PATH)/2**20:.1f} MB)")


if __name__ == "__main__":
    main()
