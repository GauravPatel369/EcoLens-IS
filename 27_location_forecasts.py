"""
EcoLens Step 27 -- PER-LOCATION FOREST-LOSS FORECASTS FOR THE DASHBOARD

WHAT THIS ADDS
--------------
The retrieval dashboard shows what a place *is* and what it resembles. It never showed what
is likely to *happen* to it -- `grep -c "risk|forecast|predict" 08_retrieval_dashboard.py`
returned 0. Pillar B (the risk model) existed only as a CLI. This computes, per location,
the trained model's probability of tree-cover loss within RISK_HORIZON_YEARS, plus the
observed Hansen history behind it, so the dashboard shows the forecast *and* the record it
extrapolates from.

WHAT IT IS NOT
--------------
A forest-loss probability, not an "ecosystem convergence/divergence" score. Those are
different quantities and this model was never trained for the latter. Nothing here may be
presented as predicting how similar two ecosystems will become.

SCOPE -- DEFAULT IS FOREST, DELIBERATELY
----------------------------------------
The model is trained on RISK_FOREST_ECOSYSTEMS (= ["forest"]) with baseline tree cover above
HANSEN_TREECOVER_THRESHOLD (= 30 %). By default this scores exactly that: the 17 forest
locations. `--ecosystems all` scores everything, and anything outside the training set is
flagged `extrapolation` rather than quietly mixed in.

Two rules that must not be fudged:

  1. NO TREE COVER -> NO FORECAST. Deserts, tundra, grassland and urban sites have nothing
     for a tree-cover-loss model to predict. `compute_cell_label_and_features` returns None
     for them and that None is reported as "not applicable", never as "0 % risk". A
     confident 0 % on a sand dune is worse than a blank.
  2. WOODED BUT OUT-OF-DOMAIN -> FLAGGED. Mangrove/boreal/wetland pass the tree-cover gate
     but sit outside the training distribution; the number carries the flag into the UI.

DOWNLOADS ARE OPT-IN
--------------------
`ensure_hansen_tile()` downloads on a miss -- correct for the pipeline, wrong here. Scoring
all 126 global locations needs 9 tiles we do not hold (~2.7 GB). This script checks tile
presence first and skips with a reason unless `--allow-download` is passed.

Writes results/location_forecasts.json

Run:
    python 27_location_forecasts.py                      # forest only, no downloads
    python 27_location_forecasts.py --ecosystems all --allow-download
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

from config import (
    METADATA_CATALOG_PATH, RESULTS_DIR, RISK_MODEL_PATH, RISK_MODEL_DIR,
    RISK_HORIZON_YEARS, RISK_FOREST_ECOSYSTEMS, HANSEN_TREECOVER_THRESHOLD,
)

OUT_PATH = f"{RESULTS_DIR}/location_forecasts.json"


def _mod(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


def main():
    ap = argparse.ArgumentParser(description="Per-location forest-loss forecasts")
    ap.add_argument("--ecosystems", nargs="+", default=list(RISK_FOREST_ECOSYSTEMS),
                    help="ecosystems to score; default = exactly what the model was "
                         "trained on. 'all' scores every catalog location.")
    ap.add_argument("--allow-download", action="store_true",
                    help="permit fetching missing Hansen tiles (~300 MB each)")
    args = ap.parse_args()
    wanted = None if args.ecosystems == ["all"] else set(args.ecosystems)

    import joblib
    from tqdm import tqdm

    # Prefer the CALIBRATED model. The raw model is trained with class_weight="balanced",
    # which is right for ranking and inflates probabilities 2-3x in the mid-range -- it
    # reported 56 % where the true frequency is ~21 %. The dashboard shows these numbers to
    # a person, so it must use the calibrated one or it is simply lying to the viewer.
    calibrated_path = f"{RISK_MODEL_DIR}/forest_loss_risk_model_calibrated.joblib"
    if os.path.exists(calibrated_path):
        import importlib.util as _il, sys as _sys
        if "_cal28" not in _sys.modules:      # the wrapper class must be importable to unpickle
            _sp = _il.spec_from_file_location("_cal28", "28_calibrate_risk_model.py")
            _m = _il.module_from_spec(_sp); _sys.modules["_cal28"] = _m; _sp.loader.exec_module(_m)
            _sys.modules["__main__"].CalibratedRiskModel = _m.CalibratedRiskModel
        bundle = joblib.load(calibrated_path)
        is_calibrated = True
        print(f"  using CALIBRATED model ({calibrated_path})")
    elif os.path.exists(RISK_MODEL_PATH):
        bundle = joblib.load(RISK_MODEL_PATH)
        is_calibrated = False
        print("  WARNING: using the UNCALIBRATED model -- probabilities are inflated "
              "~2-3x in the mid-range. Run 28_calibrate_risk_model.py.")
    else:
        raise SystemExit(f"No trained model at {RISK_MODEL_PATH}. Run 11 first.")
    model, feature_names = bundle["model"], bundle["feature_names"]

    tiling = _mod("_t10", "10_grid_tiling_labels.py")
    explain = _mod("_t09", "09_explainability_engine.py")
    import geo_lookups

    with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    # One forecast per PLACE, not per sub-crop -- sub-crops share a coordinate to within a
    # few hundred metres and would all return the same number.
    locations, seen = [], set()
    for e in catalog:
        base = e.get("base_id", e["id"].rsplit("_p", 1)[0])
        if base in seen:
            continue
        seen.add(base)
        if wanted is not None and e["ecosystem"] not in wanted:
            continue
        locations.append({"base_id": base, "name": e["name"], "ecosystem": e["ecosystem"],
                          "lon": e["lon"], "lat": e["lat"]})

    print(f"\n{'='*78}")
    print(f"PER-LOCATION FOREST-LOSS FORECASTS  (horizon {RISK_HORIZON_YEARS} y)")
    print(f"{'='*78}")
    print(f"  scope: {'all ecosystems' if wanted is None else ', '.join(sorted(wanted))}")
    print(f"  {len(locations)} locations; model trained on {list(RISK_FOREST_ECOSYSTEMS)}, "
          f"tree cover >= {HANSEN_TREECOVER_THRESHOLD} %\n")

    year = tiling.OBS_YEAR_END
    out, n_ok, n_na, n_extrap = {}, 0, 0, 0

    for loc in tqdm(locations, desc="forecasting"):
        lon, lat = loc["lon"], loc["lat"]
        rec = dict(loc)
        rec.update({"horizon_years": RISK_HORIZON_YEARS, "obs_year": year, "risk": None,
                    "applicable": False, "extrapolation": False, "reason": None,
                    "history": None})

        if not args.allow_download:
            tile = tiling.hansen_tile_name(lat, lon)
            need = [os.path.join(tiling.HANSEN_DATA_DIR, f"Hansen_{l}_{tile}.tif")
                    for l in ("treecover2000", "lossyear")]
            if not all(os.path.exists(q) for q in need):
                rec["reason"] = (f"Hansen tile {tile} not held locally. Re-run with "
                                 f"--allow-download to fetch it (~300 MB).")
                out[loc["base_id"]] = rec
                n_na += 1
                continue

        try:
            tc = tiling.ensure_hansen_tile("treecover2000", lat, lon)
            ly = tiling.ensure_hansen_tile("lossyear", lat, lon)
            feats = tiling.compute_cell_label_and_features(
                lon, lat, cell_half_km=0.5, tiles=(tc, ly), obs_year=year)
        except Exception as ex:
            rec["reason"] = f"Hansen read failed ({type(ex).__name__})"
            out[loc["base_id"]] = rec
            n_na += 1
            continue

        if feats is None:
            # The honest branch: no forecast at all, not "0 % risk".
            rec["reason"] = (f"Not forested: baseline tree cover below "
                             f"{HANSEN_TREECOVER_THRESHOLD} %, or already >50 % cleared. "
                             f"A tree-cover-loss model has nothing to predict here.")
            out[loc["base_id"]] = rec
            n_na += 1
            continue

        geo = geo_lookups.get_physical_descriptors(lon, lat)
        row = {
            "baseline_treecover_pct": feats["baseline_treecover_pct"],
            "distance_to_prior_loss_m": feats["distance_to_prior_loss_m"],
            "protected_area": geo["protected_area"],
            "temp_c": geo["temp_c"], "rainfall_mm": geo["rainfall_mm"],
            "elevation_m": geo["elevation_m"], "ruggedness_m": geo["ruggedness_m"],
            "obs_year": year,
        }
        X = np.full((1, len(feature_names)), np.nan)
        for j, fname in enumerate(feature_names):
            v = row.get(fname)
            if v is None:
                continue                     # left NaN; the model handles missing natively
            X[0, j] = float(bool(v)) if fname == "protected_area" else float(v)

        rec["risk"] = float(model.predict_proba(X)[0, 1])
        rec["applicable"] = True
        rec["drivers"] = {k: (None if v is None else
                              (bool(v) if k == "protected_area" else float(v)))
                          for k, v in row.items()}
        if loc["ecosystem"] not in RISK_FOREST_ECOSYSTEMS:
            rec["extrapolation"] = True
            rec["reason"] = (f"'{loc['ecosystem']}' is outside the training distribution "
                             f"({', '.join(RISK_FOREST_ECOSYSTEMS)}). Wooded enough to "
                             f"score, but treat as extrapolation.")
            n_extrap += 1
        n_ok += 1

        # OBSERVED history = strictly what was knowable AT obs_year.
        #
        # Hansen runs to 2023 while obs_year is capped at 2023 - horizon = 2021, so the raw
        # loss-year list contains 2022 and 2023 -- precisely the window the model is being
        # asked to predict. Showing those next to the prediction displays the answer beside
        # the question, and produced the visible tell of "last -2y ago" (2021 - 2023).
        # Anything after obs_year is future information and is split out, never mixed in.
        try:
            h = explain.get_disturbance_history(lon, lat)
            all_years = sorted(h.get("loss_years") or []) if h else []
            past = [y for y in all_years if y <= year]
            future = [y for y in all_years if y > year]
            rec["history"] = {
                "loss_years": past,
                "n_loss_years": len(past),
                "last_loss_year": past[-1] if past else None,
                "years_since_last_loss": (year - past[-1]) if past else None,
                # kept for auditing only; the UI must not show it as "observed"
                "_loss_years_after_obs_year": future,
            }
        except Exception:
            pass

        out[loc["base_id"]] = rec

    risks = [r["risk"] for r in out.values() if r["risk"] is not None]
    summary = {
        "horizon_years": RISK_HORIZON_YEARS, "obs_year": year,
        "scope": "all" if wanted is None else sorted(wanted),
        "n_locations": len(locations), "n_with_forecast": n_ok,
        "n_not_applicable": n_na, "n_extrapolation": n_extrap,
        "trained_on_ecosystems": list(RISK_FOREST_ECOSYSTEMS),
        "treecover_threshold_pct": HANSEN_TREECOVER_THRESHOLD,
        "risk_median": float(np.median(risks)) if risks else None,
        "risk_min": float(np.min(risks)) if risks else None,
        "risk_max": float(np.max(risks)) if risks else None,
        "calibrated": is_calibrated,
        "caveat": ("Probability of tree-cover loss within the horizon. NOT an ecosystem "
                   "convergence/divergence score -- the model was never trained for that."),
    }

    print(f"\n  forecast produced : {n_ok}   (extrapolation: {n_extrap})")
    print(f"  not applicable    : {n_na}")
    if risks:
        print(f"  risk range        : {min(risks)*100:.1f} % - {max(risks)*100:.1f} %, "
              f"median {np.median(risks)*100:.1f} %")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "locations": out}, f, indent=2)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
