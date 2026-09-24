"""
EcoLens Step 28 -- PROBABILITY CALIBRATION FOR THE RISK MODEL

THE PROBLEM
-----------
Every candidate model in `11` is trained with `class_weight="balanced"` (or
`scale_pos_weight`). Forest loss is rare -- ~25 % of cell-years -- and without that
re-weighting a model learns the lazy answer "say no to everything". The re-weighting is
correct and it is why ranking metrics are good.

It also deliberately distorts the OUTPUT PROBABILITIES upward, and `11` selects on PR-AUC,
a pure ranking metric, so nothing in the pipeline ever noticed. Measured on the held-out
split before this script existed:

    observed positive rate 21.7 %   MEAN PREDICTED 37.8 %

    predicted 55 %  ->  actually happened 21 %
    predicted 65 %  ->  actually happened 30 %
    predicted 75 %  ->  actually happened 41 %

Over-prediction by 2-3x through the mid-range. Harmless for "rank these places by danger";
badly wrong the moment a number is shown to a human as a probability, which is exactly what
the dashboard does.

WHAT THIS FIXES, AND WHAT IT CANNOT
-----------------------------------
Isotonic regression learns a monotonic map from raw score -> true frequency. Because it is
monotonic it CANNOT change the ranking, so every ranking-based result in this project
(PR-AUC, the +11.0 % analog finding, the ablations, the sweeps) is untouched by design.
It only makes the printed number mean what it says.

THE SPLIT -- THE PART THAT MUST BE LEAK-FREE
--------------------------------------------
Three disjoint time periods, because calibrating on data the model trained on would learn
the model's memorised answers rather than its real error, and validating on the data the
CALIBRATOR saw would report a fit to itself:

    train    obs_year <= 2018   the base model was fit here (by 11), untouched below
    calib    2019 - 2020        the isotonic map is fit here
    holdout  2021               used ONLY to report whether calibration worked

Nothing that follows ever fits on the holdout. The reported calibration table is therefore
an honest out-of-sample measurement, not a self-check.

Writes outputs/risk_model/forest_loss_risk_model_calibrated.joblib
       outputs/results/risk_calibration.json

Run:
    python 28_calibrate_risk_model.py
"""

import json
import os

import numpy as np
import pandas as pd

from config import RISK_FEATURES_PATH, RISK_MODEL_PATH, RISK_MODEL_DIR, RESULTS_DIR

OUT_MODEL = f"{RISK_MODEL_DIR}/forest_loss_risk_model_calibrated.joblib"
OUT_JSON = f"{RESULTS_DIR}/risk_calibration.json"

TRAIN_MAX = 2018          # the base model in 11 was fit on obs_year <= this
CALIB_YEARS = (2019, 2020)
HOLDOUT_YEAR = 2021
LABEL = "label_loss_within_horizon"


class CalibratedRiskModel:
    """Base model + an isotonic score->frequency map, in one picklable object.

    Written explicitly rather than via CalibratedClassifierCV(cv="prefit") because that
    argument was removed in this scikit-learn version. Doing it by hand is also clearer:
    isotonic regression on (raw_score, outcome) IS what that wrapper fits internally, and
    having it visible makes the monotonicity guarantee -- and therefore the claim that
    ranking is unchanged -- checkable rather than asserted.
    """

    def __init__(self, base, iso, feature_names):
        self.base, self.iso, self.feature_names = base, iso, feature_names
        self.calibrated = True

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        p = np.clip(self.iso.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def build_X(df, feature_names):
    """Same construction as 11.build_matrix: missing stays NaN, never imputed."""
    X = np.full((len(df), len(feature_names)), np.nan, dtype=np.float64)
    for j, f in enumerate(feature_names):
        if f not in df.columns:
            continue
        col = df[f]
        if f == "protected_area":
            X[:, j] = col.map(lambda v: np.nan if pd.isna(v) else float(bool(v))).to_numpy()
        else:
            X[:, j] = pd.to_numeric(col, errors="coerce").to_numpy()
    return X


def calibration_table(y, p, bins=None):
    bins = bins or [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5),
                    (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]
    rows = []
    for lo, hi in bins:
        m = (p >= lo) & (p < hi)
        if m.sum() < 25:
            continue
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                     "mean_predicted": float(p[m].mean()),
                     "observed": float(y[m].mean()),
                     "gap": float(p[m].mean() - y[m].mean())})
    return rows


def show(title, y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    print(f"\n  {title}")
    print(f"    observed {y.mean()*100:.1f} %   mean predicted {p.mean()*100:.1f} %   "
          f"Brier {brier_score_loss(y, p):.4f}   PR-AUC {average_precision_score(y, p):.4f}   "
          f"ROC-AUC {roc_auc_score(y, p):.4f}")
    print(f"    {'bin':>10}{'n':>8}{'predicted':>11}{'ACTUAL':>9}{'gap':>9}")
    rows = calibration_table(y, p)
    for r in rows:
        print(f"    {r['bin']:>10}{r['n']:>8}{r['mean_predicted']*100:>10.1f}%"
              f"{r['observed']*100:>8.1f}%{r['gap']*100:>+8.1f}")
    return rows


def main():
    from sklearn.isotonic import IsotonicRegression
    import joblib

    if not os.path.exists(RISK_MODEL_PATH):
        raise SystemExit(f"No base model at {RISK_MODEL_PATH}. Run 11 first.")
    bundle = joblib.load(RISK_MODEL_PATH)
    model, feats = bundle["model"], bundle["feature_names"]

    df = pd.read_csv(RISK_FEATURES_PATH)
    calib = df[df.obs_year.isin(CALIB_YEARS)]
    hold = df[df.obs_year == HOLDOUT_YEAR]

    print(f"\n{'='*78}")
    print("RISK-MODEL PROBABILITY CALIBRATION")
    print(f"{'='*78}")
    print(f"  base model : {type(model).__name__}, fit by 11 on obs_year <= {TRAIN_MAX}")
    print(f"  calibrate  : obs_year {CALIB_YEARS[0]}-{CALIB_YEARS[1]}  ({len(calib):,} rows)")
    print(f"  holdout    : obs_year {HOLDOUT_YEAR}              ({len(hold):,} rows) "
          f"-- never fit on, used only to report")

    if len(calib) < 500 or len(hold) < 500:
        raise SystemExit("Not enough rows in the calibration/holdout periods.")

    Xc, yc = build_X(calib, feats), calib[LABEL].to_numpy()
    Xh, yh = build_X(hold, feats), hold[LABEL].to_numpy()

    before = show("BEFORE -- raw model on the holdout year", yh, model.predict_proba(Xh)[:, 1])

    # Fit the map on the CALIBRATION years only; the base model is never refit.
    raw_c = model.predict_proba(Xc)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_c, yc)
    cal = CalibratedRiskModel(model, iso, feats)
    p_after = cal.predict_proba(Xh)[:, 1]
    after = show("AFTER  -- isotonic calibration, same holdout", yh, p_after)

    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    p_before = model.predict_proba(Xh)[:, 1]
    a_before, a_after = roc_auc_score(yh, p_before), roc_auc_score(yh, p_after)
    # Isotonic is WEAKLY monotonic: it maps ranges of raw scores onto a single calibrated
    # value, so distinct scores can become ties. AUC scores ties at 0.5, which moves it a
    # little. Claiming "ranking cannot change" would be tidier and would be wrong -- the
    # honest statement is that order is never reversed, but it can be merged.
    print(f"\nROC-AUC {a_before:.6f} -> {a_after:.6f}   (delta {a_after - a_before:+.6f})")
    print("  ^ isotonic never REVERSES two scores, but it maps ranges onto one value,")
    print("    creating ties that shift AUC by ~1e-3. Every ranking-based result in this")
    print("    project is computed from the UNCALIBRATED model and is untouched --")
    print("    calibration is applied only where a number is shown to a human.")

    mae_b = np.mean([abs(r["gap"]) for r in before])
    mae_a = np.mean([abs(r["gap"]) for r in after])
    print(f"\n  mean |predicted - actual| across bins:  {mae_b*100:.1f} %  ->  {mae_a*100:.1f} %")

    joblib.dump({"model": cal, "feature_names": feats, "calibrated": True,
                 "base_model": type(model).__name__,
                 "calibration": {"method": "isotonic", "train_max_year": TRAIN_MAX,
                                 "calib_years": list(CALIB_YEARS),
                                 "holdout_year": HOLDOUT_YEAR}}, OUT_MODEL)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"method": "isotonic", "train_max_year": TRAIN_MAX,
                   "calib_years": list(CALIB_YEARS), "holdout_year": HOLDOUT_YEAR,
                   "n_calib": int(len(calib)), "n_holdout": int(len(hold)),
                   "holdout_observed_rate": float(yh.mean()),
                   "brier_before": float(brier_score_loss(yh, p_before)),
                   "brier_after": float(brier_score_loss(yh, p_after)),
                   "roc_auc_before": float(roc_auc_score(yh, p_before)),
                   "roc_auc_after": float(roc_auc_score(yh, p_after)),
                   "mean_abs_gap_before": float(mae_b), "mean_abs_gap_after": float(mae_a),
                   "bins_before": before, "bins_after": after}, f, indent=2)
    print(f"\n  saved {OUT_MODEL}")
    print(f"  saved {OUT_JSON}")


if __name__ == "__main__":
    main()
