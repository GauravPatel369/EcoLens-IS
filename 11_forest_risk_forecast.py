"""
EcoLens Objective 4 - Step 11: Forest-Loss Risk Forecasting

Trains a classifier that predicts P(tree-cover loss within
config.RISK_HORIZON_YEARS) for a forest/mangrove grid cell, using the
cell-year table produced by 10_grid_tiling_labels.py.

---------------------------------------------------------------------
WHAT THIS IS AND ISN'T
---------------------------------------------------------------------
This is forest-LOSS-RISK forecasting -- a probability estimate,
validated against real historical Hansen Global Forest Change loss
records -- NOT a claim to "predict the future" of an ecosystem in any
stronger sense. Loss labels come from Hansen's tree-cover-loss layer,
which does not distinguish clear-cutting from fire from storm damage;
read "loss" as "any tree-cover loss", not specifically deforestation,
unless you've cross-referenced a fire dataset (see README.md).

---------------------------------------------------------------------
VALIDATION METHODOLOGY -- temporal holdout, not random split
---------------------------------------------------------------------
Cell-year samples are split by OBSERVATION YEAR: the model trains
only on samples with obs_year <= TRAIN_CUTOFF_YEAR and is evaluated
only on samples with obs_year > TRAIN_CUTOFF_YEAR. A random split
would let the model implicitly learn from "future" cells that are
geographically near "past" cells in the same training fold, which
overstates real forecasting skill. This also means: don't lower
TRAIN_CUTOFF_YEAR just to get more training data if it leaves too few
years for a meaningful test set.

Because loss events are rare, this script reports Average Precision
(area under the precision-recall curve) as the primary metric, not
accuracy or plain ROC-AUC -- with a realistic loss rate, a classifier
that always predicts "no loss" gets high accuracy and a deceptively
decent-looking ROC-AUC while being useless.

---------------------------------------------------------------------
ABLATION: driver features vs. driver features + embedding drift
---------------------------------------------------------------------
The research question this project set out to ask is whether
foundation-model embeddings add predictive power beyond cheap driver
features (baseline tree cover, distance to existing loss edges,
protection status, climate). This script runs that ablation IF a
"embedding_drift" column is present in the features CSV -- it does
NOT fabricate one. Wiring up a real embedding_drift feature means
re-running the Sentinel-2 acquisition + Prithvi embedding pipeline at
the grid-cell level across multiple years and computing year-over-year
cosine distance per cell, which is a real (and fairly heavy) extension
beyond 10_grid_tiling_labels.py's current scope -- see README.md's
"Next steps" section. Running this script without that column simply
reports the driver-only baseline and says so explicitly.

Run:
    python 11_forest_risk_forecast.py                    # train + evaluate
    python 11_forest_risk_forecast.py --predict LON LAT   # score a single point
"""

import argparse
import csv
import os

import numpy as np

from config import RISK_FEATURES_PATH, RISK_MODEL_PATH, RISK_HORIZON_YEARS, RISK_MODEL_DIR

DRIVER_FEATURES = [
    "baseline_treecover_pct",
    "distance_to_prior_loss_m",
    "protected_area",
    "temp_c",
    "rainfall_mm",
    "elevation_m",
    "ruggedness_m",
    "obs_year",
]
OPTIONAL_FEATURES = ["embedding_drift"]
TARGET_COLUMN = "label_loss_within_horizon"


def load_features(path=RISK_FEATURES_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run 10_grid_tiling_labels.py first to generate "
            f"the cell-year feature table."
        )
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _to_float_or_nan(value):
    if value is None or value == "" or value == "None":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _to_bool_float(value):
    if value in (None, "", "None"):
        return np.nan
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in ("true", "1", "yes") else 0.0
    return float(bool(value))


def build_matrix(rows, feature_names):
    """
    Convert the CSV rows into (X, y, groups) arrays. groups holds
    obs_year per row, used for the temporal split.

    Missing values become np.nan -- HistGradientBoostingClassifier
    handles NaN natively (it learns which branch to send missing
    values down), so we do NOT impute a guessed value here. A missing
    climate/elevation reading (reference dataset not downloaded, see
    geo_lookups.py) stays missing all the way into the model.
    """
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.int32)
    obs_years = np.zeros(len(rows), dtype=np.int32)

    for i, row in enumerate(rows):
        for j, feat in enumerate(feature_names):
            if feat == "protected_area":
                X[i, j] = _to_bool_float(row.get(feat))
            else:
                X[i, j] = _to_float_or_nan(row.get(feat))
        y[i] = int(row[TARGET_COLUMN])
        obs_years[i] = int(row["obs_year"])

    return X, y, obs_years


def temporal_train_test_split(X, y, obs_years, train_cutoff_year):
    train_mask = obs_years <= train_cutoff_year
    test_mask = ~train_mask
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask], train_mask.sum(), test_mask.sum()


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names, label):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
    from sklearn.inspection import permutation_importance

    n_pos_train = int(y_train.sum())
    n_pos_test = int(y_test.sum())
    print(f"\n{'='*70}")
    print(f"MODEL: {label}")
    print(f"{'='*70}")
    print(f"  Features: {feature_names}")
    print(f"  Train: {len(y_train)} samples ({n_pos_train} positive, "
          f"{100*n_pos_train/max(len(y_train),1):.1f}%)")
    print(f"  Test:  {len(y_test)} samples ({n_pos_test} positive, "
          f"{100*n_pos_test/max(len(y_test),1):.1f}%)")

    if n_pos_train < 10 or n_pos_test < 5:
        print(f"\n  WARNING: too few positive examples to train/evaluate reliably.")
        print(f"  See README.md -> 'Sample size' -- widen GRID_REGION_BUFFER_KM,")
        print(f"  add more forest/mangrove base locations, or widen the")
        print(f"  observation-year range in 10_grid_tiling_labels.py, then rerun.")
        return None

    # Filter out feature columns that have fewer than 2 distinct non-NaN values in X_train
    # (e.g. constant columns or all-NaN columns when optional reference data is missing).
    valid_indices = []
    skipped = []
    for j, feat in enumerate(feature_names):
        col_train = X_train[:, j]
        non_nan = col_train[~np.isnan(col_train)]
        if len(np.unique(non_nan)) >= 2:
            valid_indices.append(j)
        else:
            skipped.append(feat)

    if skipped:
        print(f"  Skipped constant/uninformative feature(s) (<2 distinct values in train set): {skipped}")

    if not valid_indices:
        print("  WARNING: No feature has enough distinct values to train a model.")
        return None

    active_feature_names = [feature_names[i] for i in valid_indices]
    X_train_active = X_train[:, valid_indices]
    X_test_active = X_test[:, valid_indices]

    model = HistGradientBoostingClassifier(
        class_weight="balanced",  # loss events are rare; don't let the
                                    # majority (no-loss) class dominate
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train_active, y_train)

    y_scores = model.predict_proba(X_test_active)[:, 1]
    ap = average_precision_score(y_test, y_scores)
    try:
        roc_auc = roc_auc_score(y_test, y_scores)
    except ValueError:
        roc_auc = float("nan")  # only one class present in y_test

    print(f"\n  Average Precision (PR-AUC): {ap:.4f}  <- primary metric, class-imbalance-aware")
    print(f"  ROC-AUC:                    {roc_auc:.4f}  <- secondary, can look inflated under imbalance")

    # Precision/recall at a couple of operating points, since a single
    # AUC number doesn't tell you what threshold to actually use.
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    for target_recall in [0.3, 0.5, 0.7]:
        idx = np.argmin(np.abs(recall[:-1] - target_recall)) if len(recall) > 1 else None
        if idx is not None and idx < len(thresholds):
            print(f"    at recall~{target_recall:.1f}: precision={precision[idx]:.3f}, "
                  f"threshold={thresholds[idx]:.3f}")

    # Permutation importance (HistGradientBoostingClassifier has no
    # built-in feature_importances_, unlike tree-bagging models).
    try:
        result = permutation_importance(model, X_test_active, y_test, n_repeats=10,
                                         random_state=42, scoring="average_precision")
        print(f"\n  Feature importance (permutation, drop in PR-AUC when shuffled):")
        order = np.argsort(result.importances_mean)[::-1]
        for idx in order:
            print(f"    {active_feature_names[idx]:<28} {result.importances_mean[idx]:+.4f} "
                  f"(+/- {result.importances_std[idx]:.4f})")
    except Exception as e:
        print(f"  (permutation importance skipped: {e})")

    return {"model": model, "ap": ap, "roc_auc": roc_auc, "feature_names": active_feature_names}


def predict_risk(lon, lat, model_path=RISK_MODEL_PATH):
    """
    Score a single (lon, lat) point with the trained model -- this is
    the "capable tool" side of the risk forecast: given a query
    location, fetch its current driver features and report an
    estimated loss-risk probability.
    """
    import joblib
    import importlib.util

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model at {model_path}. Run this script without --predict first.")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # Reuse 10_grid_tiling_labels.py's Hansen tile + feature helpers via
    # dynamic import, since numbered scripts in this project aren't
    # meant to be imported with a normal `import 10_...` statement.
    spec = importlib.util.spec_from_file_location("tiling", os.path.join(os.path.dirname(__file__), "10_grid_tiling_labels.py"))
    tiling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tiling)
    import geo_lookups

    current_year = tiling.OBS_YEAR_END
    tc_path = tiling.ensure_hansen_tile("treecover2000", lat, lon)
    lossyear_path = tiling.ensure_hansen_tile("lossyear", lat, lon)

    result = tiling.compute_cell_label_and_features(
        lon, lat, cell_half_km=0.5, tiles=(tc_path, lossyear_path), obs_year=current_year
    )
    if result is None:
        print(f"Could not compute Hansen-derived features for ({lon}, {lat}) at year "
              f"{current_year} -- either Hansen tiles are unavailable, or this cell "
              f"isn't forested (baseline tree cover below threshold).")
        return None

    geo = geo_lookups.get_physical_descriptors(lon, lat)
    row = {
        "baseline_treecover_pct": result["baseline_treecover_pct"],
        "distance_to_prior_loss_m": result["distance_to_prior_loss_m"],
        "protected_area": geo["protected_area"],
        "temp_c": geo["temp_c"],
        "rainfall_mm": geo["rainfall_mm"],
        "elevation_m": geo["elevation_m"],
        "ruggedness_m": geo["ruggedness_m"],
        "obs_year": current_year,
    }

    X = np.zeros((1, len(feature_names)))
    for j, feat in enumerate(feature_names):
        if feat not in row:
            X[0, j] = np.nan  # e.g. embedding_drift, not computed here
            continue
        X[0, j] = _to_bool_float(row[feat]) if feat == "protected_area" else _to_float_or_nan(row[feat])

    risk = model.predict_proba(X)[0, 1]
    print(f"\nEstimated risk of tree-cover loss within {RISK_HORIZON_YEARS} years "
          f"at ({lon}, {lat}): {risk*100:.1f}%")
    print(f"  (baseline tree cover: {result['baseline_treecover_pct']:.1f}%, "
          f"distance to nearest existing loss: "
          f"{result['distance_to_prior_loss_m']:.0f}m)" if result["distance_to_prior_loss_m"] is not None
          else "  (no existing loss detected nearby)")
    return risk


def run_spatial_holdout(rows, feature_names=DRIVER_FEATURES):
    """
    Perform a leave-one-location-out spatial holdout validation.
    For each unique region_id, trains a model on all other regions
    and evaluates on the held-out region.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    regions = sorted(list(set(row["region_id"] for row in rows)))

    print(f"\n{'='*70}")
    print("SPATIAL HOLDOUT (leave-one-location-out)")
    print(f"{'='*70}")

    y_all = np.array([int(r[TARGET_COLUMN]) for r in rows], dtype=np.int32)
    base_pos_rate = float(y_all.mean())

    print(f"  {len(regions)} regions, {len(rows)} total samples, base positive rate: {base_pos_rate:.4f}")
    print("  Training a fresh model per fold -- this takes a while (~15x a single training run).\n")

    results = []

    for held_out in regions:
        train_rows = [r for r in rows if r["region_id"] != held_out]
        test_rows = [r for r in rows if r["region_id"] == held_out]

        X_train, y_train, _ = build_matrix(train_rows, feature_names)
        X_test, y_test, _ = build_matrix(test_rows, feature_names)

        valid_indices = []
        for j, feat in enumerate(feature_names):
            col_train = X_train[:, j]
            non_nan = col_train[~np.isnan(col_train)]
            if len(np.unique(non_nan)) >= 2:
                valid_indices.append(j)

        X_train_active = X_train[:, valid_indices]
        X_test_active = X_test[:, valid_indices]

        model = HistGradientBoostingClassifier(random_state=42, max_iter=100)
        model.fit(X_train_active, y_train)

        y_scores = model.predict_proba(X_test_active)[:, 1]

        ap = average_precision_score(y_test, y_scores)
        try:
            roc_auc = roc_auc_score(y_test, y_scores)
        except ValueError:
            roc_auc = np.nan

        pos_test = int(y_test.sum())
        print(f"  [held out: {held_out}] AP={ap:.4f}  ROC-AUC={roc_auc:.4f}  "
              f"(train={len(y_train)}, test={len(y_test)}, pos_test={pos_test})")

        results.append({
            "region": held_out,
            "ap": ap,
            "roc_auc": roc_auc,
            "n": len(y_test),
            "pos": pos_test
        })

    print(f"\n{'='*70}")
    print("SPATIAL HOLDOUT SUMMARY")
    print(f"{'='*70}")
    print(f"  Folds completed: {len(regions)}/{len(regions)}")
    mean_ap = np.mean([r["ap"] for r in results])
    std_ap = np.std([r["ap"] for r in results])
    print(f"  Mean PR-AUC across held-out regions: {mean_ap:.4f}  (+/- {std_ap:.4f} std)")
    print(f"  Base positive rate (no-skill baseline): {base_pos_rate:.4f}\n")

    print("  Per-region results, worst to best:")
    sorted_results = sorted(results, key=lambda x: x["ap"])
    for r in sorted_results:
        print(f"    {r['region']:<20} AP={r['ap']:.4f}  ROC-AUC={r['roc_auc']:.4f}  (n={r['n']}, pos={r['pos']})")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print(f"  Mean spatial-holdout PR-AUC ({mean_ap:.4f}) is {'meaningfully above' if mean_ap > base_pos_rate + 0.02 else 'close to'} the base")
    print(f"  rate ({base_pos_rate:.4f}) -- the model shows {'real' if mean_ap > base_pos_rate + 0.02 else 'limited'} signal even on locations")
    print("  it never trained on. Still worth checking the per-region breakdown above:")
    print("  a few strong locations can pull the mean up while most others sit near")
    print("  baseline, which the mean alone won't show you.")


def main():
    parser = argparse.ArgumentParser(description="EcoLens Step 11: Forest-Loss Risk Forecasting")
    parser.add_argument("--predict", nargs=2, type=float, metavar=("LON", "LAT"),
                         help="Score a single point instead of training.")
    parser.add_argument("--spatial-holdout", action="store_true",
                         help="Run leave-one-location-out spatial holdout validation.")
    parser.add_argument("--train-cutoff-year", type=int, default=None,
                         help="Last obs_year included in training (default: computed as an "
                              "80/20-ish split over the available years).")
    args = parser.parse_args()

    if args.predict:
        lon, lat = args.predict
        predict_risk(lon, lat)
        return

    if args.spatial_holdout:
        rows = load_features()
        run_spatial_holdout(rows)
        return

    print(f"\n{'='*70}")
    print("EcoLens Step 11: Forest-Loss Risk Forecasting -- training")
    print(f"{'='*70}")

    rows = load_features()
    print(f"Loaded {len(rows)} cell-year samples from {RISK_FEATURES_PATH}")

    has_embedding_drift = len(rows) > 0 and "embedding_drift" in rows[0]
    if has_embedding_drift:
        print("embedding_drift column found -- running the driver-only vs. "
              "driver+embedding-drift ablation.")
    else:
        print("No embedding_drift column found in the features CSV -- reporting "
              "the driver-only baseline only. See this script's module docstring "
              "for what's needed to add that ablation.")

    all_years = sorted(set(int(r["obs_year"]) for r in rows))
    if args.train_cutoff_year is not None:
        cutoff = args.train_cutoff_year
    else:
        cutoff = all_years[int(len(all_years) * 0.8)] if len(all_years) > 1 else all_years[0]
    print(f"Years present: {all_years[0]}-{all_years[-1]}. Temporal train/test cutoff: {cutoff} "
          f"(train: obs_year <= {cutoff}, test: obs_year > {cutoff})")

    # --- Driver-only model ---
    X, y, obs_years = build_matrix(rows, DRIVER_FEATURES)
    X_train, y_train, X_test, y_test, n_train, n_test = temporal_train_test_split(X, y, obs_years, cutoff)
    driver_result = train_and_evaluate(X_train, y_train, X_test, y_test, DRIVER_FEATURES, "Driver features only")

    best_result = driver_result
    best_features = driver_result["feature_names"] if driver_result else None

    # --- Driver + embedding-drift model (only if the column is real) ---
    if has_embedding_drift and driver_result is not None:
        feats_with_drift = DRIVER_FEATURES + ["embedding_drift"]
        X2, y2, obs_years2 = build_matrix(rows, feats_with_drift)
        X2_train, y2_train, X2_test, y2_test, _, _ = temporal_train_test_split(X2, y2, obs_years2, cutoff)
        drift_result = train_and_evaluate(X2_train, y2_train, X2_test, y2_test, feats_with_drift,
                                           "Driver features + embedding drift")
        if drift_result is not None:
            print(f"\n{'='*70}")
            print("ABLATION RESULT")
            print(f"{'='*70}")
            print(f"  Driver-only PR-AUC:            {driver_result['ap']:.4f}")
            print(f"  Driver + embedding-drift PR-AUC: {drift_result['ap']:.4f}")
            delta = drift_result["ap"] - driver_result["ap"]
            print(f"  Delta: {delta:+.4f}  "
                  f"({'embedding drift adds predictive power' if delta > 0.01 else 'no clear improvement from embedding drift'})")
            if drift_result["ap"] >= driver_result["ap"]:
                best_result, best_features = drift_result, drift_result["feature_names"]

    if best_result is None:
        print("\nNo model could be trained -- see warnings above. Not saving a model file.")
        return

    import joblib
    os.makedirs(RISK_MODEL_DIR, exist_ok=True)
    joblib.dump({"model": best_result["model"], "feature_names": best_features}, RISK_MODEL_PATH)
    print(f"\nBest model saved to: {RISK_MODEL_PATH}")
    print(f"Score a location with:")
    print(f"    python 11_forest_risk_forecast.py --predict <lon> <lat>")


if __name__ == "__main__":
    main()