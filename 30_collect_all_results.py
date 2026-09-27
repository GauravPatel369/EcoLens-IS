"""
EcoLens Step 30 -- COLLECT EVERY RESULT INTO ONE FILE

WHY THIS EXISTS
---------------
Results were spread across ~30 JSON files plus several run logs, and the report documents
were filled in by hand from whichever file someone happened to open. That produced exactly
the failure you would predict: the same quantity appeared with different values in different
documents (+11.0% vs +11.5%, 14/16 vs 13/16 regions improved, p = 0.021 vs p = 0.23 for
Clay's replication), because four result files had gone stale relative to a rebuilt
cell_year_features.csv and nobody could tell which numbers came from which run.

This script is the single source of truth. Everything downstream -- report tables, figures,
the review-response document -- reads ALL_RESULTS.json and _report_numbers.json, never the
individual files. If a number is wrong it is wrong in exactly one place.

It is read-only with respect to the pipeline: it loads artifacts and writes two aggregates.
Missing inputs are recorded as {"_missing": ...} rather than skipped silently, so a gap shows
up in the output instead of becoming an absent table nobody notices.

Usage:
    python 30_collect_all_results.py
"""

import json
import os
import re
import sys
from datetime import datetime

from config import (RESULTS_DIR, RISK_MODEL_DIR, LOGS_DIR, METADATA_CATALOG_PATH,
                    PROJECT_ROOT, risk_regions)

OUT_ALL = os.path.join(RESULTS_DIR, "ALL_RESULTS.json")
OUT_NUM = os.path.join(RESULTS_DIR, "_report_numbers.json")

MODELS = ["prithvi", "clay", "satlas", "vit", "resnet"]

# Files in results/ keyed by the section name the documents refer to.
RESULT_FILES = {
    "retrieval_evaluation": "evaluation_report.json",
    "ecological_agreement": "ecological_similarity.json",
    "cross_region": "cross_region_retrieval.json",
    "layer_ablation": "layer_ablation.json",
    "dimension_sweep": "dimension_sweep.json",
    "patch_size_sweep": "patch_size_sweep.json",
    "horizon_sweep": "horizon_sweep.json",
    "retrieval_diagnostics": "retrieval_diagnostics.json",
    "seasonal_stability": "temporal_stability.json",
    "atmospheric_stability": "atmospheric_stability.json",
    "interannual_stability": "interannual_stability.json",
    "stability_normalised": "stability_normalized.json",
    "successional_stages": "successional_stages.json",
    "risk_calibration": "risk_calibration.json",
    "retrieval_performance": "retrieval_perf.json",
    "location_forecasts": "location_forecasts.json",
    "region_risk": "region_risk_grids.json",
    "spectral_baseline": "retrieval_results_spectral_baseline.json",
    "candidate_risk_regions": "candidate_risk_regions.json",
    "pillar_bridge": "pillar_bridge.json",
}

# Files in risk_model/
RISK_FILES = {
    "risk_model_metrics": "risk_model_metrics.json",
    "analog_ablation_spatial": "analog_ablation_spatial.json",
    "analog_ablation_temporal": "analog_ablation_temporal.json",
    "analog_ablation_spatial_clay": "analog_ablation_spatial_clay.json",
    "analog_ablation_temporal_clay": "analog_ablation_temporal_clay.json",
    "drift_ablation_matched": "drift_ablation_matched.json",
}


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_missing": os.path.basename(path), "_error": f"{type(e).__name__}: {e}"}


def scrape_risk_log():
    """Fall back to the run log for the headline PR-AUC.

    11_forest_risk_forecast.py now writes risk_model_metrics.json, so this is only needed
    for runs that predate that change. Kept because without it a project whose log was
    deleted had no machine-readable copy of its own primary metric.
    """
    for name in ("11_ablation.log", "11_wdpa_ablation.log", "11_base.log"):
        p = os.path.join(LOGS_DIR, name)
        if not os.path.exists(p):
            continue
        s = open(p, encoding="utf-8", errors="ignore").read()
        blocks = {}
        for m in re.finditer(r"MODEL: ([^\n]+)\n=+\n(.*?)(?=\nMODEL: |\Z)", s, re.S):
            label, body = m.group(1).strip(), m.group(2)
            ap = re.search(r"Average Precision \(PR-AUC\): ([0-9.]+)", body)
            roc = re.search(r"Best Model:.*?\n\s*ROC-AUC:\s+([0-9.]+)", body, re.S)
            ci = re.search(r"PR-AUC\s+95% CI[^:]*: \[([0-9.]+), ([0-9.]+)\]", body)
            if ap:
                blocks[label] = {
                    "pr_auc": float(ap.group(1)),
                    "roc_auc": float(roc.group(1)) if roc else None,
                    "pr_auc_ci95": [float(ci.group(1)), float(ci.group(2))] if ci else None,
                    "_source": f"scraped from logs/{name}",
                }
        if blocks:
            return {"arms": blocks, "_source": f"logs/{name}"}
    return None


def fsize(path):
    try:
        n = (sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(path) for f in fs)
             if os.path.isdir(path) else os.path.getsize(path))
    except Exception:
        return "absent"
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    R = {"generated": datetime.now().isoformat(timespec="seconds"),
         "project_root": PROJECT_ROOT, "sections": {}}
    S = R["sections"]

    # ---- catalogue ----
    cat = load(METADATA_CATALOG_PATH)
    if isinstance(cat, list):
        eco = {}
        for e in cat:
            eco[e["ecosystem"]] = eco.get(e["ecosystem"], 0) + 1
        S["catalog"] = {
            "sub_crops": len(cat),
            "base_locations": len({e.get("base_id", e["id"].rsplit("_p", 1)[0]) for e in cat}),
            "ecosystems": eco, "n_ecosystems": len(eco),
        }
    else:
        S["catalog"] = cat

    # ---- risk regions actually configured ----
    try:
        rr = risk_regions()
        S["risk_regions"] = {
            "n": len(rr),
            "ids": [r["id"] for r in rr],
            "n_auto_selected": sum(1 for r in rr if r.get("selection") == "auto"),
        }
    except Exception as e:
        S["risk_regions"] = {"_error": str(e)}

    for key, fn in RESULT_FILES.items():
        S[key] = load(os.path.join(RESULTS_DIR, fn))
    for key, fn in RISK_FILES.items():
        S[key] = load(os.path.join(RISK_MODEL_DIR, fn))

    # headline metrics: prefer the JSON the model now writes, fall back to the log
    if "_missing" in (S.get("risk_model_metrics") or {}):
        scraped = scrape_risk_log()
        if scraped:
            S["risk_model_metrics"] = scraped

    # ---- what is NOT in version control, and how to rebuild it ----
    # The heavy inputs and intermediates are regenerable but expensive, and none of them are
    # committed. Recording size and the exact command that produces each one is what makes a
    # fresh clone reproducible rather than merely readable.
    NOT_IN_GIT = [
        ("data/hansen", "Hansen Global Forest Change tiles (lossyear + treecover2000)",
         "auto-downloaded on first use by 10_grid_tiling_labels.py"),
        ("data/reference", "RESOLVE ecoregions, WorldClim, DEM tiles, WDPA GeoPackage",
         "download_reference_data.py, then 26_build_wdpa_layer.py"),
        ("data/patches", "raw 6-band Sentinel-2 patches, one per location",
         "01_acquire_patches.py"),
        ("data/patches_processed", "BOA-corrected, nodata-filled, cropped sub-crops",
         "02_preprocess_patches.py"),
        ("data/embeddings", "catalogue embeddings for all five backbones",
         "03_extract_embeddings.py --model <name>"),
        ("models", "Prithvi and Clay checkpoints",
         "downloaded on first run of 03_extract_embeddings.py"),
        ("outputs/risk_model/cell_year_features.csv",
         "labelled cell-years across all configured risk regions",
         "10_grid_tiling_labels.py"),
        ("outputs/results/analog_cell_embeddings_prithvi.npz",
         "per-cell Sentinel-2 embeddings (the expensive half of stage 13)",
         "13_analog_risk_features.py --model prithvi"),
        ("outputs/dashboards", "interactive HTML dashboards",
         "05_create_database_and_dashboard.py and 08_retrieval_dashboard.py"),
    ]
    S["artifacts_not_in_git"] = [
        {"path": p, "what": w, "regenerate_with": h, "size": fsize(
            os.path.join(PROJECT_ROOT, p.replace("/", os.sep)))}
        for p, w, h in NOT_IN_GIT]

    # ---- figures on disk ----
    figs = []
    for root in (RESULTS_DIR, RISK_MODEL_DIR):
        for r_, _, fs in os.walk(root):
            for f in sorted(fs):
                if f.endswith(".png"):
                    rel = os.path.join(r_, f).replace(PROJECT_ROOT, ".").replace("\\", "/")
                    figs.append({"path": rel, "size": fsize(os.path.join(r_, f))})
    S["figures_on_disk"] = figs
    # a 0-byte figure is the NTFS alternate-data-stream bug that silently emptied three SHAP
    # plots; surface it here so it can never look like evidence again
    S["empty_figures"] = [f["path"] for f in figs if f["size"].startswith("0 ")]

    with open(OUT_ALL, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=str)

    # ---- flattened numbers the report tables read ----
    N = {}
    ev = S.get("retrieval_evaluation") or {}
    if "_missing" not in ev:
        N["map"] = {m: {"mAP": v["cosine"]["overall"]["mAP"],
                        "lo": v["cosine"]["overall"]["mAP_95ci_lower"],
                        "hi": v["cosine"]["overall"]["mAP_95ci_upper"],
                        "MRR": v["cosine"]["overall"]["MRR"]}
                    for m, v in ev.items() if isinstance(v, dict) and "cosine" in v}
    dg = S.get("retrieval_diagnostics") or {}
    if "_missing" not in dg:
        N["diag"] = {}
        for m, v in dg.items():
            if not isinstance(v, dict) or "sweep" not in v:
                continue
            best = [r for r in v["sweep"] if r["threshold"] == v["best_f1_threshold"]]
            b = best[0] if best else {}
            N["diag"][m] = {"sil": v["cluster"]["silhouette_cosine"],
                            "ari": v["cluster"]["adjusted_rand_index"],
                            "best_t": v["best_f1_threshold"],
                            "chance": v["chance_precision"],
                            "prec": b.get("precision"), "rec": b.get("recall"),
                            "f1": b.get("f1")}
    for k, src in (("horizon", "horizon_sweep"), ("stab", "stability_normalised"),
                   ("dim", "dimension_sweep"), ("patch", "patch_size_sweep"),
                   ("layer", "layer_ablation"), ("perf", "retrieval_performance"),
                   ("drift", "drift_ablation_matched"), ("risk", "risk_model_metrics"),
                   ("bridge", "pillar_bridge")):
        v = S.get(src)
        if v and "_missing" not in v:
            N[k] = v.get("horizons") if src == "horizon_sweep" else v
    cr = S.get("cross_region") or {}
    if "_missing" not in cr:
        N["cross"] = {m: {"strat": v["stratified"]["overall_map_at_k"],
                          "loro": v["leave_one_realm_out"]["overall_map_at_k"]}
                      for m, v in cr.items()
                      if isinstance(v, dict) and "stratified" in v}
    sc = S.get("successional_stages") or {}
    if "_missing" not in sc and "by_stage" in sc:
        N["succ"] = sc["by_stage"]
    cal = S.get("risk_calibration") or {}
    if "_missing" not in cal:
        N["cal"] = {k: cal.get(k) for k in
                    ("brier_before", "brier_after", "mean_abs_gap_before",
                     "mean_abs_gap_after", "roc_auc_before", "roc_auc_after",
                     "holdout_observed_rate", "n_holdout")}

    with open(OUT_NUM, "w", encoding="utf-8") as f:
        json.dump(N, f, indent=1, default=str)

    # ---- report ----
    missing = sorted(k for k, v in S.items()
                     if isinstance(v, dict) and "_missing" in v)
    print(f"sections collected : {len(S)}")
    print(f"figures on disk    : {len(figs)}")
    if S["empty_figures"]:
        print(f"EMPTY FIGURES ({len(S['empty_figures'])}) -- these are not evidence:")
        for p in S["empty_figures"]:
            print(f"    {p}")
    if missing:
        print(f"MISSING inputs ({len(missing)}) -- recorded in the output, not hidden:")
        for k in missing:
            print(f"    {k:32s} <- {S[k]['_missing']}")
    if "risk_regions" in S and "n" in S["risk_regions"]:
        print(f"risk regions       : {S['risk_regions']['n']} "
              f"({S['risk_regions']['n_auto_selected']} auto-selected)")
    print(f"\nwrote {OUT_ALL}")
    print(f"wrote {OUT_NUM}")


if __name__ == "__main__":
    main()
