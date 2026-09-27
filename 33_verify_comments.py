"""
EcoLens Step 33 -- VERIFY EVERY REVIEW COMMENT AGAINST REAL EVIDENCE

WHY THIS EXISTS
---------------
REVIEW_RESPONSE.md said "25 complete, 0 partial, 0 not started". Two of those 25 were not
actually satisfied, and nothing caught it:

  C11 (do retrieved ecosystems share real ecological characteristics?)
      07b and 07c both write results/ecological_similarity.json, and 07c ran last. The
      per-model agreement scores for all five backbones were replaced by a single
      spectral-baseline entry. The documents still quoted "Temperature MAE 6.34 degC (Clay)"
      from a file that no longer contained it.

  C19 (SHAP / explainable AI)
      Three of five SHAP plots were 0-byte files. A label like "Ablation: No Climate" became
      a path containing a colon, and on NTFS everything after a colon is an alternate data
      stream -- so Windows created an empty file and wrote the plot into a hidden stream.
      savefig raised nothing, so the failure was invisible.

Both were marked complete because "complete" meant somebody had written a tick in a table.
This script replaces the tick with a check: for each comment, does its evidence file EXIST,
is it NON-EMPTY, and does it contain the KEYS the claim depends on?

A comment is only PASS if every check passes. Anything else is reported as FAIL or PARTIAL
with the reason, and the exit code is non-zero so a pipeline run cannot quietly end with
broken evidence.

Usage:
    python 33_verify_comments.py
    python 33_verify_comments.py --json      # machine-readable
"""

import argparse
import json
import os
import sys

from config import RESULTS_DIR, RISK_MODEL_DIR, LOGS_DIR, DASHBOARDS_DIR, PROJECT_ROOT

MODELS = ["prithvi", "clay", "satlas", "vit", "resnet"]


def R(name):
    return os.path.join(RESULTS_DIR, name)


def M(name):
    return os.path.join(RISK_MODEL_DIR, name)


def _load(path):
    if not os.path.exists(path):
        return None, f"missing file: {os.path.relpath(path, PROJECT_ROOT)}"
    if os.path.getsize(path) == 0:
        return None, f"EMPTY file: {os.path.relpath(path, PROJECT_ROOT)}"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"unreadable: {type(e).__name__}: {e}"


def need_json(path, keys=(), all_models=False, min_len=None):
    """Evidence check: file loads, and the keys the claim rests on are present."""
    def check():
        d, err = _load(path)
        if err:
            return False, err
        if all_models:
            miss = [m for m in MODELS if m not in d]
            if miss:
                return False, f"missing models: {', '.join(miss)}"
        for k in keys:
            if isinstance(d, dict) and k not in d:
                return False, f"missing key '{k}'"
        if min_len is not None and hasattr(d, "__len__") and len(d) < min_len:
            return False, f"only {len(d)} entries, expected >= {min_len}"
        return True, f"{os.path.basename(path)} ok"
    return check


def need_files(paths, nonempty=True, min_count=None):
    def check():
        found, empty = [], []
        for p in paths:
            if os.path.exists(p):
                if nonempty and os.path.getsize(p) == 0:
                    empty.append(os.path.relpath(p, PROJECT_ROOT))
                else:
                    found.append(p)
        if empty:
            return False, f"EMPTY: {', '.join(empty)}"
        want = min_count if min_count is not None else len(paths)
        if len(found) < want:
            return False, f"found {len(found)}/{want}"
        return True, f"{len(found)} file(s) ok"
    return check


def need_all(*checks):
    def check():
        msgs = []
        for c in checks:
            ok, msg = c()
            if not ok:
                return False, msg
            msgs.append(msg)
        return True, "; ".join(msgs)
    return check


def c5_regions():
    """C5 asks for MORE ecosystems, regions and environmental conditions. Count what is
    actually configured rather than trusting prose."""
    def check():
        try:
            from config import risk_regions, PATCH_LOCATIONS
        except Exception as e:
            return False, f"config import failed: {e}"
        rr = risk_regions()
        cat, _ = _load(os.path.join(PROJECT_ROOT, "data", "metadata", "catalog.json"))
        n_eco = len({e["ecosystem"] for e in cat}) if isinstance(cat, list) else 0
        n_loc = len({e.get("base_id", e["id"].rsplit("_p", 1)[0])
                     for e in cat}) if isinstance(cat, list) else 0
        auto = sum(1 for r in rr if r.get("selection") == "auto")
        detail = (f"{len(rr)} risk regions ({auto} auto-selected), "
                  f"{n_loc} catalogue locations across {n_eco} ecosystems")
        # 17 regions gives only 16 folds, which is the sample-size limitation the whole
        # comment is about -- treat an unexpanded project as PARTIAL, not PASS
        if len(rr) < 40:
            return "PARTIAL", detail + " -- fewer than 40 regions, folds remain few"
        return True, detail
    return check


def c19_shap():
    """C19 needs SHAP plots that are real files, not 0-byte alternate-data-stream victims."""
    def check():
        if not os.path.isdir(RISK_MODEL_DIR):
            return False, "risk_model dir missing"
        shap = [f for f in os.listdir(RISK_MODEL_DIR) if f.startswith("shap_summary")]
        empty = [f for f in shap
                 if os.path.getsize(os.path.join(RISK_MODEL_DIR, f)) == 0]
        real = [f for f in shap if f.endswith(".png") and f not in empty]
        if empty:
            return False, (f"{len(empty)} EMPTY shap file(s): {', '.join(empty)} "
                           f"(NTFS stream bug -- see 11's filename sanitiser)")
        if not real:
            return False, "no SHAP plots found"
        return True, f"{len(real)} SHAP plot(s), none empty"
    return check


def c1_bridge():
    """C1 is the connection between the two pillars. The analog features ARE that bridge,
    but a bridge with no control is not evidence -- 32's random-analog arm decides it."""
    def check():
        ok_feat = os.path.exists(R("analog_cell_features.csv"))
        d, err = _load(R("pillar_bridge.json"))
        if not ok_feat:
            return False, "analog_cell_features.csv missing -- no bridge at all"
        if err:
            return "PARTIAL", ("analog features exist, but pillar_bridge.json missing: the "
                               "random-analog control that rules out 'any 8 extra columns "
                               "would help' has not been run")
        sp = (d.get("protocols") or {}).get("spatial") or {}
        tests = sp.get("paired_tests") or {}
        bc = tests.get("B_vs_C_similarity_matters")
        if not d.get("has_random_control") or bc is None:
            return "PARTIAL", "bridge scored but the random-analog control arm is absent"
        if bc["wilcoxon_p"] < 0.05 and bc["mean_delta"] > 0:
            return True, (f"real analogs beat random: delta {bc['mean_delta']:+.4f}, "
                          f"{bc['improved']}/{bc['n']} folds, p={bc['wilcoxon_p']:.4f}")
        return "PARTIAL", (f"control run but NOT passed: real vs random delta "
                           f"{bc['mean_delta']:+.4f}, p={bc['wilcoxon_p']:.4f} -- the gain "
                           f"cannot be attributed to similarity")
    return check


# comment id -> (short description, evidence check)
COMMENTS = {
    "C1": ("Scientific hypothesis; how retrieval feeds forecasting", c1_bridge()),
    "C2": ("What new ecological knowledge do analogs give?",
           need_json(R("successional_stages.json"))),
    "C3": ("Why foundation models rather than spectral indices?",
           need_all(need_json(R("retrieval_results_spectral_baseline.json")),
                    need_json(R("evaluation_report.json"), all_models=True))),
    "C4": ("Distinguish from existing systems; contribution",
           need_json(R("cross_region_retrieval.json"))),
    "C5": ("Expand the retrieval / region database", c5_regions()),
    "C6": ("Document preprocessing / QC / acquisition",
           need_files([R("qc_report.csv"), R("qc_report.md")])),
    "C7": ("Retrieval consistent across geographic regions?",
           need_json(R("cross_region_retrieval.json"), all_models=True)),
    "C8": ("Compare multiple geospatial foundation models",
           need_json(R("evaluation_report.json"), all_models=True)),
    "C9": ("Justify embedding layer; does dimension matter?",
           need_all(need_json(R("layer_ablation.json")),
                    need_json(R("dimension_sweep.json")))),
    "C10": ("Cluster visualisation + seasonal/atmospheric stability",
            need_all(need_json(R("temporal_stability.json")),
                     need_json(R("atmospheric_stability.json")),
                     need_json(R("interannual_stability.json")),
                     need_json(R("stability_normalized.json")))),
    "C11": ("Do analogs share real ecological characteristics?",
            # the overwrite bug: this file must contain the MODELS, not just the baseline
            need_json(R("ecological_similarity.json"), all_models=True)),
    "C12": ("Multiple similarity metrics",
            need_json(R("retrieval_perf.json"), all_models=True)),
    "C13": ("Visual examples of success / partial / failure",
            need_files([R(f"retrieval_case_studies_{m}.json") for m in MODELS])),
    "C14": ("Visually similar but ecologically different failures",
            need_json(R("explainable_retrieval.json"))),
    "C15": ("Can analogs improve forest-loss prediction?",
            need_all(need_json(M("analog_ablation_spatial.json"),
                               keys=("mean_delta", "wilcoxon_p", "regions_improved")),
                     need_json(M("analog_ablation_temporal.json")))),
    "C16": ("Do embedding features help alongside drivers?",
            need_json(M("drift_ablation_matched.json"), keys=("arms",))),
    "C17": ("Benchmark GB vs RF, XGBoost, LightGBM",
            need_json(M("risk_model_metrics.json"), keys=("arms",))),
    "C18": ("Feature-group ablation",
            need_json(M("risk_model_metrics.json"), keys=("feature_group_ablation",))),
    "C19": ("SHAP / explainable AI", c19_shap()),
    "C20": ("Disturbance pathways, succession, fragmentation",
            need_json(R("successional_stages.json"), keys=("by_stage",))),
    "C21": ("Applications", need_json(R("location_forecasts.json"), keys=("summary",))),
    "C22": ("Sensitivity: patch size, dimension, window, threshold",
            need_all(need_json(R("patch_size_sweep.json")),
                     need_json(R("dimension_sweep.json")),
                     need_json(R("horizon_sweep.json"), keys=("horizons",)),
                     need_json(R("retrieval_diagnostics.json"), all_models=True))),
    "C23": ("Processing time, memory, scalability",
            need_json(R("retrieval_perf.json"), all_models=True)),
    "C24": ("Limitations", need_files([os.path.join(PROJECT_ROOT, "docs",
                                                    "REVIEW_RESPONSE.md")])),
    "C25": ("Package as toolkit / desktop application",
            need_all(need_files([os.path.join(PROJECT_ROOT, "docs", "TOOLKIT.md"),
                                 os.path.join(PROJECT_ROOT, "setup.py")]),
                     need_files([os.path.join(DASHBOARDS_DIR, "retrieval_dashboard.html")]))),
}


def main():
    ap = argparse.ArgumentParser(description="EcoLens 33: verify review comments")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows, n_pass, n_part, n_fail = [], 0, 0, 0
    for cid, (desc, check) in COMMENTS.items():
        try:
            ok, msg = check()
        except Exception as e:
            ok, msg = False, f"check raised {type(e).__name__}: {e}"
        status = "PASS" if ok is True else ("PARTIAL" if ok == "PARTIAL" else "FAIL")
        n_pass += status == "PASS"
        n_part += status == "PARTIAL"
        n_fail += status == "FAIL"
        rows.append({"id": cid, "comment": desc, "status": status, "evidence": msg})

    if args.json:
        print(json.dumps({"summary": {"pass": n_pass, "partial": n_part, "fail": n_fail},
                          "comments": rows}, indent=2))
    else:
        print(f"\n{'=' * 100}")
        print("REVIEW COMMENT VERIFICATION -- evidence checked, not assumed")
        print(f"{'=' * 100}")
        print(f"{'ID':5} {'STATUS':8} {'COMMENT':52} EVIDENCE")
        print("-" * 100)
        for r in rows:
            print(f"{r['id']:5} {r['status']:8} {r['comment'][:52]:52} {r['evidence'][:80]}")
        print("-" * 100)
        print(f"  {n_pass} PASS   {n_part} PARTIAL   {n_fail} FAIL   of {len(rows)}")
        if n_fail or n_part:
            print("\n  Anything not PASS is listed above with the reason. A comment is PASS only")
            print("  when its evidence file exists, is non-empty, and holds the keys the claim")
            print("  depends on -- which is what C11 and C19 previously failed silently.")

    out = os.path.join(RESULTS_DIR, "comment_verification.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": {"pass": n_pass, "partial": n_part, "fail": n_fail},
                   "comments": rows}, f, indent=2)
    if not args.json:
        print(f"\nwrote {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
