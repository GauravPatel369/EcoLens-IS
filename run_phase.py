"""
EcoLens resumable pipeline runner.

WHY THIS EXISTS (and why not run_pipeline.py)
---------------------------------------------
run_pipeline.py rmtree's patches_processed/ and the embedding dirs before it
starts, has no logging and no state, and stops at 08. That makes every
interruption a full restart -- and a full restart is ~40 min of embedding
alone (03: Clay is 20 min of it; see implementation.md 3.4b).

This runner:
  * writes each step's output to logs/<step>.log UNBUFFERED, so progress is
    visible WHILE it runs (PowerShell pipelines buffer; a file does not)
  * records each step in logs/pipeline_state.json with start/end/duration/rc
  * on restart, SKIPS steps already marked done whose declared artifacts are
    still on disk -- so an interrupted run resumes where it stopped
  * NEVER deletes anything

Usage:
    python run_phase.py --list              show steps and their state
    python run_phase.py                     run everything not yet done
    python run_phase.py --only 11_base      run one step
    python run_phase.py --from 10_tiling    run from this step onward
    python run_phase.py --redo 07b          clear one step's state, then run it
    python run_phase.py --dry-run           show what would run

Interrupt with Ctrl-C: the running step is marked "interrupted" and will be
re-run next time; every completed step stays done.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# All artifact paths come from config, which anchors them to the project root -- so the
# runner works from any working directory and its "is this step done?" checks look in the
# same place the steps actually write.
from config import (LOGS_DIR as LOG_DIR, RESULTS_DIR, METADATA_CATALOG_PATH,
                    PATCHES_DIR, PROCESSED_PATCHES_DIR, RISK_FEATURES_PATH)
STATE_PATH = os.path.join(LOG_DIR, "pipeline_state.json")
MODELS = ["prithvi", "vit", "resnet", "clay", "satlas"]

PY = sys.executable


def emb_dir(model):
    """Where a model's embeddings live -- read from config, not a second hardcoded map.
    The duplicate map here went stale at the 15 Sep restructure and made every embedding
    step report done(no art): the state file said done, the checker looked in a directory
    that no longer existed."""
    from config import SUPPORTED_MODELS
    return SUPPORTED_MODELS[model]["embeddings_dir"]


def n_npy(d):
    """Artifact check helper: how many .npy files a directory holds."""
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.endswith(".npy"))


def expected_subcrops():
    """How many sub-crops the CURRENT catalog should contain.

    This must not be a hardcoded number. Phase 1 took the catalog from 77 to
    131 locations, and a fixed `>= 700` check would have marked the embedding
    steps "done" at 770 files when 1310 were required -- silently reusing
    stale vectors, which is the same class of failure as 03's skip-if-exists
    (see implementation.md 3.2). Derive it from the catalog on disk, falling
    back to config when the catalog has not been rebuilt yet.
    """
    try:
        with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
            cat = json.load(f)
        n = len({e.get("base_id", e["id"].rsplit("_p", 1)[0]) for e in cat})
        if n:
            return n * 10
    except Exception:
        pass
    try:
        from config import PATCH_LOCATIONS
        return len(PATCH_LOCATIONS) * 10
    except Exception:
        return 0


def enough(d):
    """True when directory d holds an embedding for every current sub-crop."""
    want = expected_subcrops()
    return want > 0 and n_npy(d) >= want


# Each step: (name, argv, artifacts)
# `artifacts` is a list of paths that must exist for the step to count as done.
# A callable artifact is invoked and must return truthy.
def build_steps():
    steps = [
        ("01_acquire", [PY, "01_acquire_patches.py"], [f"{PATCHES_DIR}/acquisition_manifest.json"]),
        ("00_validate", [PY, "00_validate_locations.py"], []),
        ("02_preprocess", [PY, "02_preprocess_patches.py"],
         [lambda: enough(PROCESSED_PATCHES_DIR), METADATA_CATALOG_PATH]),
    ]
    for m in MODELS:
        steps.append((f"03_embed_{m}",
                      [PY, "03_extract_embeddings.py", "--model", m, "--force"],
                      [lambda d=emb_dir(m): enough(d)]))
    steps += [
        ("04_finalize", [PY, "04_finalize_and_analyze.py"], []),
        ("05_dashboard", [PY, "05_create_database_and_dashboard.py"],
         ["embedding_dashboard.html"]),
    ]
    for m in MODELS:
        steps.append((f"06_retrieval_{m}",
                      [PY, "06_retrieval_engine.py", "--model", m],
                      [f"{RESULTS_DIR}/retrieval_results_{m}.json"]))
    steps += [
        ("07_evaluate", [PY, "07_evaluate_retrieval.py"],
         [f"{RESULTS_DIR}/evaluation_report.json"]),
        ("09_explain", [PY, "09_explainability_engine.py"],
         [f"{RESULTS_DIR}/ecosystem_descriptors.json", f"{RESULTS_DIR}/explainable_retrieval.json"]),
        ("07b_ecological", [PY, "07b_evaluate_ecological_similarity.py"],
         [f"{RESULTS_DIR}/ecological_similarity.json"]),
        ("07c_baseline", [PY, "07c_evaluate_baseline_retrieval.py"],
         [f"{RESULTS_DIR}/retrieval_results_spectral_baseline.json"]),
    ]
    for m in MODELS:
        steps.append((f"07d_cases_{m}",
                      [PY, "07d_retrieval_case_studies.py", "--model", m],
                      [f"{RESULTS_DIR}/retrieval_case_studies_{m}.json"]))
    steps += [
        ("08_dashboard", [PY, "08_retrieval_dashboard.py"], ["retrieval_dashboard.html"]),
        ("phase0_qc_report", [PY, "phase0_qc_report.py"],
         [f"{RESULTS_DIR}/qc_report.csv", f"{RESULTS_DIR}/qc_report.md"]),
        ("10_tiling", [PY, "10_grid_tiling_labels.py"],
         [RISK_FEATURES_PATH]),
        ("11_base", [PY, "11_forest_risk_forecast.py"], []),
        ("11_ablation", [PY, "11_forest_risk_forecast.py", "--ablation"], []),
        ("11_spatial", [PY, "11_forest_risk_forecast.py", "--spatial-holdout"], []),
        ("11_predict", [PY, "11_forest_risk_forecast.py", "--predict", "88.85", "21.95"], []),
    ]
    return steps


def artifacts_ok(artifacts):
    for a in artifacts:
        if callable(a):
            if not a():
                return False
        elif not os.path.exists(a):
            return False
    return True


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state):
    os.makedirs(LOG_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def is_done(name, artifacts, state):
    rec = state.get(name)
    if not rec or rec.get("status") != "done":
        return False
    return artifacts_ok(artifacts)


# 07d overwrites results/retrieval_case_studies.json each run, so each model's
# output has to be copied aside immediately or only the last survives.
def post_step(name):
    if name.startswith("07d_cases_"):
        model = name.replace("07d_cases_", "")
        src = f"{RESULTS_DIR}/retrieval_case_studies.json"
        dst = f"{RESULTS_DIR}/retrieval_case_studies_{model}.json"
        if os.path.exists(src):
            import shutil
            shutil.copyfile(src, dst)


def run_step(name, argv, log_path):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"          # so the log streams instead of buffering
    env["PYTHONIOENCODING"] = "utf-8"
    started = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(f"# {name}\n# {' '.join(argv)}\n# started {datetime.now().isoformat()}\n\n")
        log.flush()
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
        rc = proc.wait()
    return rc, time.time() - started


def main():
    ap = argparse.ArgumentParser(description="EcoLens resumable pipeline runner")
    ap.add_argument("--list", action="store_true", help="show steps and state, run nothing")
    ap.add_argument("--only", help="run just this step")
    ap.add_argument("--from", dest="from_step", help="run from this step onward")
    ap.add_argument("--redo", action="append", default=[],
                    help="clear this step's state so it re-runs (repeatable)")
    ap.add_argument("--redo-from", dest="redo_from",
                    help="clear the state of this step and EVERY step after it. Use after "
                         "changing config.PATCH_LOCATIONS: the artifact checks alone cannot "
                         "detect that, because expected_subcrops() reads the catalog, and the "
                         "catalog is not rebuilt until 02 runs.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    steps = build_steps()
    state = load_state()

    for r in args.redo:
        state.pop(r, None)
    if args.redo_from:
        names = [s[0] for s in steps]
        if args.redo_from not in names:
            print(f"no such step: {args.redo_from}")
            return 2
        cleared = names[names.index(args.redo_from):]
        for n in cleared:
            state.pop(n, None)
        print(f"cleared state for {len(cleared)} step(s) from {args.redo_from} onward")
    if args.redo or args.redo_from:
        save_state(state)

    if args.list:
        print(f"{'step':<22}{'state':<14}{'seconds':>9}  log")
        print("-" * 70)
        for name, argv, arts in steps:
            rec = state.get(name, {})
            st = rec.get("status", "-")
            if st == "done" and not artifacts_ok(arts):
                st = "done(no art)"
            secs = rec.get("seconds")
            print(f"{name:<22}{st:<14}{('%.0f' % secs) if secs else '':>9}  "
                  f"{LOG_DIR}/{name}.log")
        return 0

    selected = steps
    if args.only:
        selected = [s for s in steps if s[0] == args.only]
        if not selected:
            print(f"no such step: {args.only}")
            return 2
    elif args.from_step:
        names = [s[0] for s in steps]
        if args.from_step not in names:
            print(f"no such step: {args.from_step}")
            return 2
        selected = steps[names.index(args.from_step):]

    print(f"EcoLens pipeline — {len(selected)} step(s) considered")
    print(f"state: {STATE_PATH}   logs: {LOG_DIR}/<step>.log\n")

    for name, argv, arts in selected:
        if is_done(name, arts, state) and not args.only:
            print(f"  SKIP  {name}  (already done)")
            continue
        if args.dry_run:
            print(f"  RUN   {name}  ->  {' '.join(argv[1:])}")
            continue

        log_path = os.path.join(LOG_DIR, f"{name}.log")
        state[name] = {"status": "running",
                       "started": datetime.now().isoformat(),
                       "cmd": " ".join(argv[1:])}
        save_state(state)
        print(f"  RUN   {name} ... ", end="", flush=True)
        try:
            rc, secs = run_step(name, argv, log_path)
        except KeyboardInterrupt:
            state[name]["status"] = "interrupted"
            save_state(state)
            print("INTERRUPTED")
            return 130

        post_step(name)
        ok = (rc == 0) and artifacts_ok(arts)
        state[name].update({
            "status": "done" if ok else "failed",
            "finished": datetime.now().isoformat(),
            "seconds": round(secs, 1),
            "returncode": rc,
            "artifacts_ok": artifacts_ok(arts),
        })
        save_state(state)
        print(f"{'ok' if ok else 'FAILED'}  ({secs:.0f}s, rc={rc})")
        if not ok:
            print(f"        see {log_path}")
            print("        fix, then re-run this script -- completed steps are skipped.")
            return 1

    print("\nAll selected steps complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
