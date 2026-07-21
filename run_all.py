"""
EcoLens -- run_all.py

Runs the entire pipeline (steps 01-11) in order, in one command.
Prints per-step timing and a final summary, and stops (by default) on
the first failure rather than plowing ahead on broken state.

---------------------------------------------------------------------
USAGE
---------------------------------------------------------------------
    python run_all.py                        # full run, all 3 models, steps 1-11
    python run_all.py --models prithvi        # only run Prithvi (skip vit/resnet)
    python run_all.py --skip-risk             # stop after step 9 (skip 10/11)
    python run_all.py --only 3,6              # run only steps 3 and 6
    python run_all.py --start-from 7          # resume from step 7 onward
    python run_all.py --predict 88.85 21.95   # after training, score one point
    python run_all.py --dry-run               # print the commands without running them
    python run_all.py --continue-on-error      # don't stop the whole run on one failure

---------------------------------------------------------------------
WHAT THIS DOES NOT DO
---------------------------------------------------------------------
It does not download reference data (WDPA/WorldClim/SRTM/RESOLVE
Ecoregions) -- see README.md's "Reference data setup". Steps 9-11
will run without it, just with fewer real fields populated (never
fabricated ones -- see geo_lookups.py). It does not shrink
10_grid_tiling_labels.py's grid size for you; if you want a fast first
pass through step 10, edit GRID_REGION_BUFFER_KM / the observation-year
range in config.py / 10_grid_tiling_labels.py before running with
--only 10, then widen it once you've confirmed the output looks right
(see README.md's timing discussion).

Run from the same directory as the numbered scripts (this file
expects 01_acquire_patches.py etc. to be siblings of itself).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (step number, script filename, extra args per invocation)
# 03 and 06 run once per requested model; every other step runs once.
MODEL_STEPS = {3: "03_extract_embeddings.py", 6: "06_retrieval_engine.py"}

STEPS = [
    (1, "01_acquire_patches.py", []),
    (2, "02_preprocess_patches.py", []),
    (3, "03_extract_embeddings.py", []),   # expanded per-model below
    (4, "04_finalize_and_analyze.py", []),
    (5, "05_create_database_and_dashboard.py", []),
    (6, "06_retrieval_engine.py", []),     # expanded per-model below
    (7, "07_evaluate_retrieval.py", []),
    (8, "08_retrieval_dashboard.py", []),
    (9, "09_explainability_engine.py", []),
    (10, "10_grid_tiling_labels.py", []),
    (11, "11_forest_risk_forecast.py", []),
]


def build_run_plan(models, only, start_from, skip_risk):
    """Expand STEPS into a flat list of (step_num, description, argv)."""
    selected = {n for n, _, _ in STEPS}
    if only is not None:
        selected &= set(only)
    if start_from is not None:
        selected = {n for n in selected if n >= start_from}
    if skip_risk:
        selected -= {10, 11}

    plan = []
    for step_num, script, extra_args in STEPS:
        if step_num not in selected:
            continue
        if step_num in MODEL_STEPS:
            for model in models:
                desc = f"Step {step_num:02d} ({script} --model {model})"
                plan.append((step_num, desc, [sys.executable, script, "--model", model]))
        else:
            desc = f"Step {step_num:02d} ({script})"
            plan.append((step_num, desc, [sys.executable, script] + extra_args))
    return plan


def run_step(desc, argv, dry_run):
    print(f"\n{'='*70}")
    print(f"{desc}")
    print(f"{'='*70}")
    if dry_run:
        print(f"  [dry-run] would run: {' '.join(argv)}")
        return True, 0.0

    start = time.time()
    result = subprocess.run(argv, cwd=HERE)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED ({desc}) after {elapsed:.1f}s -- exit code {result.returncode}")
        return False, elapsed

    print(f"\n  OK ({desc}) in {elapsed:.1f}s")
    return True, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the full EcoLens pipeline end-to-end.")
    parser.add_argument("--models", default="prithvi,vit,resnet",
                         help="Comma-separated models for steps 3/6 (default: all three).")
    parser.add_argument("--only", default=None,
                         help="Comma-separated step numbers to run, e.g. '3,6'. Default: all.")
    parser.add_argument("--start-from", type=int, default=None,
                         help="Skip steps before this number (e.g. --start-from 7 to resume).")
    parser.add_argument("--skip-risk", action="store_true",
                         help="Stop after step 9; skip steps 10-11 (grid tiling + risk model).")
    parser.add_argument("--predict", nargs=2, type=float, metavar=("LON", "LAT"), default=None,
                         help="After the run, score this location with the trained risk model "
                              "(requires steps 10-11 to have run at some point).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the commands that would run, without running them.")
    parser.add_argument("--continue-on-error", action="store_true",
                         help="Keep going after a step fails instead of stopping the whole run.")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    only = {int(x) for x in args.only.split(",")} if args.only else None

    plan = build_run_plan(models, only, args.start_from, args.skip_risk)
    if not plan:
        print("Nothing to run with the given filters.")
        return

    print(f"EcoLens run_all.py -- {len(plan)} step(s) planned:")
    for _, desc, argv in plan:
        print(f"  - {desc}")

    results = []
    overall_start = time.time()

    for step_num, desc, argv in plan:
        ok, elapsed = run_step(desc, argv, args.dry_run)
        results.append((desc, ok, elapsed))
        if not ok and not args.continue_on_error:
            print("\nStopping (pass --continue-on-error to keep going past failures).")
            break

    if args.predict is not None and not args.dry_run:
        lon, lat = args.predict
        print(f"\n{'='*70}")
        print(f"Scoring ({lon}, {lat}) with the trained risk model")
        print(f"{'='*70}")
        subprocess.run([sys.executable, "11_forest_risk_forecast.py", "--predict", str(lon), str(lat)], cwd=HERE)

    total_elapsed = time.time() - overall_start

    print(f"\n{'='*70}")
    print("RUN SUMMARY")
    print(f"{'='*70}")
    for desc, ok, elapsed in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {desc:<55} {elapsed:6.1f}s")
    n_failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n  {len(results)} step(s) run, {n_failed} failed, "
          f"total wall time {total_elapsed/60:.1f} min")
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()