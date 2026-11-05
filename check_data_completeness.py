"""
EcoLens data-completeness guard -- run automatically by run_phase.py after 10_grid_tiling_labels.py.

WHY THIS EXISTS
---------------
Two silent data defects make the analog-transfer result (step 14) look FALSELY significant. Both
work the same way: they cripple the DRIVER baseline, so the analog features get credit for filling
the gap rather than for genuine analog transfer.

  1. Climate columns (temp_c, rainfall_mm) mostly EMPTY -- a misnamed or missing WorldClim raster,
     so the per-cell climate lookup returns NaN for most cells. Climate is the single most
     important driver group, so losing it inflates any added feature.
  2. FEWER regions than configured -- unreadable/missing Hansen GFC tiles drop whole regions.

Either one invalidates the ablation. This guard FAILS the pipeline loudly (exit 1) so a run cannot
proceed to the risk/analog steps on broken inputs, and tells you exactly how to fix it.

It checks temp_c and rainfall_mm ONLY. It does NOT check elevation_m: the DEM is looked up on
demand and is legitimately sparse in this project (~31% filled even on a correct run), so a low
elevation fill is expected and is not a completeness signal.

Exit 0 = OK (safe to continue), exit 1 = incomplete (pipeline stops).
"""
import sys

import pandas as pd

from config import RISK_FEATURES_PATH, risk_regions

CLIMATE_COLS = ["temp_c", "rainfall_mm"]
MIN_CLIMATE_FILL = 0.95


def main():
    try:
        expected_regions = len(risk_regions())
    except Exception:
        expected_regions = 50

    df = pd.read_csv(RISK_FEATURES_PATH)
    problems = []

    n_regions = df["region_id"].nunique() if "region_id" in df.columns else 0
    if n_regions < expected_regions:
        problems.append(
            f"only {n_regions}/{expected_regions} regions present. Likely unreadable/missing "
            "Hansen GFC-2023-v1.11 tiles -- delete the bad tiles under data/hansen and re-run "
            "10_grid_tiling_labels.py so they re-download."
        )

    for c in CLIMATE_COLS:
        if c not in df.columns:
            problems.append(f"column '{c}' is missing entirely.")
            continue
        fill = float(df[c].notna().mean())
        if fill < MIN_CLIMATE_FILL:
            problems.append(
                f"'{c}' is only {fill * 100:.0f}% filled (need >= {MIN_CLIMATE_FILL * 100:.0f}%). "
                "The WorldClim raster is misnamed/missing, so most cells get no climate value. "
                "Fix: run  python download_reference_data.py --worldclim  (it places the correct "
                "global file at config.WORLDCLIM_TEMP_PATH / WORLDCLIM_PRECIP_PATH), then re-run "
                "10_grid_tiling_labels.py."
            )

    print("=" * 74)
    print("DATA COMPLETENESS CHECK (driver features)")
    print(f"  file   : {RISK_FEATURES_PATH}")
    print(f"  regions: {n_regions}/{expected_regions}")
    for c in CLIMATE_COLS:
        if c in df.columns:
            print(f"  {c:<14} {df[c].notna().mean() * 100:5.1f}% filled")
    print("=" * 74)

    if problems:
        print("\nFAILED -- the driver data is incomplete:")
        for p in problems:
            print("  - " + p)
        print(
            "\nThe analog-transfer ablation (step 14) is NOT trustworthy on incomplete driver data:\n"
            "with climate missing, the analog features falsely compensate for the gap and produce a\n"
            "spurious 'significant' gain. Fix the above and re-run. Pipeline stopped."
        )
        return 1

    print("\nOK -- driver data complete; safe to run the risk/analog steps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
