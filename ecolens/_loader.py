"""Import machinery for EcoLens's numbered pipeline modules.

WHY THIS EXISTS
---------------
Every pipeline stage lives in a file whose name starts with a digit --
`01_acquire_patches.py`, `13_analog_risk_features.py`, and so on. The numbering is
load-bearing documentation: it states the order stages must run in, and it is referenced
throughout implementation.md, the logs, and the results filenames.

It is also un-importable. A Python identifier cannot begin with a digit, so
`import 13_analog_risk_features` is a syntax error. That is why the codebase reaches for
`importlib.util.spec_from_file_location` in 15 places, and why `pip install .` shipped
metadata and no code: `find_packages()` correctly found no packages.

Renaming all 30 modules would fix the import problem and break the numbering, every
documented command, and 15 dynamic-load call sites -- in a project whose results are
mid-flight. This module takes the other route: it maps clean, importable names onto the
numbered files at import time, so both spellings work and neither breaks.

    from ecolens import analog_risk_features        # importable API
    python 13_analog_risk_features.py               # unchanged, still the documented CLI

Modules are loaded LAZILY. Importing `ecolens` must stay cheap -- several stages pull in
torch, geopandas or rasterio at module scope, and eagerly importing all of them would cost
seconds and hundreds of MB for a caller who wanted one function.
"""

import importlib.util
import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# clean name -> numbered filename. Ordered as the pipeline runs.
MODULE_MAP = {
    "validate_locations":        "00_validate_locations.py",
    "acquire_patches":           "01_acquire_patches.py",
    "preprocess_patches":        "02_preprocess_patches.py",
    "extract_embeddings":        "03_extract_embeddings.py",
    "layer_ablation":            "03b_layer_ablation.py",
    "finalize_and_analyze":      "04_finalize_and_analyze.py",
    "create_database_and_dashboard": "05_create_database_and_dashboard.py",
    "retrieval_engine":          "06_retrieval_engine.py",
    "evaluate_retrieval":        "07_evaluate_retrieval.py",
    "evaluate_ecological_similarity": "07b_evaluate_ecological_similarity.py",
    "evaluate_baseline_retrieval":    "07c_evaluate_baseline_retrieval.py",
    "retrieval_case_studies":    "07d_retrieval_case_studies.py",
    "retrieval_dashboard":       "08_retrieval_dashboard.py",
    "explainability_engine":     "09_explainability_engine.py",
    "grid_tiling_labels":        "10_grid_tiling_labels.py",
    "forest_risk_forecast":      "11_forest_risk_forecast.py",
    "temporal_stability_analysis": "12_temporal_stability_analysis.py",
    "analog_risk_features":      "13_analog_risk_features.py",
    "analog_ablation":           "14_analog_ablation.py",
    "analog_trajectory_figure":  "15_analog_trajectory_figure.py",
    "derive_location_geography": "16_derive_location_geography.py",
    "cross_region_retrieval":    "17_cross_region_retrieval.py",
    "case_study_figures":        "18_case_study_figures.py",
    "dimension_sweep":           "19_dimension_sweep.py",
    "retrieval_diagnostics":     "20_retrieval_diagnostics.py",
    "stability_normalized":      "21_stability_normalized.py",
    "patch_size_sweep":          "22_patch_size_sweep.py",
    "drift_ablation_matched":    "23_drift_ablation_matched.py",
    "successional_stages":       "24_successional_stages.py",
    "horizon_sweep":             "25_horizon_sweep.py",
    "build_wdpa_layer":          "26_build_wdpa_layer.py",
}

# Plain modules that are already importable; re-exported for a uniform API.
PLAIN_MODULES = ("config", "geo_lookups", "inference", "run_phase", "prithvi_mae")

_cache = {}


def load(name):
    """Load a pipeline module by its clean name. Cached; safe to call repeatedly."""
    if name in _cache:
        return _cache[name]

    if name in PLAIN_MODULES:
        if PACKAGE_ROOT not in sys.path:
            sys.path.insert(0, PACKAGE_ROOT)
        mod = importlib.import_module(name)
        _cache[name] = mod
        return mod

    if name not in MODULE_MAP:
        raise AttributeError(
            f"'{name}' is not an EcoLens module. Available: "
            f"{', '.join(sorted(list(MODULE_MAP) + list(PLAIN_MODULES)))}"
        )

    path = os.path.join(PACKAGE_ROOT, MODULE_MAP[name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{MODULE_MAP[name]} not found at {path}")

    # Numbered modules import each other and `config` by plain name, so the repo root
    # has to be importable before we exec one.
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)

    spec = importlib.util.spec_from_file_location(f"ecolens._impl_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _cache[name] = mod
    return mod
