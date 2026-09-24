"""EcoLens -- ecosystem analog retrieval and forest-loss risk forecasting.

Pipeline stages live in numbered files (`01_acquire_patches.py`, ...) because the numbering
documents the order they must run in. Those filenames are not importable -- a Python
identifier cannot start with a digit -- so this package exposes them under clean names
without renaming anything:

    import ecolens
    ecolens.retrieval_engine          # -> 06_retrieval_engine.py
    ecolens.analog_risk_features      # -> 13_analog_risk_features.py
    ecolens.config.PATCH_SIZE_PX

The command-line form documented throughout implementation.md is unchanged and remains the
supported way to run a stage:

    python 13_analog_risk_features.py --model clay

Modules load lazily: several stages import torch, geopandas or rasterio at module scope, so
`import ecolens` deliberately loads none of them until you name one.

`ecolens.stages()` lists what is available.
"""

from ._loader import MODULE_MAP, PLAIN_MODULES, load

__version__ = "1.1.0"
__all__ = sorted(list(MODULE_MAP) + list(PLAIN_MODULES)) + ["stages", "load"]


def stages():
    """Return {clean_name: filename} for every pipeline stage, in run order."""
    return dict(MODULE_MAP)


def __getattr__(name):
    # PEP 562 module-level __getattr__ -- this is what makes `ecolens.retrieval_engine`
    # resolve lazily instead of at package import.
    return load(name)


def __dir__():
    return __all__
