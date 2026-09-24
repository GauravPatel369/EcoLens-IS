from setuptools import setup, find_packages

setup(
    name="ecolens",
    version="1.1.0",
    description="EcoLens: Ecosystem Analogs and Deforestation Risk Forecasting via Foundation Models",
    author="EcoLens Team",
    # NOTE (6 Sep): this was `find_packages()`, which correctly found NOTHING -- every
    # module sits at the repo root with a leading digit, so there were no packages and
    # `pip install .` shipped metadata with no code (top_level.txt was empty).
    # `ecolens/` is a thin importable wrapper that maps clean names onto the numbered
    # files; see ecolens/_loader.py for why renaming them was rejected.
    packages=["ecolens"],
    py_modules=["config", "geo_lookups", "inference", "run_phase", "prithvi_mae"],
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "timm",
        "pystac-client",
        "planetary-computer",
        "rasterio",
        "pyyaml",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "shap",
        "tqdm",
        "matplotlib"
    ],
    # NOTE (4 Sep, C25 audit): this pointed at `run_pipeline:main`, which begins by
    # rmtree-ing patches_processed/ and the embedding directories. Running the one command
    # the packaging advertised would have destroyed a large part of the 1,260 sub-crop
    # catalog -- and, because run_pipeline's list predates Clay and Satlas, it would have
    # left embeddings_clay/ and embeddings_satlas/ behind: a silently INCONSISTENT catalog
    # rather than a clean one. Repointed at the resumable runner, which destroys nothing.
    #
    # This does not make the package work -- find_packages() still finds no packages
    # because every module lives at the repo root with a leading digit (see
    # implementation.md C25). It only removes the data-loss hazard.
    entry_points={
        "console_scripts": [
            "ecolens-pipeline=run_phase:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
