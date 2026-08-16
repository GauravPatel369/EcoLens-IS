from setuptools import setup, find_packages

setup(
    name="ecolens",
    version="1.0.0",
    description="EcoLens: Ecosystem Analogs and Deforestation Risk Forecasting via Foundation Models",
    author="EcoLens Team",
    packages=find_packages(),
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
    entry_points={
        "console_scripts": [
            "ecolens-pipeline=run_pipeline:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
