# EcoLens — Phase 1: Foundation & Data Pipeline

This repository contains the complete Phase 1 pipeline for **EcoLens: Explainable Ecosystem Retrieval Using Geospatial Foundation Models**. It takes raw geographic coordinates, downloads Sentinel-2 imagery, processes them, runs GPU-accelerated inference using the **NASA/IBM Prithvi-100M** Vision Transformer, and generates a searchable interactive ecosystem database.

---

## 1. Setup & Installation

### Step 1: Create Virtual Environment
Create and activate a local Python virtual environment to manage dependencies:
```powershell
# Create venv
python -m venv .venv

# Activate venv (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

### Step 2: Install Base Dependencies
Install core geospatial and modeling packages:
```powershell
pip install pystac-client planetary-computer rasterio numpy pyyaml einops timm huggingface_hub
```

### Step 3: Install GPU-Enabled PyTorch (Optional but Recommended)
If you have an NVIDIA GPU, install PyTorch with CUDA support to accelerate embedding extraction (replaces CPU-only PyTorch):
```powershell
# Re-install PyTorch & Torchvision with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```
Verify GPU status:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

---

## 2. Model & Architecture Acquisition
You **do not need** to manually clone `hls-foundation-os` or manually download model files. The extraction pipeline is configured to **auto-download** the following files from the Hugging Face repository `ibm-nasa-geospatial/Prithvi-EO-1.0-100M` upon execution:
- `Prithvi_100M.pt` (Pretrained weights, ~450MB)
- `config.yaml` / `Prithvi_100M_config.yaml` (Model architecture configuration & norm parameters)
- `prithvi_mae.py` (Self-contained model architecture classes)

---

## 3. End-to-End Pipeline Execution

Run these scripts sequentially. Each script feeds its output into the next stage of the database:

### Step 1: Data Acquisition
```powershell
python 01_acquire_patches.py
```
- **What it does:** Queries the Microsoft Planetary Computer STAC catalog for the clearest Sentinel-2 scene over your configured coordinates in `config.py` (75 patches). Crops a 224x224 patch and resamples bands to 10m.
- **Optimization:** Automatically checks if a patch is already downloaded and skips STAC requests/downloads for existing files to save bandwidth.

### Step 2: Normalization & Preprocessing
```powershell
python 02_preprocess_patches.py
```
- **What it does:** Clips raw values to `[0, 10000]`, fills cloud/nodata pixels using the per-band median, and normalizes coordinates using the exact `data_mean`/`data_std` training statistics loaded from `Prithvi_100M_config.yaml`.
- **Output:** Standardized normalized tensors saved in `patches_processed/`.

### Step 3: Neural Network Embedding Extraction
```powershell
python 03_extract_embeddings.py
```
- **What it does:** Loads the Prithvi-100M model onto your device (automatically detects and runs on `cuda` if available, otherwise falls back to `cpu`). Projects the patches through `model.forward_features()` and pools the patch tokens into 768-dimensional embeddings.
- **Optimization:** Caches extracted vectors and skips inference for patches that already have `.npy` embeddings saved.

### Step 4: Finalization & Similarity Validation
```powershell
python 04_finalize_and_analyze.py
```
- **What it does:** Computes pairwise cosine similarity between all 71 successfully acquired patches and validates that the same-ecosystem similarity outperforms cross-ecosystem pairs.

### Step 5: Database and Visual Dashboard Generation
```powershell
python 05_create_database_and_dashboard.py
```
- **What it does:** Performs Principal Component Analysis (PCA) directly in NumPy to project the 768D embeddings into a 2D space, builds the cosine similarity search index, and exports a standalone dashboard `embedding_dashboard.html`.

---

## 4. Explore the Database Dashboard

To open and explore the searchable ecosystem database:
1. Start a local HTTP server inside the workspace directory:
   ```powershell
   python -m http.server 8000
   ```
2. Open your web browser and navigate to:
   ```text
   http://localhost:8000/embedding_dashboard.html
   ```

### Features:
- **Interactive 2D Space Map:** Move and click points on the scatter plot.
- **Category Filtering:** Filter points by Forest, Wetland, Mangrove, Agri, or Urban Green.
- **Ecosystem Search:** Select any query patch to instantly trigger a cosine-similarity nearest-neighbor search, returning similar ecological analogs across the globe ranked by score.

---

## 5. Objective 2 — Ecosystem Similarity Retrieval Framework

These scripts build on the Phase 1 embeddings to implement a full retrieval framework.

### Step 6: Retrieval Engine
```powershell
python 06_retrieval_engine.py
```
- **What it does:** Implements three similarity measures (Cosine Similarity, Euclidean Distance, k-Nearest Neighbor). For every patch, retrieves and ranks all other patches by each method. Builds an ecosystem analog database.
- **Output:** `results/retrieval_results.json`, `results/analog_database.json`

### Step 7: Retrieval Evaluation
```powershell
python 07_evaluate_retrieval.py
```
- **What it does:** Evaluates retrieval quality using standard IR metrics (Precision@K, Recall@K, mAP, MRR). Analyzes per-category performance for Forest, Wetland, Mangrove, Agricultural, and Urban Green ecosystems. Compares all three similarity methods. Produces a confusion matrix.
- **Output:** `results/evaluation_report.json` + console-printed tables

### Step 8: Retrieval Dashboard
```powershell
python 08_retrieval_dashboard.py
```
- **What it does:** Generates a standalone interactive HTML dashboard (`retrieval_dashboard.html`) with multi-method search, per-category performance charts, and a confusion matrix heatmap.
- **Output:** `retrieval_dashboard.html`

### Explore the Retrieval Dashboard
```text
http://localhost:8000/retrieval_dashboard.html
```

#### Retrieval Dashboard Features:
- **Method Selector:** Switch between Cosine, Euclidean, and kNN similarity
- **Per-Category Performance:** Bar chart comparing mAP across ecosystem types
- **Confusion Matrix:** Heatmap showing retrieval confusion between categories
- **Enhanced Search:** Multi-method ecosystem similarity search with ranked results

