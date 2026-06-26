# EcoLens Phase 2: Ecosystem Similarity Retrieval Framework (Multi-Model)

This document outlines the architectural additions, objectives, and specific implementations introduced in Phase 2 (Objective 2) of the EcoLens project, including support for comparing multiple foundation models.

---

## 1. What We Added

To build a formal retrieval engine on top of Phase 1 embeddings, we introduced the following components to the repository:

1. **`06_retrieval_engine.py`**: The core search engine module, updated to support choosing between foundation models via `--model {prithvi,vit,resnet}`.
2. **`07_evaluate_retrieval.py`**: The evaluation and metrics module, which auto-discovers all model results and generates a unified comparative report.
3. **`08_retrieval_dashboard.py`**: The visual interactive UI generator. It builds a dashboard that compares cross-model performance and allows active model switching.
4. **Configuration Updates**: Added `SUPPORTED_MODELS` registry containing details (embedding dimensions, model paths, timm names) for **Prithvi-100M**, **ViT-Base**, and **ResNet-50** in `config.py`.
5. **Output Artifacts**: A `results/` directory containing model-specific retrieval results (`retrieval_results_<model>.json`), analog databases (`analog_database_<model>.json`), and a combined `evaluation_report.json`.

---

## 2. What We Are Doing & How We Are Doing It (Code Breakdown)

### Step 6: The Retrieval Engine (`06_retrieval_engine.py`)

**What we are doing:**
We are building an engine that takes a "query" ecosystem patch and searches the entire database to find the mathematically closest ecological analogs using three different similarity measures, across different foundation models.

**How we are doing it:**
1. **Loading Data**: The script loads the embedding vectors for the specified model from its designated directory (`embeddings/` for Prithvi, `embeddings_vit/` for ViT, or `embeddings_resnet/` for ResNet) along with `metadata/catalog.json`.
2. **Leave-One-Out Searching**: It loops through every single patch in the database. For each patch (acting as the "query"), it compares its embedding vector against all other patches in the database.
3. **Similarity Math**: The comparison is done using three strict mathematical formulas:
   - **Cosine Similarity**: Uses `numpy.dot()` and vector norms to measure the angle between two vectors. Returns a score up to 1.0.
   - **Euclidean Distance**: Uses `numpy.linalg.norm(a - b)` to measure the straight-line physical distance between vectors, which is then converted into a similarity score via `1 / (1 + distance)`.
   - **k-Nearest Neighbor (kNN)**: Finds the K closest patches by raw Euclidean distance, sorted in ascending order.
4. **Ranking & Saving**: For each method, the results are sorted from most similar to least similar. The complete ranked lists are saved to `results/retrieval_results_<model>.json` and the top-10 cosine analogs are saved as `results/analog_database_<model>.json`. Legacy compatibility is maintained by copying the selected model's output to the default files.

### Step 7: Retrieval Evaluation (`07_evaluate_retrieval.py`)

**What we are doing:**
We are grading the retrieval engine's accuracy using standard Information Retrieval (IR) metrics to see how well different foundation models group similar ecosystems together.

**How we are doing it:**
1. **Defining "Relevant" Ground Truth**: If a query patch belongs to `forest`, then only other patches labeled `forest` are considered correct search results.
2. **Computing Standard IR Metrics**:
   - **Precision@K (P@K)**: Calculates the percentage of the top-K retrieved items that are correct (same category).
   - **Recall@K (R@K)**: Calculates what percentage of total possible correct patches were successfully found in the top-K.
   - **Mean Reciprocal Rank (MRR)**: Evaluates the retrieval rank of the *first* correct result.
   - **Mean Average Precision (mAP)**: Aggregates precision at every position where a correct item is found, offering a single overall quality metric.
3. **Multi-Model Auto-Discovery & Aggregation**: The script reads all results matching `results/retrieval_results_*.json`. It computes metrics for each model, prints console comparison tables, and writes a combined summary to `results/evaluation_report.json`.

### Step 8: The Interactive Dashboard (`08_retrieval_dashboard.py`)

**What we are doing:**
We are generating an interactive web application ([retrieval_dashboard.html](file:///c:/IS/retrieval_dashboard.html)) to visually explore the similarity search results, compare models, and map the 2D projected embedding spaces.

**How we are doing it:**
1. **Dimensionality Reduction (PCA)**: For each model, the script uses Principal Component Analysis (`numpy.cov` and `numpy.linalg.eigh`) to project the high-dimensional embeddings (768D or 2048D) down to 2 dimensions for visual scatter plotting.
2. **Interactive UI Features**: 
   - **Model Selector**: Allows switching between Prithvi-100M, ViT-Base, and ResNet-50.
   - **Cross-Model Comparison**: Renders a Chart.js comparison chart showing mAP scores per category side-by-side across all models.
   - **Interactive Scatter Plot**: Plots PCA projections colored by ecosystem type; clicking points updates the details card and search list.
   - **Multi-Metric Search**: Lets users switch between Cosine, Euclidean, and kNN tabs dynamically.
   - **Heatmap Heat-Table**: Renders the confusion matrix as a table with background colors mapped to confusion proportions.
