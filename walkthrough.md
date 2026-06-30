# Walkthrough - Multi-Model Retrieval Quality Comparison

This walkthrough summarizes the completed implementation, execution, evaluation, and dashboard verification for comparing ecosystem similarity retrieval across multiple foundation models.

---

## 1. Accomplishments & Changes

We have extended the EcoLens framework to support comparing the specialized **Prithvi-100M** model with general-purpose **ViT-Base** and **ResNet-50** models.

### Core Architecture Updates
*   **[config.py](file:///c:/IS/config.py)**: Added `SUPPORTED_MODELS` registry containing embedding dimensions and directories for all three models.
*   **[03_extract_embeddings.py](file:///c:/IS/03_extract_embeddings.py)**: Refactored to accept a `--model` flag. Added `timm` support for RGB-based models (ViT-Base and ResNet-50) using ImageNet normalization.
*   **[06_retrieval_engine.py](file:///c:/IS/06_retrieval_engine.py)**: Refactored to accept a `--model` flag and save outputs into model-specific files (e.g., `retrieval_results_resnet.json`).
*   **[07_evaluate_retrieval.py](file:///c:/IS/07_evaluate_retrieval.py)**: Refactored to auto-discover all model results and print unified cross-model comparison tables.
*   **[08_retrieval_dashboard.py](file:///c:/IS/08_retrieval_dashboard.py)**: Added a **Model Selector** dropdown, multi-model evaluation charts, and cross-model comparison bar charts.

---

## 2. Evaluation Results (Performance Comparison)

Below is the consolidated evaluation comparison printed by `07_evaluate_retrieval.py` for **Cosine Similarity**:

| Model | mAP | MRR | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | **0.4266** | **0.6461** | **0.5211** | **0.4742** | **0.4535** | **0.3915** | **0.0394** | **0.1075** | **0.1711** | **0.2946** |
| **ViT-Base** | 0.3825 | 0.6016 | 0.4648 | 0.3662 | 0.3690 | 0.3690 | 0.0350 | 0.0828 | 0.1397 | 0.2794 |
| **Prithvi-100M** | 0.3303 | 0.6098 | 0.4789 | 0.3662 | 0.3352 | 0.3028 | 0.0364 | 0.0835 | 0.1275 | 0.2292 |

### Per-Category mAP Breakdown (Cosine Similarity)

| Category | Prithvi-100M | ResNet-50 | ViT-Base |
| :--- | :--- | :--- | :--- |
| **agricultural** | 0.2618 | 0.3324 | 0.3146 |
| **forest** | 0.3307 | **0.4441** | 0.3431 |
| **mangrove** | 0.2467 | 0.2568 | **0.2620** |
| **urban_green** | 0.5508 | **0.7343** | 0.6596 |
| **wetland** | 0.2603 | **0.3589** | 0.3322 |

> [!NOTE]
> **Key Analysis**: General-purpose ImageNet pre-trained models (specifically **ResNet-50** and **ViT-Base**) outperformed the specialized geospatial **Prithvi-100M** on this specific retrieval task. This is likely because the current retrieval tasks map to RGB-rich, highly textured categories (such as *urban green* and *forests*) where ImageNet pre-training provides strong general visual representations.

---

## 3. Verification Screenshots and Video

We verified the interactive dashboard using a browser subagent:
*   Initial page load with Prithvi stats (mAP 0.330) and cross-model comparison charts.
*   Selecting "ResNet-50" (stats updated to mAP 0.427) and "ViT-Base" (stats updated to mAP 0.382).
*   Selecting `forest_001` (Periyar Forest) and switching tabs from Cosine to Euclidean and kNN (rankings and scores updated dynamically).
*   Verified that no JavaScript errors were thrown.

### Visual Proof

````carousel
![Dashboard Initial Load](C:/Users/Gaurav/.gemini/antigravity-ide/brain/020e79ea-2698-4607-ba85-d808866065c5/dashboard_initial_1782468999780.png)
<!-- slide -->
![Dashboard Similarity Search](C:/Users/Gaurav/.gemini/antigravity-ide/brain/020e79ea-2698-4607-ba85-d808866065c5/dashboard_similarity_search_1782469072465.png)
<!-- slide -->
![Dashboard Demo Recording](C:/Users/Gaurav/.gemini/antigravity-ide/brain/020e79ea-2698-4607-ba85-d808866065c5/retrieval_dashboard_demo_1782468854224.webp)
````

---

## 4. Objective 3: Explainable Similarity Retrieval Framework

We successfully designed and implemented the explainability framework (Objective 3) to interpret ecosystem similarity by deriving land-cover indices directly from Sentinel-2 bands.

### Implementation Summary
1.  **Explainability Engine ([09_explainability_engine.py](file:///c:/IS/09_explainability_engine.py))**:
    *   **Reflectance Scaling**: Normalizes raw `uint16` Sentinel-2 arrays (`patches/*.npy`) to reflectance $[0.0 - 1.0]$.
    *   **Spectral Indexes**:
        *   **NDVI** (Normalized Difference Vegetation Index) for vegetation vigor.
        *   **NDWI** (Normalized Difference Water Index) for surface water presence.
        *   **NDBI** (Normalized Difference Built-up Index) for artificial surfaces and bare soil.
    *   **Land-Cover Descriptor Estimation**:
        *   **Forest Cover %**: Pixels where $NDVI > 0.45$, $NDWI < 0.1$, and $NDBI < 0.1$.
        *   **Water Cover %**: Pixels where $NDWI > 0.0$.
        *   **Urban/Bare Ground %**: Pixels where $NDBI > 0.0$, $NDVI < 0.35$, and $NDWI < 0.0$.
        *   **Vegetation Health**: Mean NDVI of vegetated pixels ($NDVI > 0.2$).
        *   **Protected Area**: Extracted from catalog metadata.
    *   **Explanation Generator**: Compares descriptors pairwise for all 71 patches. Identifies matching dominant features and constructs high-quality natural language explanations.
    *   **Serialized Outputs**: Saves `results/ecosystem_descriptors.json` and `results/explainable_retrieval.json`.
2.  **Dashboard Integration**:
    *   Updated [08_retrieval_dashboard.py](file:///c:/IS/08_retrieval_dashboard.py) to inject precomputed explainable reports.
    *   Added **Comparison & Explainability Card** to the UI sidebar.
    *   Users can click on any similar ecosystem analog in the rankings list to show a side-by-side comparison grid of all 5 ecological descriptors and the natural-language explanation.
    *   Added **"Set as Query Ecosystem"** button to allow users to pivot search queries dynamically to the analog.

### Verification
*   Ran the explainability engine and dashboard compilation.
*   Verified browser interactions using a browser subagent (screenshots captured and stored).

### Visual Proof (Explainability Integration)
![Ecological Comparison Report Screenshot](C:/Users/Gaurav/.gemini/antigravity-ide/brain/020e79ea-2698-4607-ba85-d808866065c5/updated_query_ecosystem_1782749919342.png)

