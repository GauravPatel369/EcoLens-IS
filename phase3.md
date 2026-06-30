# EcoLens Phase 3: Explainable Ecosystem Similarity Retrieval Framework

This document outlines the architectural additions, objectives, and specific implementations introduced in Phase 3 (Objective 3) of the EcoLens project, including the explainability engine and dashboard report visualizer.

---

## 1. What We Added

To build an explainable similarity interpretation layer on top of our retrieval engine, we introduced the following components to the repository:

1. **`09_explainability_engine.py`**: The core explainability script that parses raw 6-band Sentinel-2 image arrays, extracts physical ecosystem metrics, and precomputes pair-wise explanations.
2. **`results/ecosystem_descriptors.json`**: Extracted indicators (Forest %, Water %, Urban %, NDVI Health, Protected status) for all 71 patches.
3. **`results/explainable_retrieval.json`**: Pair-wise statistical comparisons and natural-language explanations for all query-analog combinations.
4. **Dashboard Integration (`08_retrieval_dashboard.py`)**: Modified to inject and display comparison cards and explanation text dynamically in the browser UI when ranking entries are clicked.

---

## 2. What We Did & How We Did It (Code Breakdown)

### Step 9: The Explainability Engine (`09_explainability_engine.py`)

**What we are doing:**
We are extracting deterministic, physical land-cover descriptors directly from Sentinel-2 spectral bands. Rather than using external APIs, we calculate standard ecological indexes pixel-by-pixel for each 224x224 patch.

**How we are doing it:**
1. **Reflectance Scaling**: The raw 16-bit integers (`patches/*.npy`) are normalized to physical reflectance values by scaling: `patch / 10000.0`.
2. **Spectral Band Mapping**: We extract key bands from the 6-channel arrays: Blue (ch 0), Green (ch 1), Red (ch 2), NIR (ch 3), and SWIR1 (ch 4).
3. **Index Computation**:
   * **NDVI** (Normalized Difference Vegetation Index): measures green leaf density.
     $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
   * **NDWI** (Normalized Difference Water Index - McFeeters): delineates surface water.
     $$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$
   * **NDBI** (Normalized Difference Built-up Index): highlights artificial structures and bare soil.
     $$\text{NDBI} = \frac{\text{SWIR1} - \text{NIR}}{\text{SWIR1} + \text{NIR}}$$
4. **Descriptor Estimation (Percentage thresholds)**:
   * **Forest Cover %**: Percentage of pixels satisfying $\text{NDVI} > 0.45$, $\text{NDWI} < 0.1$, and $\text{NDBI} < 0.1$ (dense green vegetation).
   * **Water Cover %**: Percentage of pixels where $\text{NDWI} > 0.0$ (open surface water).
   * **Urban / Bare Soil %**: Percentage of pixels where $\text{NDBI} > 0.0$, $\text{NDVI} < 0.35$, and $\text{NDWI} < 0.0$ (roads, buildings, exposed ground).
   * **Vegetation Health**: Average NDVI of all pixels with active vegetative growth ($\text{NDVI} > 0.2$).
   * **Protected Status**: Loaded from local database metadata.
5. **Explanation Generation**: Compare query descriptors ($Q$) and analog descriptors ($A$). If differences are small (e.g. $< 15\%$), and the characteristic is present, it is added as a matching factor. The engine combines these matching factors into fluid, human-readable explanations (e.g., *"These two ecosystems are considered similar because both exhibit high forest canopy coverage (80.8% vs 70.4%), comparable vegetation health, and shared status as designated protected areas"*).

---

### Dashboard Integration (`08_retrieval_dashboard.py`)

**What we are doing:**
We integrated the explainability reports directly into the EcoLens Explorer UI, providing side-by-side comparisons and natural-language summaries for any query-analog pair.

**How we are doing it:**
1. **Report Injection**: Injects the complete `results/explainable_retrieval.json` data as a JavaScript constant (`EXPLAIN_DATA`).
2. **Ecological Comparison Card**: Added a dedicated card to the sidebar right below the query details.
3. **Interactive Ranking Clicks**: When a user clicks a result in the "Top Similar Ecosystems" list:
   * It highlights the selection.
   * It displays the natural language explanation.
   * It renders a comparison grid showing the Query vs. Analog metrics (Forest %, Water %, Urban %, Veg Health, Protected Area).
4. **Pivot Feature**: Provides a "Set as Query Ecosystem" button to let users search the database using the clicked analog as the new query.

---

## 3. How the Models Performed (Performance & Accuracy Analysis)

Our quantitative evaluations across **71 patches** using Cosine similarity revealed distinct characteristics:

### Overall Accuracy Summary
* **ResNet-50 (mAP: 0.427, MRR: 0.646)**: **Best performer overall**. The first relevant analog is found at rank **1.5** on average. The top-1 analog is correct **52%** of the time.
* **ViT-Base (mAP: 0.383, MRR: 0.602)**: Moderate performer. Shows strong texture awareness.
* **Prithvi-100M (mAP: 0.330, MRR: 0.610)**: Weakest overall mAP, but maintains strong retrieval ranks (MRR 0.610).

### Category Analysis: What's Good, What's Bad?

1. **Urban Green (mAP: 0.734) — Excellent**
   * **Why**: High contrast between roads/buildings (high NDBI) and park vegetation (moderate NDVI) makes this category highly distinct. Models locate urban parks with near-perfect accuracy.
2. **Forests (mAP: 0.444) — Good**
   * **Why**: Large contiguous canopy textures are easily grouped by models, though they occasionally overlap with wetlands.
3. **Wetlands (mAP: 0.359) & Farmland (mAP: 0.332) — Moderate**
   * **Why**: Farmlands change colors and structures radically with seasons (harvested fields vs. green crops), and wetlands fluctuate between open water, reeds, and mud, creating overlap.
4. **Mangroves (mAP: 0.257) — Weak / Difficult**
   * **Why**: A mixed canopy of trees, muddy soils, and tidal water. Models struggle to distinguish them from freshwater swamps or regular coastal forests.

### Model Analysis: Why did ResNet-50 beat Prithvi-100M?
While Prithvi is pre-trained on multi-spectral satellite data, **ResNet-50** has a larger capacity and is pre-trained on **ImageNet** (natural photographs). ImageNet training forces models to learn highly detailed visual filters for texture, edge patterns, and colors. For Sentinel-2 patches, crop rows, forest canopy structures, and urban layouts represent high-frequency visual textures that ResNet maps with high precision.
