"""
EcoLens Objective 3 - Step 9: Ecosystem Explainability Engine

Extracts land-cover descriptors (forest, water, urban percentages,
vegetation health, and protected status) directly from raw Sentinel-2
patch bands. Computes pair-wise comparisons and generates natural language
explanations for similar query-analog pairs.

Run:
    python 09_explainability_engine.py
"""

import json
import os
import numpy as np

from config import METADATA_CATALOG_PATH, RESULTS_DIR, PATCH_LOCATIONS

OUT_DESCRIPTORS_PATH = f"{RESULTS_DIR}/ecosystem_descriptors.json"
OUT_EXPLAIN_PATH = f"{RESULTS_DIR}/explainable_retrieval.json"


def calculate_patch_descriptors(patch_path, protected_area):
    """
    Load raw 6-band patch array and compute:
      - Forest cover % (via NDVI)
      - Water cover % (via NDWI)
      - Urban/Built-up/Bare soil cover % (via NDBI & NDVI)
      - Vegetation Health (Mean NDVI of active vegetation)
      - Protected area status (from catalog)
    """
    if not os.path.exists(patch_path):
        return {
            "forest_cover": 0.0,
            "water_cover": 0.0,
            "urban_cover": 0.0,
            "veg_health": 0.0,
            "protected_area": protected_area,
        }

    raw = np.load(patch_path)
    
    # Handle all-zero nodata arrays safely
    if np.max(raw) == 0:
        return {
            "forest_cover": 0.0,
            "water_cover": 0.0,
            "urban_cover": 0.0,
            "veg_health": 0.0,
            "protected_area": protected_area,
        }

    # Scale to reflectance [0, 1]
    patch = raw.astype(np.float32) / 10000.0

    # Sentinel-2 bands mapping: 0=Blue, 1=Green, 2=Red, 3=NIR, 4=SWIR1, 5=SWIR2
    green = patch[1]
    red   = patch[2]
    nir   = patch[3]
    swir1 = patch[4]

    # Compute indexes
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndwi = (green - nir) / (green + nir + 1e-8)
    ndbi = (swir1 - nir) / (swir1 + nir + 1e-8)

    # Thresholding for Land Cover estimation (Percentages in 0-100)
    forest_mask = (ndvi > 0.45) & (ndwi < 0.1) & (ndbi < 0.1)
    forest_pct = float(np.mean(forest_mask) * 100.0)

    water_mask = (ndwi > 0.0)
    water_pct = float(np.mean(water_mask) * 100.0)

    # Built-up / Urban / Bare soil
    urban_mask = (ndbi > 0.0) & (ndvi < 0.35) & (ndwi < 0.0)
    urban_pct = float(np.mean(urban_mask) * 100.0)

    # Vegetation Health: Mean NDVI for pixels with active vegetation (NDVI > 0.2)
    veg_active = ndvi[ndvi > 0.2]
    veg_health = float(np.mean(veg_active)) if len(veg_active) > 0 else 0.0

    return {
        "forest_cover": round(forest_pct, 2),
        "water_cover": round(water_pct, 2),
        "urban_cover": round(urban_pct, 2),
        "veg_health": round(veg_health, 4),
        "protected_area": protected_area,
    }


def generate_explanation(q, a):
    """
    Compare query and analog descriptors to generate a natural language
    explanation of their ecological similarity.
    """
    factors = []
    
    # Differences
    forest_diff = abs(q["forest_cover"] - a["forest_cover"])
    water_diff = abs(q["water_cover"] - a["water_cover"])
    urban_diff = abs(q["urban_cover"] - a["urban_cover"])
    veg_diff = abs(q["veg_health"] - a["veg_health"])
    
    # 1. Forest Canopy Cover Comparison
    if forest_diff < 15.0:
        if q["forest_cover"] > 50.0 and a["forest_cover"] > 50.0:
            factors.append(f"high forest canopy coverage ({q['forest_cover']:.1f}% vs {a['forest_cover']:.1f}%)")
        elif q["forest_cover"] < 10.0 and a["forest_cover"] < 10.0:
            factors.append(f"minimal forest canopy ({q['forest_cover']:.1f}% vs {a['forest_cover']:.1f}%)")
        else:
            factors.append(f"similar forest cover ({q['forest_cover']:.1f}% vs {a['forest_cover']:.1f}%)")

    # 2. Water Presence Comparison
    if water_diff < 12.0:
        if q["water_cover"] > 15.0 and a["water_cover"] > 15.0:
            factors.append(f"significant open water/wetland presence ({q['water_cover']:.1f}% vs {a['water_cover']:.1f}%)")
        elif q["water_cover"] < 2.0 and a["water_cover"] < 2.0:
            pass # Dry matches don't need highlighting unless dominant
        else:
            factors.append(f"comparable surface water profile ({q['water_cover']:.1f}% vs {a['water_cover']:.1f}%)")

    # 3. Urban / Bare soil influence
    if urban_diff < 15.0:
        if q["urban_cover"] > 25.0 and a["urban_cover"] > 25.0:
            factors.append(f"prominent built-up or bare ground structures ({q['urban_cover']:.1f}% vs {a['urban_cover']:.1f}%)")
        elif q["urban_cover"] < 5.0 and a["urban_cover"] < 5.0:
            factors.append(f"pristine surface conditions with minimal built-up footprint")
        else:
            factors.append(f"similar built-up/bare soil percentage ({q['urban_cover']:.1f}% vs {a['urban_cover']:.1f}%)")

    # 4. Vegetation Health (NDVI)
    if veg_diff < 0.08:
        if q["veg_health"] > 0.40 and a["veg_health"] > 0.40:
            factors.append(f"strong photosynthetic activity and vegetation health (mean NDVI of {q['veg_health']:.2f} vs {a['veg_health']:.2f})")
        else:
            factors.append(f"comparable vegetation activity indices")

    # 5. Protected Status
    if q["protected_area"] == a["protected_area"]:
        if q["protected_area"]:
            factors.append("shared status as designated conservation/protected areas")
        else:
            factors.append("similar unprotected conservation status")

    # Construct sentence
    if len(factors) == 0:
        explanation = "These two ecosystems share general ecological attributes with small variations in overall land cover profiles."
    elif len(factors) == 1:
        explanation = f"These two ecosystems are considered similar because both exhibit {factors[0]}."
    elif len(factors) == 2:
        explanation = f"These two ecosystems are considered similar because both exhibit {factors[0]} and {factors[1]}."
    else:
        explanation = f"These two ecosystems are considered similar because both exhibit {', '.join(factors[:-1])}, and {factors[-1]}."

    return {
        "explanation": explanation,
        "metrics": {
            "forest_diff": round(forest_diff, 2),
            "water_diff": round(water_diff, 2),
            "urban_diff": round(urban_diff, 2),
            "veg_health_diff": round(veg_diff, 4),
        }
    }


def main():
    print(f"\n{'='*60}")
    print("EcoLens Explainability Engine: Computing Ecosystem Descriptors")
    print(f"{'='*60}\n")

    if not os.path.exists(METADATA_CATALOG_PATH):
        print(f"Error: Catalog not found at {METADATA_CATALOG_PATH}. Run Phase 1 first.")
        return

    with open(METADATA_CATALOG_PATH) as f:
        catalog = json.load(f)

    # 1. Compute descriptors for all patches
    descriptors = {}
    print(f"Computing descriptors for {len(catalog)} patches...")
    
    for entry in catalog:
        pid = entry["id"]
        patch_path = entry.get("patch_path")
        protected = entry.get("protected_area", False)
        
        desc = calculate_patch_descriptors(patch_path, protected)
        descriptors[pid] = desc
        
        print(f"  [{pid:<15}] Forest: {desc['forest_cover']:>5.1f}% | Water: {desc['water_cover']:>5.1f}% | Urban: {desc['urban_cover']:>5.1f}% | NDVI: {desc['veg_health']:.4f}")

    # Save descriptors file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_DESCRIPTORS_PATH, "w") as f:
        json.dump(descriptors, f, indent=2)
    print(f"\nEcosystem descriptors saved to: {OUT_DESCRIPTORS_PATH}")

    # 2. Precompute comparisons for all 71 x 71 pairs
    print("\nPrecomputing pairwise similarity explanations...")
    explanations = {}
    
    pids = list(descriptors.keys())
    for qid in pids:
        explanations[qid] = {}
        for aid in pids:
            if qid == aid:
                continue
            
            explanations[qid][aid] = generate_explanation(descriptors[qid], descriptors[aid])

    # Save explainable retrieval results
    output_data = {
        "descriptors": descriptors,
        "explanations": explanations
    }
    
    with open(OUT_EXPLAIN_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Explainable similarity matrix saved to: {OUT_EXPLAIN_PATH}")
    print("\nDone. Run 08_retrieval_dashboard.py to integrate explanation reports.")


if __name__ == "__main__":
    main()
