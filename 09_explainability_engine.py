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
from tqdm import tqdm

from config import METADATA_CATALOG_PATH, RESULTS_DIR, PATCH_LOCATIONS

OUT_DESCRIPTORS_PATH = f"{RESULTS_DIR}/ecosystem_descriptors.json"
OUT_EXPLAIN_PATH = f"{RESULTS_DIR}/explainable_retrieval.json"


def get_physical_descriptors(pid, ecosystem, name, lat, lon):
    """
    Generates heuristic-based physical descriptors for proof-of-concept purposes.

    WARNING: These values are ESTIMATED using latitude/ecosystem heuristics,
    NOT queried from real geospatial databases (e.g., SRTM for elevation,
    WorldClim for climate). For production use, replace with actual API calls
    to elevation services, climate datasets, and soil classification databases.
    """
    abs_lat = abs(lat)
    
    # Heuristics based hash for deterministic location values
    h_val = sum(ord(c) for c in pid)
    
    # 1. Elevation (meters)
    if "Himalaya" in name or "Andes" in name or "Alps" in name or "Kenya" in name:
        elevation = 2500.0 + (h_val % 10) * 120.0
    elif ecosystem in ["wetland", "mangrove"]:
        elevation = 2.0 + (h_val % 10) * 4.0
    elif "Cubbon" in name or "Bangalore" in name:
        elevation = 920.0
    else:
        elevation = 150.0 + (h_val % 10) * 65.0
        
    # 2. Temperature (Mean Annual °C)
    if abs_lat < 12.0:
        temp = 25.5 + (h_val % 10) * 0.3
    elif abs_lat < 23.5:
        temp = 20.0 + (h_val % 10) * 0.5
    elif abs_lat < 45.0:
        temp = 11.5 + (h_val % 10) * 0.4
    else:
        temp = 2.5 + (h_val % 10) * 0.6
        
    # 3. Rainfall (Mean Annual mm)
    if ecosystem == "mangrove" or "rainforest" in name.lower() or "congo" in name.lower():
        rainfall = 2200.0 + (h_val % 10) * 150.0
    elif ecosystem == "wetland":
        rainfall = 1100.0 + (h_val % 10) * 90.0
    elif abs_lat > 50.0:
        # High latitude boreal/temperate
        rainfall = 550.0 + (h_val % 10) * 45.0
    else:
        # Semi-arid or moderate temperate
        rainfall = 800.0 + (h_val % 10) * 110.0
        
    # 4. Soil Type
    if ecosystem == "mangrove":
        soil = "Saline Clay / Mud"
    elif ecosystem == "wetland":
        soil = "Gleysol / Histosol Peat"
    elif "rainforest" in name.lower() or "congo" in name.lower() or "amazon" in name.lower():
        soil = "Oxisol / Ferralsol"
    elif abs_lat > 50.0:
        soil = "Spodosol / Podzol"
    elif ecosystem == "agricultural":
        soil = "Luvisol / Vertisol"
    else:
        soil = "Alfisol / Cambisol"
        
    # 5. Ecoregion Classification
    if "Amazon" in name:
        ecoregion = "Amazonian Moist Forests"
    elif "Congo" in name:
        ecoregion = "Congolian Lowland Forests"
    elif "Ghats" in name or "Kerala" in name or "Periyar" in name:
        ecoregion = "South Western Ghats Moist Forests"
    elif "Sundarbans" in name:
        ecoregion = "Sundarbans Mangroves"
    elif "Everglades" in name:
        ecoregion = "Everglades Flooded Grasslands"
    elif "Germany" in name or "Poland" in name or "UK" in name:
        ecoregion = "European Mixed Broadleaf Forests"
    elif "Siberian" in name:
        ecoregion = "East Siberian Boreal Taiga"
    elif "Olympic" in name or "Redwood" in name:
        ecoregion = "Pacific Temperate Rainforests"
    elif "Kakadu" in name:
        ecoregion = "Arnhem Land Tropical Savanna"
    else:
        ecoregion = f"{ecosystem.capitalize()} Ecoregion"
        
    return {
        "elevation": round(float(elevation), 1),
        "temp": round(float(temp), 1),
        "rainfall": round(float(rainfall), 1),
        "soil": soil,
        "ecoregion": ecoregion
    }


def calculate_patch_descriptors(entry):
    """
    Load raw 6-band patch array and compute spectral descriptors,
    then merge with physical ecoregion descriptors.
    """
    patch_path = entry.get("patch_path")
    protected_area = entry.get("protected_area", False)
    
    # Naming heuristic for automatic WDPA verification
    name_str = entry.get("name", "").lower()
    if "national park" in name_str or "reserve" in name_str or "sanctuary" in name_str or "forest reserve" in name_str:
        protected_area = True

    # Base dictionary
    desc = {
        "forest_cover": 0.0,
        "water_cover": 0.0,
        "urban_cover": 0.0,
        "veg_health": 0.0,
        "protected_area": protected_area,
    }

    if patch_path and os.path.exists(patch_path):
        try:
            raw = np.load(patch_path)
            if np.max(raw) > 0:
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

                desc["forest_cover"] = round(forest_pct, 2)
                desc["water_cover"] = round(water_pct, 2)
                desc["urban_cover"] = round(urban_pct, 2)
                desc["veg_health"] = round(veg_health, 4)
        except Exception as e:
            print(f"Error reading patch {patch_path}: {e}")

    # Add dynamic ecoregion physical parameters
    phys = get_physical_descriptors(
        entry["id"], entry["ecosystem"], entry["name"], 
        entry["lat"], entry["lon"]
    )
    desc.update(phys)

    return desc


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
    
    elev_diff = abs(q["elevation"] - a["elevation"])
    temp_diff = abs(q["temp"] - a["temp"])
    rain_diff = abs(q["rainfall"] - a["rainfall"])
    
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
            factors.append(f"significant open water presence ({q['water_cover']:.1f}% vs {a['water_cover']:.1f}%)")
        elif q["water_cover"] < 2.0 and a["water_cover"] < 2.0:
            pass # Dry matches don't need highlighting unless dominant
        else:
            factors.append(f"comparable surface water profile ({q['water_cover']:.1f}% vs {a['water_cover']:.1f}%)")

    # 3. Urban / Bare soil influence
    if urban_diff < 15.0:
        if q["urban_cover"] > 25.0 and a["urban_cover"] > 25.0:
            factors.append(f"prominent built-up/exposed soil footprint ({q['urban_cover']:.1f}% vs {a['urban_cover']:.1f}%)")
        elif q["urban_cover"] < 5.0 and a["urban_cover"] < 5.0:
            factors.append(f"pristine surface conditions with minimal human footprint")
        else:
            factors.append(f"similar built-up/bare ground percentage ({q['urban_cover']:.1f}% vs {a['urban_cover']:.1f}%)")

    # 4. Vegetation Health (NDVI)
    if veg_diff < 0.08:
        if q["veg_health"] > 0.40 and a["veg_health"] > 0.40:
            factors.append(f"strong photosynthetic activity (mean NDVI of {q['veg_health']:.2f} vs {a['veg_health']:.2f})")
        else:
            factors.append(f"comparable vegetation vigor indices")

    # 5. Elevation & Climate matching
    if temp_diff < 3.5:
        factors.append(f"matching temperature profiles ({q['temp']:.1f}°C vs {a['temp']:.1f}°C)")
    if rain_diff < 300.0:
        factors.append(f"comparable annual precipitation ({q['rainfall']:.0f}mm vs {a['rainfall']:.0f}mm)")
    if elev_diff < 200.0:
        factors.append(f"similar altitude profiles ({q['elevation']:.0f}m vs {a['elevation']:.0f}m)")
        
    # 6. Ecoregion & Soil matching
    if q["ecoregion"] == a["ecoregion"]:
        factors.append(f"shared ecoregion designation ('{q['ecoregion']}')")
    if q["soil"] == a["soil"]:
        factors.append(f"matching soil profile ('{q['soil']}')")

    # 7. Protected Status
    if q["protected_area"] == a["protected_area"]:
        if q["protected_area"]:
            factors.append("shared conservation status as designated protected zones")
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
            "elev_diff": round(elev_diff, 1),
            "temp_diff": round(temp_diff, 1),
            "rain_diff": round(rain_diff, 1),
        }
    }


def main():
    print(f"\n{'='*60}")
    print("EcoLens Explainability Engine: Computing Ecosystem Descriptors")
    print(f"{'='*60}\n")

    if not os.path.exists(METADATA_CATALOG_PATH):
        print(f"Error: Catalog not found at {METADATA_CATALOG_PATH}. Run steps 1-3 first.")
        return

    with open(METADATA_CATALOG_PATH) as f:
        catalog = json.load(f)

    # 1. Compute descriptors for all patches
    descriptors = {}
    print(f"Computing descriptors for {len(catalog)} patches...")
    
    for entry in tqdm(catalog, desc="Ecosystem Descriptors Extraction"):
        desc = calculate_patch_descriptors(entry)
        descriptors[entry["id"]] = desc

    # Save descriptors file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_DESCRIPTORS_PATH, "w") as f:
        json.dump(descriptors, f, indent=2)
    print(f"\nEcosystem descriptors saved to: {OUT_DESCRIPTORS_PATH}")

    # 2. Precompute comparisons for similar pairs (top 50 candidate analogs per query)
    print("\nPrecomputing pairwise similarity explanations (top 50 candidate analogs per query)...")
    explanations = {}
    
    pids = list(descriptors.keys())
    for qid in tqdm(pids, desc="Generating Explanations"):
        explanations[qid] = {}
        q = descriptors[qid]
        
        # Calculate a simple distance metric for all candidates
        candidates = []
        for aid in pids:
            if qid == aid:
                continue
            a = descriptors[aid]
            dist = (
                (q["forest_cover"] - a["forest_cover"])**2 +
                (q["water_cover"] - a["water_cover"])**2 +
                (q["urban_cover"] - a["urban_cover"])**2 +
                (q["veg_health"] * 100 - a["veg_health"] * 100)**2
            )**0.5
            candidates.append((dist, aid))
            
        # Sort by distance and select the top 50 closest
        candidates.sort(key=lambda x: x[0])
        top_candidates = [aid for _, aid in candidates[:50]]
        
        for aid in top_candidates:
            explanations[qid][aid] = generate_explanation(q, descriptors[aid])

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
