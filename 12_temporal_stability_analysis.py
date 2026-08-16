"""
EcoLens Objective 2 - Step 12: Temporal Stability Analysis

Evaluates whether the foundation model embeddings are temporally robust.
Downloads a summer patch (July) and a winter patch (January) for a subset
of locations, extracts their embeddings, and computes cosine similarity.
High similarity indicates the model captures stable ecosystem traits
rather than transient seasonal artifacts.
"""

import os
import json
import numpy as np
import pystac_client
import planetary_computer
import torch
import warnings
from scipy.spatial.distance import cosine

# Suppress noisy STAC/Rasterio warnings
warnings.filterwarnings("ignore")

from config import (
    PC_STAC_URL, MAX_CLOUD_COVER, PATCH_SIZE_M, PATCH_SIZE_PX, 
    PRITHVI_BANDS, PATCH_LOCATIONS, SUPPORTED_MODELS, DEFAULT_MODEL
)
from importlib import import_module
acq = import_module("01_acquire_patches")
ext = import_module("03_extract_embeddings")

# Define the seasons
SEASONS = {
    "summer": "2023-07-01/2023-08-31",
    "winter": "2023-01-01/2023-02-28"
}

def search_seasonal_scene(catalog, lon, lat, date_range, buffer_deg=0.05):
    bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}},
    )
    items = list(search.items())
    if not items:
        return None
    return min(items, key=lambda i: i.properties["eo:cloud_cover"])

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Temporal Stability Analysis")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=list(SUPPORTED_MODELS.keys()))
    args = parser.parse_args()
    model_key = args.model
    
    print(f"\n{'='*70}")
    print(f"EcoLens Temporal Stability Analysis ({SUPPORTED_MODELS[model_key]['label']})")
    print(f"{'='*70}")
    print("Testing if embeddings remain consistent across Winter and Summer seasons.\n")
    
    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)
    
    if model_key == "prithvi":
        model = ext.load_prithvi_model()
    else:
        model = ext.load_timm_model(SUPPORTED_MODELS[model_key]["timm_name"])
    
    # We will test on a subset of locations to save time (1 from each ecosystem)
    tested_ecosystems = set()
    test_locations = []
    for loc in PATCH_LOCATIONS:
        if loc["ecosystem"] not in tested_ecosystems:
            test_locations.append(loc)
            tested_ecosystems.add(loc["ecosystem"])
            
    print(f"Testing temporal stability on {len(test_locations)} diverse locations...")
    
    results = []
    
    for loc in test_locations:
        print(f"\nLocation: {loc['name']} ({loc['ecosystem']})")
        
        # 1. Fetch Summer Patch
        item_summer = search_seasonal_scene(catalog, loc["lon"], loc["lat"], SEASONS["summer"])
        if not item_summer:
            print("  [Skip] No clear summer scene found.")
            continue
            
        # 2. Fetch Winter Patch
        item_winter = search_seasonal_scene(catalog, loc["lon"], loc["lat"], SEASONS["winter"])
        if not item_winter:
            print("  [Skip] No clear winter scene found.")
            continue
            
        print(f"  Summer Cloud Cover: {item_summer.properties['eo:cloud_cover']:.1f}%")
        print(f"  Winter Cloud Cover: {item_winter.properties['eo:cloud_cover']:.1f}%")
        
        try:
            # 3. Extract Patches
            patch_summer = acq.extract_patch(item_summer, loc["lon"], loc["lat"], PATCH_SIZE_M, PATCH_SIZE_PX, PRITHVI_BANDS)
            patch_winter = acq.extract_patch(item_winter, loc["lon"], loc["lat"], PATCH_SIZE_M, PATCH_SIZE_PX, PRITHVI_BANDS)
            
            # 4. Extract Embeddings
            if model_key == "prithvi":
                tensor_summer = torch.from_numpy(patch_summer).float()
                tensor_winter = torch.from_numpy(patch_winter).float()
                emb_summer = ext.extract_embedding(model, tensor_summer)
                emb_winter = ext.extract_embedding(model, tensor_winter)
            else:
                tensor_summer = ext.prepare_rgb_tensor(patch_summer)
                tensor_winter = ext.prepare_rgb_tensor(patch_winter)
                emb_summer = ext.extract_timm_embedding(model, tensor_summer)
                emb_winter = ext.extract_timm_embedding(model, tensor_winter)
                
            # Normalize embeddings to match downstream FAISS logic
            emb_summer /= np.linalg.norm(emb_summer) + 1e-8
            emb_winter /= np.linalg.norm(emb_winter) + 1e-8
                
            # 5. Compute Similarity
            sim = 1.0 - cosine(emb_summer, emb_winter)
            print(f"  --> Temporal Cosine Similarity: {sim:.4f}")
            
            results.append({
                "ecosystem": loc["ecosystem"],
                "similarity": float(sim)
            })
        except Exception as e:
            print(f"  [Error] Failed to process patches: {e}")
        
    print(f"\n{'='*70}")
    print("TEMPORAL STABILITY SUMMARY")
    print(f"{'='*70}")
    if not results:
        print("No valid paired seasons found for testing.")
        return
        
    avg_sim = np.mean([r["similarity"] for r in results])
    print(f"Overall Average Seasonal Similarity: {avg_sim:.4f}")
    
    for eco in sorted(set([r["ecosystem"] for r in results])):
        eco_sims = [r["similarity"] for r in results if r["ecosystem"] == eco]
        print(f"  {eco:<15}: {np.mean(eco_sims):.4f}")

if __name__ == "__main__":
    main()
