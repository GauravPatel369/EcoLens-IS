"""
EcoLens Phase 1 -- Step 2: Preprocessing

Takes raw patches saved by 01_acquire_patches.py and prepares them
for Prithvi-100M inference:
  - Cast and clip reflectance values
  - Normalize using Prithvi's expected per-band mean/std
  - Handle missing/nodata pixels
  - Save as a cleaned tensor ready for model input

Run:
    python 02_preprocess_patches.py
"""

import json
import os
import numpy as np
import yaml

from config import PATCHES_DIR, METADATA_CATALOG_PATH, PRITHVI_BANDS

# Sentinel-2 L2A surface reflectance is stored as uint16, scaled by 10000.
# i.e. a raw value of 4500 means 0.45 reflectance.
REFLECTANCE_SCALE = 10000.0

# IMPORTANT: Prithvi-100M's per-band mean/std come from its own training
# config (data_mean / data_std in the released YAML), not from generic
# reflectance assumptions. Do NOT hardcode guessed values -- load them
# directly from the config shipped alongside the model checkpoint:
#
#   https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M
#   -> Prithvi_100M_config.yaml  (look for data_mean / data_std)
#
# Download that file once and point PRITHVI_CONFIG_PATH at it below.
# This script refuses to fall back to placeholder numbers, because wrong
# normalization stats silently produce bad embeddings rather than an
# obvious error -- exactly the kind of bug that's expensive to catch
# later in Phase 2 or 3.
#
# KNOWN LIMITATION: Prithvi-100M was trained on NASA HLS (Harmonized
# Landsat-Sentinel) data, which has different radiometric calibration
# than raw Sentinel-2 L2A imagery used in this pipeline. The HLS
# data_mean/data_std stats may not perfectly match Sentinel-2 L2A
# reflectance distributions, causing a subtle normalization mismatch
# that can degrade embedding quality. For best results, consider:
#   1. Using HLS data directly via NASA's LP DAAC, or
#   2. Computing Sentinel-2-specific mean/std from your own patches.

PRITHVI_CONFIG_PATH = "Prithvi_100M_config.yaml"  # download from HF repo


def load_prithvi_norm_stats(config_path):
    """
    Load the exact data_mean / data_std arrays Prithvi-100M was trained
    with, from its own released config file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"\n\n  Missing {config_path}.\n"
            f"  Download it from the Prithvi-100M HuggingFace repo:\n"
            f"  https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M\n"
            f"  (file: Prithvi_100M_config.yaml)\n"
            f"  and place it in this directory before running this script.\n"
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "train_params" in cfg:
        means = np.array(cfg["train_params"]["data_mean"], dtype=np.float32)
        stds = np.array(cfg["train_params"]["data_std"], dtype=np.float32)
    elif "pretrained_cfg" in cfg:
        means = np.array(cfg["pretrained_cfg"]["mean"], dtype=np.float32)
        stds = np.array(cfg["pretrained_cfg"]["std"], dtype=np.float32)
    else:
        means = np.array(cfg.get("data_mean", cfg.get("mean")), dtype=np.float32)
        stds = np.array(cfg.get("data_std", cfg.get("std")), dtype=np.float32)

    return means, stds


def to_reflectance(raw_patch):
    """Convert raw uint16 DN values to surface reflectance (0-1 range)."""
    return raw_patch / REFLECTANCE_SCALE


def handle_nodata(patch, nodata_value=0):
    """
    Replace nodata/zero pixels (common at scene edges or clouds)
    with the per-band median so they don't distort normalization.
    """
    clean = patch.copy()
    for b in range(patch.shape[0]):
        band = clean[b]
        mask = band == nodata_value
        if mask.any() and not mask.all():
            median_val = np.median(band[~mask])
            band[mask] = median_val
    return clean


def normalize(patch, means, stds):
    """Z-score normalize each band using Prithvi's training statistics."""
    # Clip raw DN values to 0-10000 range (corresponding to 0.0 - 1.0 reflectance)
    clipped = np.clip(patch, 0, 10000)
    normalized = (clipped - means[:, None, None]) / stds[:, None, None]
    return normalized.astype(np.float32)


def resize_patch_torch(patch_np, target_size=224):
    import torch
    import torch.nn.functional as F
    # patch_np shape: (channels, H, W)
    t = torch.from_numpy(patch_np).unsqueeze(0) # (1, C, H, W)
    resized = F.interpolate(t, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return resized.squeeze(0).numpy()


def calculate_offset_coords(lat, lon, dy, dx):
    import math
    lat_rad = math.radians(lat)
    # Latitude offset: 1 degree latitude ~ 110,540 meters
    d_lat = (dy * 10.0) / 110540.0
    # Longitude offset: 1 degree longitude ~ 111,320 * cos(lat) meters
    d_lon = (dx * 10.0) / (111320.0 * math.cos(lat_rad) + 1e-8)
    return round(lat + d_lat, 6), round(lon + d_lon, 6)


def main():
    means, stds = load_prithvi_norm_stats(PRITHVI_CONFIG_PATH)

    print(f"Loaded Prithvi norm stats -- means: {means}, stds: {stds}")

    with open(METADATA_CATALOG_PATH) as f:
        catalog = json.load(f)

    # Filter base patches to ensure idempotency (rerun safety)
    base_entries = []
    seen_base_ids = set()
    for entry in catalog:
        # If ID contains '_p', extract the base ID (e.g. 'forest_001' from 'forest_001_p0')
        base_id = entry["id"].split("_p")[0]
        if base_id not in seen_base_ids:
            seen_base_ids.add(base_id)
            # Reconstruct clean raw entry path references
            raw_entry = entry.copy()
            raw_entry["id"] = base_id
            raw_entry["patch_path"] = f"patches/{base_id}.npy"
            base_entries.append(raw_entry)

    processed_dir = f"{PATCHES_DIR}_processed"
    os.makedirs(processed_dir, exist_ok=True)

    updated_catalog = []
    crop_size = 160

    print(f"Expanding {len(base_entries)} base patches into 10 deterministic-random sub-crops each (710 total)...")

    for entry in base_entries:
        raw_path = entry["patch_path"]
        if not os.path.exists(raw_path):
            print(f"Warning: Raw patch {raw_path} not found. Skipping.")
            continue
            
        raw_patch = np.load(raw_path)

        # Set deterministic random seed per base patch using its base ID hash
        h_val = sum(ord(c) for c in entry["id"])
        rng = np.random.default_rng(h_val)

        # Generate 10 unique offsets (dy, dx) inside [-32, 32]
        offsets = []
        while len(offsets) < 10:
            dy = int(rng.integers(-32, 33))
            dx = int(rng.integers(-32, 33))
            if (dy, dx) not in offsets:
                offsets.append((dy, dx))

        for idx, (dy, dx) in enumerate(offsets):
            # Center of the main patch is (112, 112).
            # The top-left corner of the crop is at (32 + dy, 32 + dx)
            r = 32 + dy
            c = 32 + dx
            
            crop = raw_patch[:, r:r+crop_size, c:c+crop_size]
            resized_crop = resize_patch_torch(crop, target_size=224)

            clean = handle_nodata(resized_crop)
            normalized = normalize(clean, means, stds)

            sub_id = f"{entry['id']}_p{idx}"
            out_path = f"{processed_dir}/{sub_id}_processed.npy"
            np.save(out_path, normalized)

            # Offset coordinates physically
            sub_lat, sub_lon = calculate_offset_coords(entry["lat"], entry["lon"], dy, dx)

            # Create entry duplicate and update sub-crop properties
            sub_entry = entry.copy()
            sub_entry["id"] = sub_id
            sub_entry["lat"] = sub_lat
            sub_entry["lon"] = sub_lon
            sub_entry["name"] = f"{entry['name']} (Patch #{idx+1})"
            sub_entry["processed_path"] = out_path
            sub_entry["processed_shape"] = list(normalized.shape)

            # Update model-specific path references
            sub_entry["embedding_path"] = f"embeddings/{sub_id}.npy"
            sub_entry["prithvi_embedding"] = f"embeddings/{sub_id}.npy"
            sub_entry["vit_embedding"] = f"embeddings_vit/{sub_id}.npy"
            sub_entry["resnet_embedding"] = f"embeddings_resnet/{sub_id}.npy"

            updated_catalog.append(sub_entry)

    with open(METADATA_CATALOG_PATH, "w") as f:
        json.dump(updated_catalog, f, indent=2)

    print(f"\nSuccessfully generated and processed {len(updated_catalog)} sub-crops. Catalog updated.")


if __name__ == "__main__":
    main()
