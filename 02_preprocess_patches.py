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


def main():
    means, stds = load_prithvi_norm_stats(PRITHVI_CONFIG_PATH)

    print(f"Loaded Prithvi norm stats -- means: {means}, stds: {stds}")

    with open(METADATA_CATALOG_PATH) as f:
        catalog = json.load(f)

    processed_dir = f"{PATCHES_DIR}_processed"
    os.makedirs(processed_dir, exist_ok=True)

    updated_catalog = []

    for entry in catalog:
        raw_patch = np.load(entry["patch_path"])

        clean = handle_nodata(raw_patch)
        normalized = normalize(clean, means, stds)

        out_path = f"{processed_dir}/{entry['id']}_processed.npy"
        np.save(out_path, normalized)

        entry["processed_path"] = out_path
        entry["processed_shape"] = list(normalized.shape)
        updated_catalog.append(entry)

        print(f"[{entry['id']}] raw range=({raw_patch.min():.0f}, {raw_patch.max():.0f}) "
              f"-> normalized range=({normalized.min():.2f}, {normalized.max():.2f})")

    with open(METADATA_CATALOG_PATH, "w") as f:
        json.dump(updated_catalog, f, indent=2)

    print(f"\nProcessed {len(updated_catalog)} patches. Catalog updated.")


if __name__ == "__main__":
    main()
