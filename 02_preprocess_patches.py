"""
EcoLens Phase 1 -- Step 2: Preprocessing

Takes raw patches saved by 01_acquire_patches.py and prepares them
for Prithvi-100M inference:
  - Cast and clip reflectance values
  - Normalize using custom Sentinel-2-specific per-band mean/std
    (computed from the acquired patches themselves -- see the
    "Normalization strategy" comment below for why)
  - Handle missing/nodata pixels
  - Expand each base location into 10 sub-crops via small random
    offsets (see "KNOWN LIMITATION" comment near the crop loop below)

KNOWN LIMITATION -- sub-crop overlap: the 10 sub-crops generated per
base location come from offsets within +/-32px of a 160px crop inside
a 224px canvas, which means adjacent sub-crops of the same location
overlap by roughly 60-80% of their pixels. This is a deliberate
tradeoff (it's the only way to get 10 distinct crops out of a single
224px acquired patch without re-querying the STAC catalog), but it
means retrieval evaluation must NOT treat two sub-crops of the same
base location as independent "relevant" matches for each other --
that measures near-duplicate detection, not ecosystem generalization.
07_evaluate_retrieval.py's group-aware evaluation mode handles this
by excluding same-base_id sub-crops from the candidate pool entirely.
See README.md -> "Evaluation methodology" for the full discussion.

Run:
    python 02_preprocess_patches.py
"""

import json
import os
import numpy as np

from config import PATCHES_DIR, METADATA_CATALOG_PATH, PRITHVI_BANDS

# Sentinel-2 L2A surface reflectance is stored as uint16, scaled by 10000.
# i.e. a raw value of 4500 means 0.45 reflectance.
REFLECTANCE_SCALE = 10000.0

# ---------------------------------------------------------------
# Normalization strategy -- read this before changing it
# ---------------------------------------------------------------
# Prithvi-100M was pretrained on NASA HLS (Harmonized Landsat-Sentinel)
# data with its own released data_mean/data_std (see config.yaml's
# train_params). HLS and raw Sentinel-2 L2A have different radiometric
# calibration, so those HLS stats are a mismatched fit for the
# Sentinel-2 imagery this pipeline actually acquires.
#
# This script deliberately does NOT use Prithvi's HLS stats. Instead,
# compute_custom_norm_stats() below computes mean/std directly from
# the acquired Sentinel-2 patches themselves and normalizes with those.
# This is a domain-adaptation choice, not an oversight -- it keeps the
# input distribution's scale/center reasonable for a ViT-style encoder
# even though it isn't the exact distribution Prithvi was pretrained
# on. It does NOT fully solve the HLS-vs-Sentinel-2 mismatch (the
# model's learned features were still shaped by HLS statistics during
# pretraining), so Prithvi's absolute embedding quality on this data
# should still be interpreted with that caveat -- see README.md's
# "Known limitations" section for the full discussion, including the
# separate (and larger) num_frames fix in 03_extract_embeddings.py.
#
# If you'd rather use Prithvi's original HLS stats for a controlled
# comparison, load config.yaml's train_params.data_mean/data_std
# yourself and pass them to normalize() instead of the custom stats
# below -- but don't silently mix the two approaches across runs.


# NOTE: an earlier version of this script had a load_prithvi_norm_stats()
# function that loaded Prithvi's HLS training stats from a downloaded
# Prithvi_100M_config.yaml, but main() never actually called it -- it
# called compute_custom_norm_stats() instead. That left the module
# docstring and the real behavior contradicting each other. The dead
# function has been removed; see the block comment above for the
# actual (and only) normalization path this script uses. If you want
# Prithvi's original HLS stats for a controlled comparison, load
# config.yaml's train_params.data_mean/data_std directly -- that file
# is already part of this project (see config.yaml).


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


def compute_custom_norm_stats(base_entries):
    """
    Compute mean and std for each of the 6 bands across all available raw patches.
    Excludes zero (nodata) pixels to prevent skewing stats.
    """
    band_values = {b: [] for b in range(6)}
    for entry in base_entries:
        raw_path = entry["patch_path"]
        if os.path.exists(raw_path):
            patch = np.load(raw_path)
            patch = np.clip(patch, 0, 10000)
            for b in range(6):
                band_data = patch[b].flatten()
                non_zero = band_data[band_data > 0]
                if len(non_zero) > 0:
                    band_values[b].append(non_zero)
                else:
                    band_values[b].append(band_data)

    means = []
    stds = []
    for b in range(6):
        all_pixels = np.concatenate(band_values[b])
        means.append(float(np.mean(all_pixels)))
        stds.append(float(np.std(all_pixels)))
        
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def main():
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

    # Compute custom mean/std stats directly from the raw Sentinel-2 patches
    print("Computing custom mean and standard deviation from acquired Sentinel-2 patches...")
    means, stds = compute_custom_norm_stats(base_entries)
    print(f"Calculated Custom Sentinel-2 norm stats -- means: {list(means)}, stds: {list(stds)}")

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
            # Store the base location id explicitly rather than relying on
            # downstream scripts to re-derive it via id.split("_p")[0],
            # which breaks if an ecosystem/location id ever contains "_p"
            # itself (audit report Bug #4). This is also what makes
            # group-aware evaluation in 07_evaluate_retrieval.py reliable.
            sub_entry["base_id"] = entry["id"]
            sub_entry["lat"] = sub_lat
            sub_entry["lon"] = sub_lon
            clean_name = entry["name"].split(" (Patch #")[0]
            sub_entry["name"] = f"{clean_name} (Patch #{idx+1})"
            # Needed so 03_extract_embeddings.py can regenerate the correct
            # RGB sub-crop for ViT/ResNet instead of reusing the shared base patch
            sub_entry["crop_offset"] = [dy, dx]
            sub_entry["crop_size"] = crop_size
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
