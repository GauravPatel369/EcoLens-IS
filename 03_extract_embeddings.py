"""
EcoLens Phase 1 -- Step 3: Model inference (Multi-Model)

Extracts embedding vectors from ecosystem patches using one of several
foundation models:
  - prithvi  : NASA/IBM Prithvi-100M (6-band geospatial ViT, 768D)
  - vit      : ViT-Base (ImageNet RGB, 768D)  via timm
  - resnet   : ResNet-50 (ImageNet RGB, 2048D) via timm

The Prithvi-100M pathway uses the full 6-band preprocessed patches.
ViT-Base and ResNet-50 use only the RGB bands from the *raw* patches
(indices 2=Red, 1=Green, 0=Blue), rescaled to [0,1] and normalized
with ImageNet statistics.

Run:
    python 03_extract_embeddings.py                # default: prithvi
    python 03_extract_embeddings.py --model vit
    python 03_extract_embeddings.py --model resnet
"""

import argparse
import json
import os
import numpy as np
import timm
import torch
import yaml

from config import (
    PATCHES_DIR, METADATA_CATALOG_PATH, EMBEDDINGS_DIR,
    SUPPORTED_MODELS, DEFAULT_MODEL, IMAGENET_MEAN, IMAGENET_STD,
)

PRITHVI_CHECKPOINT_PATH = "Prithvi_100M.pt"
PRITHVI_CONFIG_PATH = "Prithvi_100M_config.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_prithvi_model():
    """
    Load the pretrained Prithvi-100M encoder.
    """
    # Auto-download if files are missing
    from huggingface_hub import hf_hub_download

    repo_id = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"

    if not os.path.exists(PRITHVI_CHECKPOINT_PATH):
        print(f"Downloading {PRITHVI_CHECKPOINT_PATH} from Hugging Face...")
        hf_hub_download(repo_id=repo_id, filename="Prithvi_100M.pt", local_dir=".")

    if not os.path.exists(PRITHVI_CONFIG_PATH):
        print(f"Downloading config from Hugging Face...")
        hf_hub_download(repo_id=repo_id, filename="config.yaml", local_dir=".")
        # copy config.yaml to PRITHVI_CONFIG_PATH
        if os.path.exists("config.yaml") and PRITHVI_CONFIG_PATH != "config.yaml":
            import shutil
            shutil.copy("config.yaml", PRITHVI_CONFIG_PATH)

    if not os.path.exists("prithvi_mae.py"):
        print("Downloading prithvi_mae.py from Hugging Face...")
        hf_hub_download(repo_id=repo_id, filename="prithvi_mae.py", local_dir=".")

    from prithvi_mae import PrithviMAE

    with open(PRITHVI_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # IMPORTANT: config.yaml's model_args specifies num_frames=3, because
    # Prithvi-100M was pretrained on 3-timestep HLS sequences. This
    # pipeline runs *static* single-scene inference (T=1) -- we don't
    # have multi-date stacks per patch. If we build the model with
    # num_frames=3 as-is and then feed it T=1 input (as the previous
    # version of this function did), the model's temporal patch
    # embedding / positional embedding are sized for 3 frames while the
    # actual input has 1, which is a real shape/semantics mismatch, not
    # just a cosmetic one.
    #
    # The fix -- confirmed against IBM/NASA's own official Prithvi-100M
    # usage example -- is to override num_frames to 1 BEFORE
    # instantiating the model, so the model is *built* for single-frame
    # input from the start:
    #   https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
    #   (see the model's own inference notebook / demo code)
    #
    # Copy model_args so we never mutate the loaded cfg dict.
    model_args = dict(cfg["model_args"])
    model_args["num_frames"] = 1
    model = PrithviMAE(**model_args)

    checkpoint = torch.load(PRITHVI_CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # The checkpoint's pos_embed / decoder_pos_embed were trained for
    # num_frames=3 and are the wrong shape for our num_frames=1 model,
    # so we drop them from the state dict (strict=False) and let the
    # freshly-initialized (sinusoidal, non-learned) pos_embed for T=1
    # apply instead. This is expected, standard practice for this
    # model family -- Prithvi's positional embeddings are a fixed
    # sin/cos encoding recomputed from img_size/num_frames, not a
    # learned parameter, so there is nothing lost by rebuilding it for
    # the shape we actually use.
    for k in list(state_dict.keys()):
        if "pos_embed" in k:
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(DEVICE)
    return model


@torch.no_grad()
def extract_embedding(model, patch_tensor):
    """
    Run the encoder forward pass and return a single pooled embedding
    vector for the patch.

    patch_tensor: (channels, H, W) -- a single static patch, no time dim.
    We insert T=1 to match Prithvi's expected (B, C, T, H, W) format.

    At inference we use mask_ratio=0.0 so no patches are masked out --
    we want the encoder's full representation, not a reconstruction task.
    The CLS token is dropped and the remaining patch tokens are
    mean-pooled into one fixed-length vector, which is what FAISS
    will index in Phase 2.
    """
    x = patch_tensor.unsqueeze(0)        # add batch dim -> (1, C, H, W)
    x = x.unsqueeze(2)                   # add time dim  -> (1, C, 1, H, W)
    x = x.to(DEVICE)

    latent = model.forward_features(x)[-1]

    # latent shape: (1, num_patches + 1, embed_dim) -- index 0 is CLS token
    patch_tokens = latent[:, 1:, :]
    pooled = patch_tokens.mean(dim=1)    # (1, embed_dim)

    return pooled.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------
# Generic timm model helpers (ViT-Base, ResNet-50)
# ---------------------------------------------------------------

def load_timm_model(model_name):
    """
    Load a pretrained timm model in feature-extraction mode.
    Returns the model moved to DEVICE in eval mode.
    """
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    model.to(DEVICE)
    return model


def crop_and_resize_raw(raw_patch, dy, dx, crop_size, target_size=224):
    """
    Re-derive the exact same sub-crop that 02_preprocess_patches.py produced
    for this entry, from the full (shared) base patch array. Mirrors the
    crop/resize logic in that script so ViT/ResNet see the same sub-patch
    content as Prithvi does, instead of the full un-cropped base patch.
    """
    import torch
    import torch.nn.functional as F

    r = 32 + dy
    c = 32 + dx
    crop = raw_patch[:, r:r + crop_size, c:c + crop_size]

    t = torch.from_numpy(crop).unsqueeze(0)  # (1, C, H, W)
    resized = F.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).numpy()


def prepare_rgb_tensor(raw_patch):
    """
    Prepare a 3-channel (RGB) tensor from a raw 6-band Sentinel-2 patch.

    Band order in our raw patches (from config.PRITHVI_BANDS):
        0=Blue, 1=Green, 2=Red, 3=NIR, 4=SWIR1, 5=SWIR2

    We select R, G, B (indices 2, 1, 0), rescale from raw DN [0, 10000]
    to [0, 1], then apply ImageNet normalization.
    """
    # Extract RGB bands in R, G, B order
    rgb = raw_patch[[2, 1, 0], :, :]  # shape: (3, 224, 224)
    # Clip and rescale to [0, 1]
    rgb = np.clip(rgb, 0, 10000).astype(np.float32) / 10000.0

    # Normalize with ImageNet stats
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    rgb = (rgb - mean) / std

    return torch.from_numpy(rgb).float()


@torch.no_grad()
def extract_timm_embedding(model, rgb_tensor):
    """
    Run a timm model on a single RGB patch and return the pooled embedding.

    timm models with num_classes=0 return the pooled feature vector directly.
    """
    x = rgb_tensor.unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)
    features = model(x)                     # (1, embed_dim)
    return features.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def run_prithvi(catalog):
    """Extract embeddings using the Prithvi-100M geospatial model in batches."""
    emb_dir = SUPPORTED_MODELS["prithvi"]["embeddings_dir"]
    os.makedirs(emb_dir, exist_ok=True)

    print(f"Loading Prithvi-100M on device: {DEVICE}")
    model = load_prithvi_model()
    print("Model loaded.")

    # Filter out valid entries that need extraction
    entries_to_process = []
    for entry in catalog:
        if "processed_path" not in entry:
            print(f"[{entry['id']}] No processed patch found -- run 02_preprocess_patches.py first. Skipping.")
            continue
        out_path = f"{emb_dir}/{entry['id']}.npy"
        # Always store the embedding path in the entry
        entry["prithvi_embedding"] = out_path
        if os.path.exists(out_path):
            continue
        entries_to_process.append(entry)

    if entries_to_process:
        print(f"Extracting embeddings for {len(entries_to_process)} patches...")
        from tqdm import tqdm
        batch_size = 16
        
        # Batch loop
        for i in tqdm(range(0, len(entries_to_process), batch_size), desc="Prithvi Batch Extraction"):
            batch_entries = entries_to_process[i:i+batch_size]
            
            # Load and stack tensors
            tensors = []
            for entry in batch_entries:
                patch = np.load(entry["processed_path"])
                tensors.append(torch.from_numpy(patch).float().unsqueeze(1)) # (C, 1, H, W)
                
            batch_x = torch.stack(tensors).to(DEVICE) # (B, C, 1, H, W)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                    # forward_features expects (B, C, T, H, W)
                    latent = model.forward_features(batch_x)[-1] # (B, num_patches+1, embed_dim)
                    patch_tokens = latent[:, 1:, :]
                    pooled = patch_tokens.mean(dim=1).cpu().numpy() # (B, embed_dim)
                    
            # L2 Normalize and save
            for idx, entry in enumerate(batch_entries):
                emb = pooled[idx].astype(np.float32)
                # L2 normalize
                emb /= np.linalg.norm(emb) + 1e-8
                out_path = entry["prithvi_embedding"]
                np.save(out_path, emb)
    else:
        print("All Prithvi embeddings already exist.")

    return catalog


def run_timm_model(catalog, model_key):
    """Extract embeddings using a timm model (ViT-Base or ResNet-50) in batches."""
    model_cfg = SUPPORTED_MODELS[model_key]
    emb_dir = model_cfg["embeddings_dir"]
    timm_name = model_cfg["timm_name"]
    label = model_cfg["label"]
    os.makedirs(emb_dir, exist_ok=True)

    print(f"Loading {label} ({timm_name}) on device: {DEVICE}")
    model = load_timm_model(timm_name)
    print("Model loaded.")

    entries_to_process = []
    for entry in catalog:
        patch_path = entry.get("patch_path")
        if not patch_path or not os.path.exists(patch_path):
            print(f"[{entry['id']}] No raw patch found. Skipping.")
            continue
        out_path = f"{emb_dir}/{entry['id']}.npy"
        # Always store the embedding path in the entry
        entry[f"{model_key}_embedding"] = out_path
        if os.path.exists(out_path):
            continue
        entries_to_process.append(entry)

    if entries_to_process:
        print(f"Extracting {label} embeddings for {len(entries_to_process)} patches...")
        from tqdm import tqdm
        batch_size = 16
        
        for i in tqdm(range(0, len(entries_to_process), batch_size), desc=f"{label} Batch Extraction"):
            batch_entries = entries_to_process[i:i+batch_size]
            
            tensors = []
            for entry in batch_entries:
                raw_patch = np.load(entry["patch_path"])
                # IMPORTANT: patch_path points at the shared full base patch --
                # every sub-crop of a location has the same path. Without this,
                # ViT/ResNet would extract an identical RGB tensor for all 10
                # sub-crops of a location instead of the actual unique sub-crop.
                if "crop_offset" in entry:
                    dy, dx = entry["crop_offset"]
                    crop_size = entry.get("crop_size", 160)
                    raw_patch = crop_and_resize_raw(raw_patch, dy, dx, crop_size)
                rgb_tensor = prepare_rgb_tensor(raw_patch)
                tensors.append(rgb_tensor)
                
            batch_x = torch.stack(tensors).to(DEVICE) # (B, 3, H, W)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                    pooled = model(batch_x).cpu().numpy() # (B, embed_dim)
                    
            for idx, entry in enumerate(batch_entries):
                emb = pooled[idx].astype(np.float32)
                # L2 normalize
                emb /= np.linalg.norm(emb) + 1e-8
                out_path = entry[f"{model_key}_embedding"]
                np.save(out_path, emb)
    else:
        print(f"All {label} embeddings already exist.")

    return catalog


def main():
    parser = argparse.ArgumentParser(
        description="EcoLens Step 3: Extract ecosystem embeddings"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        choices=list(SUPPORTED_MODELS.keys()),
        help=f"Foundation model to use (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    model_key = args.model

    print(f"\n{'='*60}")
    print(f"EcoLens Embedding Extraction - {SUPPORTED_MODELS[model_key]['label']}")
    print(f"{SUPPORTED_MODELS[model_key]['description']}")
    print(f"{'='*60}\n")

    with open(METADATA_CATALOG_PATH) as f:
        catalog = json.load(f)

    if model_key == "prithvi":
        updated_catalog = run_prithvi(catalog)
    else:
        updated_catalog = run_timm_model(catalog, model_key)

    with open(METADATA_CATALOG_PATH, "w") as f:
        json.dump(updated_catalog, f, indent=2)

    print(f"\nExtracted and updated catalog file saved to {METADATA_CATALOG_PATH}")


if __name__ == "__main__":
    main()
