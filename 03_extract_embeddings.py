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

    model_args = cfg["model_args"]
    model = PrithviMAE(**model_args)

    checkpoint = torch.load(PRITHVI_CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # Remove pos_embed keys to avoid loading mismatch for fixed/registered buffers
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
    """Extract embeddings using the Prithvi-100M geospatial model."""
    emb_dir = SUPPORTED_MODELS["prithvi"]["embeddings_dir"]
    os.makedirs(emb_dir, exist_ok=True)

    print(f"Loading Prithvi-100M on device: {DEVICE}")
    model = load_prithvi_model()
    print("Model loaded.")

    updated_catalog = []
    for entry in catalog:
        if "processed_path" not in entry:
            print(f"[{entry['id']}] No processed patch found -- run "
                  f"02_preprocess_patches.py first. Skipping.")
            continue

        out_path = f"{emb_dir}/{entry['id']}.npy"
        if "embedding_path" in entry and os.path.exists(out_path):
            print(f"[{entry['id']}] Embedding already exists. Skipping inference.")
            updated_catalog.append(entry)
            continue

        patch = np.load(entry["processed_path"])
        patch_tensor = torch.from_numpy(patch).float()

        try:
            embedding = extract_embedding(model, patch_tensor)
        except Exception as e:
            print(f"[{entry['id']}] Inference failed: {e}")
            continue

        np.save(out_path, embedding)

        entry["embedding_path"] = out_path
        entry["embedding_dim"] = int(embedding.shape[0])
        updated_catalog.append(entry)

        print(f"[{entry['id']}] embedding shape={embedding.shape}, "
              f"norm={np.linalg.norm(embedding):.3f}")

    return updated_catalog


def run_timm_model(catalog, model_key):
    """Extract embeddings using a timm model (ViT-Base or ResNet-50)."""
    model_cfg = SUPPORTED_MODELS[model_key]
    emb_dir = model_cfg["embeddings_dir"]
    timm_name = model_cfg["timm_name"]
    label = model_cfg["label"]
    os.makedirs(emb_dir, exist_ok=True)

    print(f"Loading {label} ({timm_name}) on device: {DEVICE}")
    model = load_timm_model(timm_name)
    print("Model loaded.")

    count = 0
    for entry in catalog:
        patch_path = entry.get("patch_path")
        if not patch_path or not os.path.exists(patch_path):
            print(f"[{entry['id']}] No raw patch found. Skipping.")
            continue

        out_path = f"{emb_dir}/{entry['id']}.npy"
        if os.path.exists(out_path):
            print(f"[{entry['id']}] Embedding already exists. Skipping.")
            count += 1
            continue

        raw_patch = np.load(patch_path)
        rgb_tensor = prepare_rgb_tensor(raw_patch)

        try:
            embedding = extract_timm_embedding(model, rgb_tensor)
        except Exception as e:
            print(f"[{entry['id']}] Inference failed: {e}")
            continue

        np.save(out_path, embedding)
        count += 1

        print(f"[{entry['id']}] embedding shape={embedding.shape}, "
              f"norm={np.linalg.norm(embedding):.3f}")

    print(f"\n{label}: Extracted/cached embeddings for {count} patches in {emb_dir}/")


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
        # Update the catalog file only for prithvi (backward compat)
        with open(METADATA_CATALOG_PATH, "w") as f:
            json.dump(updated_catalog, f, indent=2)
        print(f"\nExtracted embeddings for {len(updated_catalog)} patches.")
    else:
        run_timm_model(catalog, model_key)


if __name__ == "__main__":
    main()
