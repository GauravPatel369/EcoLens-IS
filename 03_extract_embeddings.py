"""
EcoLens Phase 1 -- Step 3: Model inference

Loads Prithvi-100M and runs it on every preprocessed patch to extract
a single embedding vector per patch. This is the core foundation-model
step that turns raw imagery into the representations EcoLens compares
in Phase 2/3.

IMPORTANT INPUT SHAPE NOTE:
Prithvi-100M is a *temporal* ViT -- it expects input shaped
(batch, channels, time, height, width), i.e. 5 dimensions, even for a
single timestep. A static patch needs an extra T=1 dimension inserted,
or the model's patch embedding layer will throw a shape mismatch.
The encoder output for a 224x224 patch is (1, 197, 768) for the CLS
token + 196 patch tokens at 768-dim (Prithvi-100M base config) --
double check this against your specific config.yaml model_args.

Prerequisites (one-time setup):
    pip install torch einops timm huggingface_hub pyyaml --break-system-packages
    git clone https://github.com/NASA-IMPACT/hls-foundation-os.git
    # download Prithvi_100M.pt and Prithvi_100M_config.yaml from:
    # https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M
    # place both files in this directory, and add hls-foundation-os to PYTHONPATH

Run:
    python 03_extract_embeddings.py
"""

import json
import os
import numpy as np
import torch
import yaml

from config import PATCHES_DIR, METADATA_CATALOG_PATH, EMBEDDINGS_DIR

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


def main():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    print(f"Loading Prithvi-100M on device: {DEVICE}")
    model = load_prithvi_model()
    print("Model loaded.")

    with open(METADATA_CATALOG_PATH) as f:
        catalog = json.load(f)

    updated_catalog = []

    for entry in catalog:
        if "processed_path" not in entry:
            print(f"[{entry['id']}] No processed patch found -- run "
                  f"02_preprocess_patches.py first. Skipping.")
            continue

        out_path = f"{EMBEDDINGS_DIR}/{entry['id']}.npy"
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

    with open(METADATA_CATALOG_PATH, "w") as f:
        json.dump(updated_catalog, f, indent=2)

    print(f"\nExtracted embeddings for {len(updated_catalog)} patches.")


if __name__ == "__main__":
    main()
