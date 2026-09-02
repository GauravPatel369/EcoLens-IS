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

    with open(PRITHVI_CONFIG_PATH, encoding="utf-8") as f:
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

    # Retrieve output from Block 8 (index -4) for optimal semantic features
    latent = model.forward_features(x)[-4]

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
# Clay (Clay Foundation Model v1.5) -- REAL weights
# ---------------------------------------------------------------
#
# HISTORY: same story as Satlas above -- this pipeline used to mock
# Clay by running plain ImageNet ViT-Base and saving it under the clay
# label (embeddings byte-identical to the "vit" model's output). This
# section replaces that with the real `claymodel` package (PyPI
# `claymodel==1.5.0`) and the real clay-v1.5.ckpt checkpoint.
#
# PACKAGING BUG WORKAROUND (environment setup, not a per-run step):
# the published `claymodel` wheel installs its modules flattened
# directly into site-packages (module.py, model.py, backbone.py, ...)
# with internal `from src.model import ...`-style absolute imports
# that assume a top-level `src` package -- which the wheel never
# actually creates. Confirmed by inspecting the wheel's own RECORD /
# top_level.txt (lists bare module names, not a `claymodel` package)
# and by the resulting ModuleNotFoundError. Fixed once, in this
# environment's site-packages, by relocating those files into a real
# `src/` package (`from src.module import ClayMAEModule` is what
# actually works here) -- not something this script can fix at import
# time, so if you're setting this up fresh, see README.md.
#
# BAND CHOICE: Clay is explicitly designed to accept an arbitrary
# band subset (see its own wall-to-wall tutorial, which uses only 4 of
# its 10 known Sentinel-2 bands) as long as each band's wavelength is
# supplied -- unlike Satlas's fixed 9-band MS checkpoint, this isn't a
# scope compromise. This project's 6 available bands map directly onto
# 6 of Clay's known Sentinel-2 band names from its own metadata.yaml:
#   Blue -> "blue", Green -> "green", Red -> "red",
#   NIR (B8A, narrow) -> "nir08", SWIR1 (B11) -> "swir16",
#   SWIR2 (B12) -> "swir22"
# (Clay's metadata.yaml also defines "rededge1/2/3" and the wide-NIR
# "nir" (B08), which this project doesn't have -- simply omitted, per
# Clay's own "any band subset" design, not approximated.)

CLAY_CHECKPOINT_PATH = "clay_checkpoint/clay-v1.5.ckpt"
CLAY_METADATA_PATH = "clay_checkpoint/metadata.yaml"
CLAY_PLATFORM = "sentinel-2-l2a"
CLAY_GSD = 10.0
CLAY_MODEL_SIZE = "large"  # paired with clay-v1.5.ckpt per Clay's own tutorial
# Order matches raw patch band order from config.PRITHVI_BANDS:
# 0=Blue, 1=Green, 2=Red, 3=NIR(B8A), 4=SWIR1(B11), 5=SWIR2(B12)
CLAY_BAND_NAMES = ["blue", "green", "red", "nir08", "swir16", "swir22"]

_clay_metadata_cache = None


def _load_clay_metadata():
    global _clay_metadata_cache
    if _clay_metadata_cache is None:
        with open(CLAY_METADATA_PATH, encoding="utf-8") as f:
            _clay_metadata_cache = yaml.safe_load(f)
    return _clay_metadata_cache


def load_clay_model():
    """Load the real Clay-v1.5 (large) MAE encoder from its official checkpoint."""
    if not os.path.exists(CLAY_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"{CLAY_CHECKPOINT_PATH} not found. Download it (~4.8GB) from "
            f"https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"
        )
    if not os.path.exists(CLAY_METADATA_PATH):
        raise FileNotFoundError(
            f"{CLAY_METADATA_PATH} not found. Download it from "
            f"https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml"
        )

    from src.module import ClayMAEModule

    model = ClayMAEModule.load_from_checkpoint(
        CLAY_CHECKPOINT_PATH,
        model_size=CLAY_MODEL_SIZE,
        metadata_path=CLAY_METADATA_PATH,
        dolls=[16, 32, 64, 128, 256, 768, 1024],
        doll_weights=[1, 1, 1, 1, 1, 1, 1],
        mask_ratio=0.0,   # no masking at inference -- we want the full representation
        shuffle=False,
    )
    model.eval()
    model.to(DEVICE)
    return model


def prepare_clay_datacube(raw_patch, lon, lat, scene_date_str, batch_size=1):
    """
    Build the input dict Clay's encoder expects (datacube), following
    the model's own wall-to-wall tutorial: per-band normalization from
    metadata.yaml, sin/cos-encoded acquisition time and lat/lon, each
    band's wavelength, and ground sample distance.

    raw_patch: (6, H, W) raw-DN array (same scale as Clay's own
    mean/std stats -- NOT pre-scaled to reflectance/[0,1] first).
    """
    metadata = _load_clay_metadata()
    band_meta = metadata[CLAY_PLATFORM]["bands"]

    mean = np.array([band_meta["mean"][b] for b in CLAY_BAND_NAMES], dtype=np.float32)
    std = np.array([band_meta["std"][b] for b in CLAY_BAND_NAMES], dtype=np.float32)
    waves = [band_meta["wavelength"][b] for b in CLAY_BAND_NAMES]

    pixels = raw_patch.astype(np.float32)
    pixels = (pixels - mean[:, None, None]) / std[:, None, None]

    import datetime
    import math
    try:
        date = datetime.datetime.fromisoformat(scene_date_str)
    except (TypeError, ValueError):
        date = datetime.datetime(2021, 6, 1)  # fallback: acquisition date missing from catalog
    week = date.isocalendar().week * 2 * math.pi / 52
    hour = getattr(date, "hour", 12) * 2 * math.pi / 24
    time_vec = [math.sin(week), math.cos(week), math.sin(hour), math.cos(hour)]

    lat_rad, lon_rad = lat * math.pi / 180, lon * math.pi / 180
    latlon_vec = [math.sin(lat_rad), math.cos(lat_rad), math.sin(lon_rad), math.cos(lon_rad)]

    return {
        "pixels": torch.from_numpy(pixels).unsqueeze(0),
        "time": torch.tensor([time_vec], dtype=torch.float32),
        "latlon": torch.tensor([latlon_vec], dtype=torch.float32),
        "gsd": torch.tensor(CLAY_GSD, dtype=torch.float32),
        "waves": torch.tensor(waves, dtype=torch.float32),
    }


@torch.no_grad()
def extract_clay_embedding(model, datacube):
    """
    Run Clay's encoder on a single-item datacube and return the class-
    token embedding -- the overall per-image embedding, same convention
    as Clay's own tutorial (`unmsk_patch[:, 0, :]`). mask_ratio=0.0 (set
    at model load) means no patches are masked, so index 0 is always the
    CLS token followed by every real patch token, none dropped.
    """
    for k in ("pixels", "time", "latlon", "waves"):
        datacube[k] = datacube[k].to(DEVICE)
    datacube["gsd"] = datacube["gsd"].to(DEVICE)
    unmsk_patch, _, _, _ = model.model.encoder(datacube)
    return unmsk_patch[:, 0, :].squeeze(0).cpu().numpy()


def run_clay(catalog):
    """Extract embeddings using the real Clay-v1.5 (large) encoder."""
    model_cfg = SUPPORTED_MODELS["clay"]
    emb_dir = model_cfg["embeddings_dir"]
    os.makedirs(emb_dir, exist_ok=True)

    print(f"Loading Clay-v1.5 ({CLAY_MODEL_SIZE}) on device: {DEVICE} -- this is a large model, may take a while to load.")
    model = load_clay_model()
    print("Model loaded.")

    entries_to_process = []
    for entry in catalog:
        patch_path = entry.get("patch_path")
        if not patch_path or not os.path.exists(patch_path):
            print(f"[{entry['id']}] No raw patch found. Skipping.")
            continue
        out_path = f"{emb_dir}/{entry['id']}.npy"
        entry["clay_embedding"] = out_path
        if os.path.exists(out_path):
            continue
        entries_to_process.append(entry)

    if entries_to_process:
        print(f"Extracting Clay embeddings for {len(entries_to_process)} patches "
              f"(processed one at a time -- Clay's datacube carries per-item time/latlon metadata)...")
        from tqdm import tqdm

        for entry in tqdm(entries_to_process, desc="Clay Extraction"):
            raw_patch = np.load(entry["patch_path"])
            if "crop_offset" in entry:
                dy, dx = entry["crop_offset"]
                crop_size = entry.get("crop_size", 160)
                raw_patch = crop_and_resize_raw(raw_patch, dy, dx, crop_size)

            datacube = prepare_clay_datacube(
                raw_patch, entry["lon"], entry["lat"], entry.get("scene_date")
            )
            emb = extract_clay_embedding(model, datacube).astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-8
            np.save(entry["clay_embedding"], emb)
    else:
        print("All Clay embeddings already exist.")

    return catalog


# ---------------------------------------------------------------
# Satlas (AllenAI SatlasPretrain) -- REAL weights
# ---------------------------------------------------------------
#
# HISTORY: this pipeline used to "mock" Satlas by running plain
# ImageNet ResNet-50 (via timm) and saving it under the satlas label --
# the embeddings were byte-identical to the "resnet" model's output.
# That made a Prithvi/ViT/ResNet/Clay/Satlas comparison table
# structurally incapable of ever showing what it claimed to show. This
# section replaces that with the real `satlaspretrain_models` package
# and real AllenAI weights.
#
# MODEL CHOICE: Satlas ships both an RGB-only and a 9-band
# multi-spectral ("MS") Sentinel-2 checkpoint. The MS checkpoint needs
# B04,B03,B02,B05,B06,B07,B08,B11,B12 -- but this project's patches
# (config.PRITHVI_BANDS) only carry 6 bands (Blue/Green/Red/B8A-narrow-
# NIR/SWIR1/SWIR2), missing the three red-edge bands (B05/B06/B07) and
# using the narrow NIR (B8A) instead of the wide NIR (B08) the MS
# checkpoint expects. Re-acquiring all 710 patches with the extra
# bands is a real option (see README "Next steps") but out of scope
# for this pass. The RGB checkpoint (Sentinel2_Resnet50_SI_RGB) only
# needs bands we already have (B04,B03,B02), so that's what's used
# here -- a real, disclosed scope choice, not a silent shortcut.

SATLAS_CHECKPOINT_ID = "Sentinel2_Resnet50_SI_RGB"
SATLAS_TCI_GAIN = 2.5  # see prepare_satlas_rgb_tensor() docstring


def load_satlas_model():
    """
    Load AllenAI's real SatlasPretrain Sentinel-2 ResNet-50 (RGB)
    backbone. Downloads the checkpoint (~215MB) from Hugging Face on
    first use via the satlaspretrain_models package; cached by that
    package afterward.
    """
    import satlaspretrain_models

    weights_manager = satlaspretrain_models.Weights()
    model = weights_manager.get_pretrained_model(
        SATLAS_CHECKPOINT_ID, fpn=False, head=None,
        device=("cuda" if DEVICE == "cuda" else "cpu"),
    )
    model.eval()
    model.to(DEVICE)
    return model


def prepare_satlas_rgb_tensor(raw_patch):
    """
    Prepare a 3-channel tensor for Satlas's Sentinel2_Resnet50_SI_RGB
    checkpoint.

    Band order Satlas expects: B04, B03, B02 (Red, Green, Blue) with
    mean=[0,0,0], std=[255,255,255] -- confirmed against AllenAI's own
    Normalization.md (https://github.com/allenai/satlas/blob/main/
    Normalization.md) and torchgeo's `_satlas_bands` /
    `_satlas_transforms` constants (torchgeo/models/swin.py). That
    normalization assumes the input is Sentinel-2's 8-bit TCI
    (true-color) product, i.e. already-visualized 0-255 imagery -- NOT
    raw L2A reflectance DN, which is what this pipeline actually has
    (see config.PRITHVI_BANDS, raw values ~0-10000).

    KNOWN LIMITATION (disclosed, not silently absorbed -- same spirit
    as the HLS-vs-Sentinel-2 caveat documented in
    02_preprocess_patches.py for Prithvi): lacking the real TCI
    product, this approximates it with the standard Sentinel-2
    true-color visualization stretch (reflectance * gain, gain=2.5 --
    the same default gain used by Sentinel Hub / EO Browser's
    true-color rendering), clipped to [0, 1]. This is a reasonable,
    documented approximation, not a pixel-identical match to what the
    checkpoint saw during pretraining -- treat Satlas results with
    that caveat in mind, exactly as Prithvi's results already carry
    the HLS-calibration caveat.
    """
    rgb = raw_patch[[2, 1, 0], :, :]  # B04, B03, B02 = Red, Green, Blue
    reflectance = np.clip(rgb, 0, None).astype(np.float32) / 10000.0
    tci_approx = np.clip(reflectance * SATLAS_TCI_GAIN, 0.0, 1.0)
    return torch.from_numpy(tci_approx).float()


@torch.no_grad()
def extract_satlas_embedding_batch(model, rgb_batch):
    """
    rgb_batch: (B, 3, H, W) tensor from prepare_satlas_rgb_tensor().
    The Satlas Model (fpn=False, head=None) returns the raw ResNet
    backbone's 4 multi-scale feature maps [layer1..layer4]; global-
    average-pool the deepest one (layer4, 2048 channels) into a single
    embedding vector per image, the same pooling convention timm's
    resnet50(num_classes=0) applies for the plain "resnet" model.
    """
    feats = model(rgb_batch.to(DEVICE))  # [layer1, layer2, layer3, layer4]
    layer4 = feats[-1]                   # (B, 2048, H/32, W/32)
    pooled = layer4.mean(dim=[2, 3])     # global average pool -> (B, 2048)
    return pooled.cpu().numpy()


def run_satlas(catalog):
    """Extract embeddings using the real Satlas Sentinel2_Resnet50_SI_RGB backbone."""
    model_cfg = SUPPORTED_MODELS["satlas"]
    emb_dir = model_cfg["embeddings_dir"]
    os.makedirs(emb_dir, exist_ok=True)

    print(f"Loading Satlas ({SATLAS_CHECKPOINT_ID}) on device: {DEVICE}")
    model = load_satlas_model()
    print("Model loaded.")

    entries_to_process = []
    for entry in catalog:
        patch_path = entry.get("patch_path")
        if not patch_path or not os.path.exists(patch_path):
            print(f"[{entry['id']}] No raw patch found. Skipping.")
            continue
        out_path = f"{emb_dir}/{entry['id']}.npy"
        entry["satlas_embedding"] = out_path
        if os.path.exists(out_path):
            continue
        entries_to_process.append(entry)

    if entries_to_process:
        print(f"Extracting Satlas embeddings for {len(entries_to_process)} patches...")
        from tqdm import tqdm
        batch_size = 16

        for i in tqdm(range(0, len(entries_to_process), batch_size), desc="Satlas Batch Extraction"):
            batch_entries = entries_to_process[i:i + batch_size]

            tensors = []
            for entry in batch_entries:
                raw_patch = np.load(entry["patch_path"])
                if "crop_offset" in entry:
                    dy, dx = entry["crop_offset"]
                    crop_size = entry.get("crop_size", 160)
                    raw_patch = crop_and_resize_raw(raw_patch, dy, dx, crop_size)
                tensors.append(prepare_satlas_rgb_tensor(raw_patch))

            batch_x = torch.stack(tensors)  # (B, 3, H, W)
            pooled = extract_satlas_embedding_batch(model, batch_x)  # (B, 2048)

            for idx, entry in enumerate(batch_entries):
                emb = pooled[idx].astype(np.float32)
                emb /= np.linalg.norm(emb) + 1e-8
                np.save(entry["satlas_embedding"], emb)
    else:
        print("All Satlas embeddings already exist.")

    return catalog


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
                    # Retrieve output from Block 8 (index -4) for optimal semantic features
                    latent = model.forward_features(batch_x)[-4] # (B, num_patches+1, embed_dim)
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

    with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    if model_key == "prithvi":
        updated_catalog = run_prithvi(catalog)
    elif model_key == "clay":
        updated_catalog = run_clay(catalog)
    elif model_key == "satlas":
        updated_catalog = run_satlas(catalog)
    else:
        updated_catalog = run_timm_model(catalog, model_key)

    with open(METADATA_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_catalog, f, indent=2)

    print(f"\nExtracted and updated catalog file saved to {METADATA_CATALOG_PATH}")


if __name__ == "__main__":
    main()
