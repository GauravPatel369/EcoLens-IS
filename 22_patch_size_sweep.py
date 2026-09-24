"""
EcoLens Step 22 -- PATCH-SIZE (GROUND FOOTPRINT) SENSITIVITY  (C22)

WHAT C22 ASKS AND WHAT THIS ANSWERS
-----------------------------------
C22 asks for sensitivity to **patch size**, and implementation.md has been recording that
as blocked on "re-acquisition at 128 px and 448 px". That is only half true:

  * LARGER footprints do need re-acquisition -- the pixels simply are not on disk.
  * SMALLER footprints do NOT. Every catalog patch is already 224 px, so a smaller ground
    footprint can be produced by centre-cropping and resizing back to 224. That holds the
    model's input size fixed and varies ONLY the area of ground the patch covers, which is
    exactly the variable C22 is asking about.

So this sweeps downward and reports it honestly as a one-sided answer.

WHAT IS ACTUALLY VARIED
-----------------------
A catalog patch is 224 px covering 1,600 m (02 centre-crops 160 of the acquired 2,240 m
patch, then resizes to 224). Cropping to S px and resizing back to 224 gives:

    footprint_m = 1600 * S / 224

    S = 224  ->  1600 m   (native -- the catalog itself)
    S = 160  ->  1143 m
    S = 112  ->   800 m
    S =  64  ->   457 m

Input resolution to the model is 224 px in every arm, so nothing changes except how much
ground each patch sees.

PREPROCESSING
-------------
Mirrors 03 exactly, per model -- prithvi from the 02 z-scored array, everything else from
the raw array. Getting this wrong is the recurring defect in this codebase (3.2, 3.2c), so
the arrays are read from the same catalog fields 03 reads.

Evaluation is 07's GROUPED protocol: a patch's own base location is never a candidate.

Writes results/patch_size_sweep.json

Run:
    python 22_patch_size_sweep.py --models resnet vit
    python 22_patch_size_sweep.py --models prithvi --limit 400
"""

import argparse
import json
import os
import sys

import numpy as np

from config import METADATA_CATALOG_PATH, RESULTS_DIR

OUT_PATH = f"{RESULTS_DIR}/patch_size_sweep.json"
K = 5
NATIVE_PX = 224
NATIVE_FOOTPRINT_M = 1600.0
SIZES = [224, 160, 112, 64]


def _mod(name, path):
    if name not in sys.modules:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


def average_precision(hits):
    n_rel, score = 0, 0.0
    for i, h in enumerate(hits, 1):
        if h:
            n_rel += 1
            score += n_rel / i
    return score / n_rel if n_rel else 0.0


def grouped_map(bases, ecos, X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -2.0)
    aps = []
    for i in range(len(Xn)):
        cand = np.where(bases != bases[i])[0]
        if cand.size == 0:
            continue
        order = cand[np.argsort(-S[i, cand])][:K]
        aps.append(average_precision([ecos[j] == ecos[i] for j in order]))
    return float(np.mean(aps)) if aps else None


def crop_resize(patch, size, pre):
    """Centre-crop to `size` px, then resize back to 224. size==224 is a no-op."""
    if size == NATIVE_PX:
        return patch
    c = (NATIVE_PX - size) // 2
    crop = patch[:, c:c + size, c:c + size]
    return pre.resize_patch_torch(crop, target_size=NATIVE_PX)


def main():
    ap = argparse.ArgumentParser(description="EcoLens 22: patch-size sensitivity (C22)")
    ap.add_argument("--models", nargs="+", default=["resnet", "vit"],
                    help="models to sweep. clay/prithvi are far slower per patch.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from tqdm import tqdm

    ext = _mod("_ext", "03_extract_embeddings.py")
    pre = _mod("_pre", "02_preprocess_patches.py")

    with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    report = {}
    for mk in args.models:
        # Same array 03 reads for this model -- see module docstring.
        field = "processed_path" if mk == "prithvi" else "patch_path"
        entries = [e for e in catalog
                   if e.get(field) and os.path.exists(e[field])]
        if args.limit:
            entries = entries[: args.limit]
        if not entries:
            print(f"[{mk}] no patches found under '{field}' -- skipped")
            continue

        bases = np.array([e.get("base_id", e["id"].rsplit("_p", 1)[0]) for e in entries])
        ecos = np.array([e["ecosystem"] for e in entries])

        print(f"\nloading {mk} ({len(entries)} patches from '{field}') ...")
        if mk == "prithvi":
            model = ext.load_prithvi_model()
        elif mk == "clay":
            model = ext.load_clay_model()
        elif mk == "satlas":
            model = ext.load_satlas_model()
        else:
            from config import SUPPORTED_MODELS
            model = ext.load_timm_model(SUPPORTED_MODELS[mk]["timm_name"])

        def embed(arr, e):
            if mk == "prithvi":
                return ext.extract_embedding(model, torch.from_numpy(arr).float())
            if mk == "clay":
                dc = ext.prepare_clay_datacube(arr, e.get("lon"), e.get("lat"),
                                               str(e.get("scene_date", ""))[:10] or None)
                return ext.extract_clay_embedding(model, dc)
            if mk == "satlas":
                t = ext.prepare_satlas_rgb_tensor(arr)
                t = t.unsqueeze(0) if t.dim() == 3 else t
                with torch.no_grad():
                    return model(t)[-1].mean(dim=[2, 3]).squeeze(0).cpu().numpy()
            return ext.extract_timm_embedding(model, ext.prepare_rgb_tensor(arr))

        rows = {}
        for size in SIZES:
            feats = []
            for e in tqdm(entries, desc=f"{mk} @{size}px", leave=False):
                arr = np.load(e[field])
                feats.append(np.asarray(embed(crop_resize(arr, size, pre), e),
                                        dtype=np.float32))
            m = grouped_map(bases, ecos, np.stack(feats))
            fp = NATIVE_FOOTPRINT_M * size / NATIVE_PX
            rows[str(size)] = {"crop_px": size, "footprint_m": round(fp, 1), "map_at_k": m}
            print(f"  {mk}  {size:>3}px  {fp:>7.0f} m footprint   mAP@{K} {m:.4f}")

        native = rows[str(NATIVE_PX)]["map_at_k"]
        for r in rows.values():
            r["delta_vs_native"] = (r["map_at_k"] - native) if native else None
            r["pct_vs_native"] = (100.0 * (r["map_at_k"] - native) / native) if native else None
        report[mk] = {"n_patches": len(entries), "source_field": field, "sizes": rows}

        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:      # save after each model
            json.dump(report, f, indent=2)

    print(f"\nsaved {OUT_PATH}")
    print("\nHOW TO READ THIS:")
    print("  Input size is 224 px in every arm; only the GROUND FOOTPRINT changes.")
    print("  A flat curve means the method is insensitive to footprint over this range,")
    print("  and the 2.24 km acquisition choice is not load-bearing. A steep one means")
    print("  footprint is a real hyperparameter and the LARGER half of the sweep")
    print("  (which needs re-acquisition) is worth the cost.")


if __name__ == "__main__":
    main()
