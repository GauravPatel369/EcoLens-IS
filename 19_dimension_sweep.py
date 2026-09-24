"""
EcoLens Step 19 -- EMBEDDING DIMENSION SWEEP  (Phase 2; C9)

C9: "Justify the embedding layer / feature representation; does embedding dimension
influence retrieval?" The second half has never been tested, and it matters for a
specific reason: **ResNet-50 and Satlas are 2048-D while Prithvi and ViT are 768-D and
Clay is 1024-D.** Any cross-model comparison is confounded by that until someone puts
them on equal footing (implementation.md Phase 2).

Method: PCA-reduce every model's embeddings to 64 / 128 / 256 / 512 (and full), re-run
the same GROUPED retrieval protocol, and report mAP@5 at each dimension.

PCA IS FIT ON THE CANDIDATE POOL, WHICH IS A DELIBERATE, DECLARED CHOICE. Strictly, a
transform fit on all 1,260 vectors and then used to retrieve among them leaks a little
global structure. It is unsupervised (no labels touched) and identical across models, so
the COMPARISON is fair, but the absolute numbers should not be read as a clean
held-out estimate. Fitting per-fold would be the stricter design.

Writes results/dimension_sweep.json

Run:
    python 19_dimension_sweep.py
"""

import json
import os

import numpy as np

from config import METADATA_CATALOG_PATH, RESULTS_DIR, SUPPORTED_MODELS

OUT_PATH = f"{RESULTS_DIR}/dimension_sweep.json"
DIMS = [64, 128, 256, 512]
K = 5


def load_model(model_key):
    with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    key = "prithvi_embedding" if model_key == "prithvi" else f"{model_key}_embedding"
    bases, ecos, vecs = [], [], []
    for e in catalog:
        p = e.get(key)
        if not p or not os.path.exists(p):
            continue
        v = np.load(p).astype(np.float32)
        bases.append(e.get("base_id", e["id"].rsplit("_p", 1)[0]))
        ecos.append(e["ecosystem"])
        vecs.append(v)
    return np.array(bases), np.array(ecos), np.stack(vecs)


def average_precision(hits):
    n_rel, score = 0, 0.0
    for i, h in enumerate(hits, 1):
        if h:
            n_rel += 1
            score += n_rel / i
    return score / n_rel if n_rel else 0.0


def grouped_map(bases, ecos, X):
    """mAP@K under 07's GROUPED rule (a patch's own base location is never a candidate)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -2.0)
    aps = []
    for i in range(len(Xn)):
        mask = bases != bases[i]
        cand = np.where(mask)[0]
        if cand.size == 0:
            continue
        order = cand[np.argsort(-S[i, cand])][:K]
        aps.append(average_precision([ecos[j] == ecos[i] for j in order]))
    return float(np.mean(aps)) if aps else None


def main():
    from sklearn.decomposition import PCA

    report = {}
    print(f"\n{'='*78}")
    print("EcoLens 19: embedding dimension sweep (C9)")
    print(f"{'='*78}")
    print(f"GROUPED mAP@{K}. PCA fit on the candidate pool -- see module docstring.\n")

    hdr = f"{'model':<16}{'native':>8}" + "".join(f"{d:>9}" for d in DIMS) + f"{'full':>9}"
    print(hdr)
    print("-" * len(hdr))

    for mk in SUPPORTED_MODELS:
        try:
            bases, ecos, X = load_model(mk)
        except Exception as e:
            print(f"{mk:<16} skipped ({type(e).__name__})")
            continue
        native = X.shape[1]
        full = grouped_map(bases, ecos, X)
        row = {"native_dim": native, "full": full, "reduced": {}}
        cells = []
        for d in DIMS:
            if d >= native:
                cells.append("     -")
                continue
            Xr = PCA(n_components=d, random_state=0).fit_transform(X)
            m = grouped_map(bases, ecos, Xr)
            row["reduced"][str(d)] = m
            cells.append(f"{m:>9.4f}")
        report[mk] = row
        print(f"{SUPPORTED_MODELS[mk]['label']:<16}{native:>8}"
              + "".join(cells) + f"{full:>9.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nsaved {OUT_PATH}")
    print("\nWHAT TO LOOK FOR (C9):")
    print("  * If mAP is flat from 64-D to native, dimension is NOT what separates the")
    print("    models, and ResNet/Satlas's 2048-D is not an unfair advantage.")
    print("  * If the 2048-D models fall sharply when cut to 768-D, part of their")
    print("    ranking was capacity, and the cross-model table in 3.4 needs that caveat.")
    print("  * A model that IMPROVES when reduced is carrying noise in its tail")
    print("    dimensions -- which would make PCA a free win for the retrieval index.")


if __name__ == "__main__":
    main()
