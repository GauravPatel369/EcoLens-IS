"""
EcoLens Step 03b -- PRITHVI LAYER / POOLING ABLATION  (Phase 2; C9)

C9 asks the project to justify its embedding layer. 03_extract_embeddings.py takes
Prithvi's **block 8** output and **mean-pools** the patch tokens:

    latent = model.forward_features(x)[-4]      # "Block 8 for optimal semantic features"
    pooled = latent[:, 1:, :].mean(dim=1)       # drop CLS, mean-pool

That comment asserts "optimal" with no sweep behind it (implementation.md C9). This
script produces the evidence: every candidate layer, both pooling strategies, evaluated
under 07's GROUPED protocol.

WHY THIS IS CHEAP (~6 min, not ~60)
-----------------------------------
The obvious implementation re-runs the encoder once per layer. But
`forward_features()` returns the output of EVERY block from a single forward pass, so
all layers are extracted together and the model runs over each patch exactly once.
Pooling variants are then free -- they are two reductions of the same tensor.

Writes results/layer_ablation.json

Run:
    python 03b_layer_ablation.py
    python 03b_layer_ablation.py --limit 300      # quick check
"""

import argparse
import json
import os
import sys

import numpy as np

from config import METADATA_CATALOG_PATH, RESULTS_DIR

OUT_PATH = f"{RESULTS_DIR}/layer_ablation.json"
K = 5


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


def main():
    ap = argparse.ArgumentParser(description="EcoLens 03b: Prithvi layer/pooling ablation")
    ap.add_argument("--limit", type=int, default=None, help="evaluate on the first N sub-crops")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import torch
    from tqdm import tqdm

    ext = _mod("_ext", "03_extract_embeddings.py")

    with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    entries = [e for e in catalog
               if e.get("processed_path") and os.path.exists(e["processed_path"])]
    if args.limit:
        entries = entries[: args.limit]
    print(f"patches: {len(entries)}")

    bases = np.array([e.get("base_id", e["id"].rsplit("_p", 1)[0]) for e in entries])
    ecos = np.array([e["ecosystem"] for e in entries])

    print("loading Prithvi ...")
    model = ext.load_prithvi_model()

    # probe how many blocks forward_features returns, so LAYERS is not hardcoded wrong
    probe = np.load(entries[0]["processed_path"])
    with torch.no_grad():
        t = torch.from_numpy(probe).float().unsqueeze(0).unsqueeze(2).to(ext.DEVICE)
        outs = model.forward_features(t)
    n_out = len(outs)
    print(f"forward_features returns {n_out} tensors; "
          f"03 uses index -4 (= {n_out - 4})")
    layers = sorted({i for i in (n_out - 12, n_out - 8, n_out - 6, n_out - 4,
                                 n_out - 2, n_out - 1) if 0 <= i < n_out})
    print(f"layers to test: {layers}")

    feats = {(li, pool): [] for li in layers for pool in ("mean", "cls")}

    for s in tqdm(range(0, len(entries), args.batch), desc="extracting"):
        chunk = entries[s: s + args.batch]
        arr = np.stack([np.load(e["processed_path"]) for e in chunk])
        with torch.no_grad():
            x = torch.from_numpy(arr).float().unsqueeze(2).to(ext.DEVICE)
            outs = model.forward_features(x)          # ALL blocks, one pass
            for li in layers:
                lat = outs[li]
                feats[(li, "mean")].append(lat[:, 1:, :].mean(dim=1).cpu().numpy())
                feats[(li, "cls")].append(lat[:, 0, :].cpu().numpy())

    report, rows = {}, []
    for (li, pool), parts in feats.items():
        X = np.concatenate(parts).astype(np.float32)
        m = grouped_map(bases, ecos, X)
        report[f"layer{li}_{pool}"] = {"layer_index": li, "pooling": pool,
                                       "dim": int(X.shape[1]), "map_at_k": m,
                                       "is_current_default": (li == n_out - 4 and pool == "mean")}
        rows.append((m, li, pool, X.shape[1]))

    print(f"\n{'='*66}")
    print(f"PRITHVI LAYER / POOLING ABLATION -- GROUPED mAP@{K}, {len(entries)} patches")
    print(f"{'='*66}")
    print(f"  {'layer':>6}{'pooling':>10}{'dim':>7}{'mAP@5':>10}")
    for m, li, pool, d in sorted(rows, reverse=True):
        mark = "   <-- 03's current default" if (li == n_out - 4 and pool == "mean") else ""
        print(f"  {li:>6}{pool:>10}{d:>7}{m:>10.4f}{mark}")

    best = max(rows)
    cur = [r for r in rows if r[1] == n_out - 4 and r[2] == "mean"]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {OUT_PATH}")
    if cur:
        d = best[0] - cur[0][0]
        print(f"\nbest = layer {best[1]} / {best[2]} at {best[0]:.4f}; "
              f"current default scores {cur[0][0]:.4f} ({d:+.4f})")
        if d > 0.005:
            print("=> The current block-8 mean-pool default is NOT optimal. Report the")
            print("   sweep and either switch or justify keeping it (implementation.md 9).")
        else:
            print("=> The current default is at or near the best option -- C9 can now cite")
            print("   this sweep instead of asserting it.")


if __name__ == "__main__":
    main()
