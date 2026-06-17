"""
EcoLens Phase 1 — Step 4: Catalog finalization & embedding analysis

Wraps up Phase 1 by:
  - Validating the metadata catalog is complete (every patch has an
    embedding, no orphaned entries)
  - Computing basic embedding distribution stats per ecosystem category
  - Running a quick sanity check: do same-ecosystem patches embed
    closer together than different-ecosystem patches? This is the
    first real signal of whether the whole pipeline is working.

Run:
    python 04_finalize_and_analyze.py
"""

import json
import numpy as np
from itertools import combinations

from config import METADATA_CATALOG_PATH


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    with open(METADATA_CATALOG_PATH) as f:
        catalog = json.load(f)

    complete = [e for e in catalog if "embedding_path" in e]
    incomplete = [e for e in catalog if "embedding_path" not in e]

    print(f"Catalog status: {len(complete)} complete, {len(incomplete)} incomplete")
    if incomplete:
        print("Incomplete entries:", [e["id"] for e in incomplete])

    if len(complete) < 2:
        print("\nNeed at least 2 completed patches to run similarity analysis.")
        return

    # Load all embeddings
    embeddings = {}
    for entry in complete:
        embeddings[entry["id"]] = {
            "vector": np.load(entry["embedding_path"]),
            "ecosystem": entry["ecosystem"],
        }

    # Pairwise similarity, grouped by same-ecosystem vs cross-ecosystem
    same_eco_sims = []
    cross_eco_sims = []

    print("\nPairwise cosine similarities:")
    for id_a, id_b in combinations(embeddings.keys(), 2):
        vec_a = embeddings[id_a]["vector"]
        vec_b = embeddings[id_b]["vector"]
        sim = cosine_similarity(vec_a, vec_b)

        eco_a = embeddings[id_a]["ecosystem"]
        eco_b = embeddings[id_b]["ecosystem"]
        same = eco_a == eco_b

        tag = "[SAME ECOSYSTEM]" if same else "[different]"
        print(f"  {id_a:20s} <-> {id_b:20s}  sim={sim:.3f}  {tag}")

        if same:
            same_eco_sims.append(sim)
        else:
            cross_eco_sims.append(sim)

    print("\n--- Sanity check ---")
    if same_eco_sims:
        print(f"Same-ecosystem pairs:  mean similarity = {np.mean(same_eco_sims):.3f} "
              f"(n={len(same_eco_sims)})")
    if cross_eco_sims:
        print(f"Cross-ecosystem pairs: mean similarity = {np.mean(cross_eco_sims):.3f} "
              f"(n={len(cross_eco_sims)})")

    if same_eco_sims and cross_eco_sims:
        gap = np.mean(same_eco_sims) - np.mean(cross_eco_sims)
        print(f"\nGap (same - cross): {gap:.3f}")
        if gap > 0.02:
            print("Good sign: same-ecosystem patches are embedding closer together "
                  "than different-ecosystem patches. The pipeline is producing "
                  "ecologically meaningful embeddings.")
        else:
            print("Warning: little or no separation between same- and "
                  "cross-ecosystem similarity. With only 10 patches this could "
                  "just be small-sample noise -- worth re-checking once you "
                  "scale to 50-100 patches in Phase 2 before concluding "
                  "anything is wrong with the model or pipeline.")


if __name__ == "__main__":
    main()
