"""
EcoLens Objective 2 - Step 6: Ecosystem Similarity Retrieval Engine

Implements the core retrieval framework using three similarity measures:
  - Cosine Similarity
  - Euclidean Distance (converted to similarity)
  - k-Nearest Neighbor (kNN) retrieval

Supports multiple foundation models via --model flag:
  prithvi, vit, resnet

Run:
    python 06_retrieval_engine.py                # default: prithvi
    python 06_retrieval_engine.py --model vit
    python 06_retrieval_engine.py --model resnet
"""

import argparse
import json
import os
import numpy as np

from config import (
    METADATA_CATALOG_PATH, EMBEDDINGS_DIR,
    RESULTS_DIR, DEFAULT_TOP_K,
    SUPPORTED_MODELS, DEFAULT_MODEL,
)


# ---------------------------------------------------------------
# Similarity Measures
# ---------------------------------------------------------------

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    Returns a value in [-1, 1], where 1 means identical direction.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_similarity(a, b):
    """
    Compute similarity based on Euclidean (L2) distance.
    Converts distance to a similarity score: 1 / (1 + distance).
    Returns a value in (0, 1], where 1 means identical vectors.
    """
    dist = float(np.linalg.norm(a - b))
    return 1.0 / (1.0 + dist)


def euclidean_distance(a, b):
    """Compute raw Euclidean (L2) distance between two vectors."""
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------
# Retrieval Engine
# ---------------------------------------------------------------

class EcosystemRetrievalEngine:
    """
    Retrieval engine that indexes ecosystem embeddings and supports
    similarity search using multiple measures.

    Supports three methods as defined in the project proposal:
      - 'cosine'    : Cosine Similarity
      - 'euclidean' : Euclidean Distance (converted to similarity)
      - 'knn'       : k-Nearest Neighbor retrieval (using Euclidean distance)
    """

    SUPPORTED_METHODS = ["cosine", "euclidean", "knn"]

    def __init__(self, catalog, embeddings):
        """
        Args:
            catalog: list of catalog entry dicts (from metadata/catalog.json)
            embeddings: dict mapping patch id -> numpy embedding vector
        """
        self.catalog = {entry["id"]: entry for entry in catalog}
        self.embeddings = embeddings
        self.ids = list(embeddings.keys())

    def search(self, query_id, method="cosine", top_k=DEFAULT_TOP_K):
        """
        Retrieve top-K most similar ecosystems for a given query patch.

        Args:
            query_id: ID of the query patch (e.g., "forest_001")
            method: similarity method — "cosine", "euclidean", or "knn"
            top_k: number of results to return

        Returns:
            list of dicts, each with:
              - id, score, ecosystem, name, lon, lat,
                protected_area, climatic_region, rank
        """
        if query_id not in self.embeddings:
            raise ValueError(f"Query ID '{query_id}' not found in embeddings.")
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method '{method}'. Use one of: {self.SUPPORTED_METHODS}")

        query_vec = self.embeddings[query_id]
        scores = []

        for candidate_id in self.ids:
            if candidate_id == query_id:
                continue

            candidate_vec = self.embeddings[candidate_id]

            if method == "cosine":
                score = cosine_similarity(query_vec, candidate_vec)
            elif method == "euclidean":
                score = euclidean_similarity(query_vec, candidate_vec)
            elif method == "knn":
                # kNN uses Euclidean distance — lower is better,
                # so we negate for consistent descending sort
                dist = euclidean_distance(query_vec, candidate_vec)
                score = -dist
            else:
                raise ValueError(f"Unknown method: {method}")

            entry = self.catalog[candidate_id]
            scores.append({
                "id": candidate_id,
                "score": score,
                "ecosystem": entry["ecosystem"],
                "name": entry["name"],
                "lon": entry["lon"],
                "lat": entry["lat"],
                "protected_area": entry.get("protected_area", False),
                "climatic_region": entry.get("climatic_region", "Unknown"),
            })

        # Sort by score descending (highest similarity / closest neighbor first)
        scores.sort(key=lambda x: x["score"], reverse=True)

        # For kNN, convert back to positive distance for readability
        if method == "knn":
            for item in scores:
                item["distance"] = -item["score"]
                item["score"] = euclidean_similarity(
                    self.embeddings[query_id],
                    self.embeddings[item["id"]]
                )

        # Add rank
        for rank, item in enumerate(scores, 1):
            item["rank"] = rank

        return scores[:top_k]

    def search_all(self, method="cosine", top_k=DEFAULT_TOP_K):
        """
        Run retrieval for every patch as a query (leave-one-out).

        Returns:
            dict mapping query_id -> list of ranked results
        """
        results = {}
        for query_id in self.ids:
            results[query_id] = self.search(query_id, method=method, top_k=top_k)
        return results

    def build_analog_database(self, method="cosine", top_k=DEFAULT_TOP_K):
        """
        Build the ecosystem analog database — for each patch, store its
        top-K most similar ecological analogs.

        This produces the 'Ecosystem analog database' expected output
        from the project proposal.

        Returns:
            list of dicts, each with query info + its analogs
        """
        analog_db = []

        for query_id in self.ids:
            entry = self.catalog[query_id]
            analogs = self.search(query_id, method=method, top_k=top_k)

            analog_db.append({
                "query_id": query_id,
                "query_ecosystem": entry["ecosystem"],
                "query_name": entry["name"],
                "query_lon": entry["lon"],
                "query_lat": entry["lat"],
                "query_protected_area": entry.get("protected_area", False),
                "query_climatic_region": entry.get("climatic_region", "Unknown"),
                "method": method,
                "analogs": analogs,
            })

        return analog_db


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EcoLens Step 6: Ecosystem Similarity Retrieval Engine"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        choices=list(SUPPORTED_MODELS.keys()),
        help=f"Foundation model embeddings to use (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    model_key = args.model
    model_cfg = SUPPORTED_MODELS[model_key]
    emb_dir = model_cfg["embeddings_dir"]
    label = model_cfg["label"]

    print(f"\n{'='*60}")
    print(f"EcoLens Retrieval Engine - {label}")
    print(f"{model_cfg['description']}")
    print(f"{'='*60}\n")

    # Validate prerequisites
    if not os.path.exists(METADATA_CATALOG_PATH):
        print(f"Error: Catalog not found at {METADATA_CATALOG_PATH}.")
        print("Run scripts 01-04 first to generate embeddings and catalog.")
        return

    with open(METADATA_CATALOG_PATH) as f:
        catalog = json.load(f)

    # Load embeddings from the model-specific directory
    valid_entries = []
    embeddings = {}
    for e in catalog:
        emb_path = f"{emb_dir}/{e['id']}.npy"
        if os.path.exists(emb_path):
            valid_entries.append(e)
            embeddings[e["id"]] = np.load(emb_path)

    if len(valid_entries) < 2:
        print(f"Error: Need at least 2 patches with embeddings in {emb_dir}/. Found {len(valid_entries)}.")
        print(f"Run: python 03_extract_embeddings.py --model {model_key}")
        return

    print(f"Loaded {len(valid_entries)} embeddings from {emb_dir}/")

    # Initialize retrieval engine
    engine = EcosystemRetrievalEngine(valid_entries, embeddings)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Run retrieval for all three methods
    # ---------------------------------------------------------------

    all_results = {}

    for method in EcosystemRetrievalEngine.SUPPORTED_METHODS:
        print(f"\nRunning {method.upper()} retrieval for all {len(valid_entries)} patches...")

        # Retrieve all (not just top_k) for evaluation purposes
        full_results = engine.search_all(method=method, top_k=len(valid_entries))
        all_results[method] = full_results

        # Print example queries
        example_ids = list(embeddings.keys())[:3]
        for qid in example_ids:
            top5 = full_results[qid][:5]
            query_eco = engine.catalog[qid]["ecosystem"]
            print(f"\n  Query: {qid} ({query_eco})")
            print(f"  {'Rank':<5} {'ID':<20} {'Score':<10} {'Ecosystem':<15} {'Name'}")
            print(f"  {'-'*5} {'-'*20} {'-'*10} {'-'*15} {'-'*30}")
            for item in top5:
                match_tag = "[Y]" if item["ecosystem"] == query_eco else "[N]"
                print(f"  {item['rank']:<5} {item['id']:<20} {item['score']:<10.4f} "
                      f"{item['ecosystem']:<15} {item['name'][:30]} {match_tag}")

    # Save model-specific retrieval results
    results_path = f"{RESULTS_DIR}/retrieval_results_{model_key}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRetrieval results saved to: {results_path}")

    # Also save as the default file for backward compatibility
    default_results_path = f"{RESULTS_DIR}/retrieval_results.json"
    with open(default_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ---------------------------------------------------------------
    # 2. Build ecosystem analog database (using cosine similarity)
    # ---------------------------------------------------------------

    print(f"\nBuilding ecosystem analog database (top-{DEFAULT_TOP_K} per patch)...")
    analog_db = engine.build_analog_database(method="cosine", top_k=DEFAULT_TOP_K)

    analog_path = f"{RESULTS_DIR}/analog_database_{model_key}.json"
    with open(analog_path, "w") as f:
        json.dump(analog_db, f, indent=2)
    print(f"Ecosystem analog database saved to: {analog_path}")

    # Also save as the default file for backward compatibility
    default_analog_path = f"{RESULTS_DIR}/analog_database.json"
    with open(default_analog_path, "w") as f:
        json.dump(analog_db, f, indent=2)

    # ---------------------------------------------------------------
    # 3. Summary statistics
    # ---------------------------------------------------------------

    print(f"\n{'='*60}")
    print(f"RETRIEVAL ENGINE SUMMARY ({label})")
    print(f"{'='*60}")
    print(f"  Model: {label}")
    print(f"  Total patches indexed: {len(valid_entries)}")
    print(f"  Ecosystem categories: {sorted(set(e['ecosystem'] for e in valid_entries))}")
    print(f"  Similarity methods: {EcosystemRetrievalEngine.SUPPORTED_METHODS}")
    print(f"  Analog database entries: {len(analog_db)}")
    print(f"\n  Output files:")
    print(f"    {results_path}")
    print(f"    {analog_path}")
    print(f"\nDone. Run 07_evaluate_retrieval.py next to evaluate retrieval performance.")


if __name__ == "__main__":
    main()
