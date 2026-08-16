"""
EcoLens Objective 2 - Step 6: Ecosystem Similarity Retrieval Engine

Implements the core retrieval framework using four similarity measures:
  - Cosine Similarity
  - Euclidean Distance (converted to similarity)
  - k-Nearest Neighbor (kNN) retrieval
  - Approximate Nearest Neighbor (ANN) retrieval via HNSW

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
import resource
import time
import numpy as np
import faiss
from tqdm import tqdm

from config import (
    METADATA_CATALOG_PATH, EMBEDDINGS_DIR,
    RESULTS_DIR, DEFAULT_TOP_K,
    SUPPORTED_MODELS, DEFAULT_MODEL,
)


# ---------------------------------------------------------------
# Retrieval Engine (FAISS-Accelerated)
# ---------------------------------------------------------------

class EcosystemRetrievalEngine:
    """
    Retrieval engine that indexes ecosystem embeddings using FAISS and supports
    similarity search using multiple measures.

    Supports four methods:
      - 'cosine'    : Cosine Similarity (via faiss.IndexFlatIP on L2-normalized embeddings)
      - 'euclidean' : Euclidean Distance (via faiss.IndexFlatL2, converted to
                      similarity via 1 / (1 + dist))
      - 'knn'       : k-Nearest Neighbor retrieval (Euclidean distance search
                      via faiss.IndexFlatL2 -- same underlying index as
                      'euclidean'; kept as a separate named method because
                      that's how the original retrieval design and the
                      downstream evaluation/dashboard scripts refer to it)
      - 'ann'       : Approximate Nearest Neighbor retrieval via
                      faiss.IndexHNSWFlat (Hierarchical Navigable Small
                      World graph search on cosine/inner-product space).
                      Unlike 'euclidean'/'knn' below, this is a genuinely
                      different retrieval STRATEGY, not just a different
                      distance formula on the same exact search -- HNSW
                      approximates the neighbor graph, so its ranking can
                      (and sometimes does) diverge from the exact methods,
                      trading a small recall loss for sub-linear query time
                      at scale. At this project's dataset size (~700
                      patches) the speed difference is not the point; it's
                      here to answer the faculty review comment asking for
                      cosine/Euclidean/ANN to be investigated as genuinely
                      distinct retrieval strategies, not to solve a
                      performance problem that doesn't exist yet at this
                      scale.

    IMPORTANT CAVEAT (applies to 'cosine' / 'euclidean' / 'knn' only):
    embeddings produced by 03_extract_embeddings.py are L2-normalized
    before saving (`emb /= np.linalg.norm(emb) + 1e-8`). For
    unit-normalized vectors, cosine similarity and Euclidean distance are
    related by a fixed monotonic transform:

        ||a - b||^2 = 2 - 2 * cos_sim(a, b)

    which means ranking by Euclidean distance and ranking by cosine
    similarity produce IDENTICAL orderings on these embeddings -- 'cosine',
    'euclidean', and 'knn' will therefore agree on rank order (their
    absolute scores differ, but not which items come first). This isn't a
    bug; it's a real property of comparing normalized vectors, and it's
    worth stating explicitly rather than presenting three methods as if
    they were three independent signals. 'ann' is the one method here that
    is NOT guaranteed to agree with the other three, precisely because it
    searches an approximate graph rather than the exact index.
    """

    SUPPORTED_METHODS = ["cosine", "euclidean", "knn", "ann"]
    HNSW_M = 32               # neighbors per node in the HNSW graph
    HNSW_EF_CONSTRUCTION = 40
    HNSW_EF_SEARCH = 32

    def __init__(self, catalog, embeddings):
        """
        Args:
            catalog: list of catalog entry dicts (from metadata/catalog.json)
            embeddings: dict mapping patch id -> numpy embedding vector
        """
        self.catalog = {entry["id"]: entry for entry in catalog}
        self.embeddings = embeddings
        self.ids = list(embeddings.keys())

        # Build FAISS indices
        # Load and stack all vectors as float32 matrix
        matrix = np.vstack([self.embeddings[pid] for pid in self.ids]).astype(np.float32)
        N, D = matrix.shape

        # Verify L2-normalization for Cosine similarity (IndexFlatIP)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized_matrix = matrix / (norms + 1e-8)

        self.index_cos = faiss.IndexFlatIP(D)
        self.index_cos.add(normalized_matrix)

        # Raw (unnormalized-as-stored) matrix for Euclidean/kNN search.
        # See the class docstring for why this produces the same ranking
        # as cosine on L2-normalized embeddings.
        self.index_l2 = faiss.IndexFlatL2(D)
        self.index_l2.add(matrix)

        # Approximate index: HNSW graph over the SAME normalized vectors
        # cosine search uses (inner product on unit vectors = cosine
        # similarity), so any divergence from 'cosine' below is genuinely
        # from the approximate graph search, not from a different metric.
        self.index_ann = faiss.IndexHNSWFlat(D, self.HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self.index_ann.hnsw.efConstruction = self.HNSW_EF_CONSTRUCTION
        self.index_ann.hnsw.efSearch = self.HNSW_EF_SEARCH
        self.index_ann.add(normalized_matrix)

    def search(self, query_id, method="cosine", top_k=DEFAULT_TOP_K):
        """
        Retrieve top-K most similar ecosystems for a given query patch.
        Uses FAISS for sub-linear search complexity.
        """
        if query_id not in self.embeddings:
            raise ValueError(f"Query ID '{query_id}' not found in embeddings.")
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method '{method}'. Use one of: {self.SUPPORTED_METHODS}")

        query_vec = self.embeddings[query_id].astype(np.float32).reshape(1, -1)
        N = len(self.ids)
        k_to_search = min(top_k + 1, N)

        if method == "cosine":
            q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            raw_scores, indices = self.index_cos.search(q_norm, k_to_search)
            raw_scores = raw_scores[0]
            # Cosine similarity is already in a natural "higher = more similar" scale.
            scores = raw_scores
        elif method == "ann":
            q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            raw_scores, indices = self.index_ann.search(q_norm, k_to_search)
            # HNSW built with METRIC_INNER_PRODUCT on unit vectors -> inner
            # product == cosine similarity, same "higher = more similar" scale
            # as the exact 'cosine' method, but via approximate graph search.
            scores = raw_scores[0]
        else:
            # 'euclidean' and 'knn' both rank by L2 distance via the same index.
            dists, indices = self.index_l2.search(query_vec, k_to_search)
            dists = dists[0]
            # faiss.IndexFlatL2 returns squared L2 distance -- take the
            # square root to get true Euclidean distance before converting.
            dists = np.sqrt(np.maximum(dists, 0.0))
            scores = 1.0 / (1.0 + dists)

        indices = indices[0]

        results = []
        for i in range(len(indices)):
            idx = indices[i]
            if idx == -1:
                continue
            candidate_id = self.ids[idx]
            if candidate_id == query_id:
                continue

            entry = self.catalog[candidate_id]
            res = {
                "id": candidate_id,
                "ecosystem": entry["ecosystem"],
                "name": entry["name"],
                "lon": entry["lon"],
                "lat": entry["lat"],
                "protected_area": entry.get("protected_area", False),
                "climatic_region": entry.get("climatic_region", "Unknown"),
            }
            
            res["score"] = float(scores[i])
            results.append(res)
            
        # Add rank
        for rank, item in enumerate(results, 1):
            item["rank"] = rank
            
        return results[:top_k]

    def search_all(self, method="cosine", top_k=DEFAULT_TOP_K):
        """
        Run retrieval for every patch as a query (leave-one-out) using tqdm progress bar.
        """
        results = {}
        for query_id in tqdm(self.ids, desc=f"FAISS {method.upper()} Search"):
            results[query_id] = self.search(query_id, method=method, top_k=top_k)
        return results

    def build_analog_database(self, method="cosine", top_k=DEFAULT_TOP_K):
        """
        Build the ecosystem analog database using a tqdm progress bar.
        """
        analog_db = []
        for query_id in tqdm(self.ids, desc="Building Analog DB"):
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

    # ---------------------------------------------------------------
    # Operational feasibility (faculty comment: report processing time,
    # memory, scalability). index_build_time_s covers building all four
    # FAISS indices (cosine/L2/HNSW); per-method search_all_time_s covers
    # a full leave-one-out sweep (every patch queried against every
    # other); peak_rss_mb is the whole process's peak resident memory at
    # the point this script finishes, via resource.getrusage -- a coarse
    # but dependency-free proxy for "memory requirements".
    # ---------------------------------------------------------------
    perf = {"model": model_key, "num_patches": len(valid_entries), "embedding_dim": model_cfg["embedding_dim"]}

    t0 = time.perf_counter()
    engine = EcosystemRetrievalEngine(valid_entries, embeddings)
    perf["index_build_time_s"] = round(time.perf_counter() - t0, 4)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Run retrieval for all four methods
    # ---------------------------------------------------------------

    all_results = {}
    perf["search"] = {}

    for method in EcosystemRetrievalEngine.SUPPORTED_METHODS:
        print(f"\nRunning {method.upper()} retrieval for all {len(valid_entries)} patches...")

        # Retrieve all (not just top_k) for evaluation purposes
        t0 = time.perf_counter()
        full_results = engine.search_all(method=method, top_k=len(valid_entries))
        elapsed = time.perf_counter() - t0
        all_results[method] = full_results
        perf["search"][method] = {
            "total_time_s": round(elapsed, 4),
            "avg_query_time_ms": round(1000.0 * elapsed / max(len(valid_entries), 1), 4),
        }

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

    # ---------------------------------------------------------------
    # 4. Save operational-feasibility report (perf: time + memory)
    # ---------------------------------------------------------------
    perf["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)  # KB -> MB on Linux

    perf_path = f"{RESULTS_DIR}/retrieval_perf.json"
    all_perf = {}
    if os.path.exists(perf_path):
        with open(perf_path) as f:
            all_perf = json.load(f)
    all_perf[model_key] = perf
    with open(perf_path, "w") as f:
        json.dump(all_perf, f, indent=2)

    print(f"\n  Operational feasibility ({label}):")
    print(f"    Index build time:  {perf['index_build_time_s']:.3f}s for {len(valid_entries)} patches ({model_cfg['embedding_dim']}D)")
    for method, m in perf["search"].items():
        print(f"    {method:<10} full leave-one-out sweep: {m['total_time_s']:.3f}s "
              f"({m['avg_query_time_ms']:.3f}ms/query avg)")
    print(f"    Peak process memory (RSS): {perf['peak_rss_mb']:.1f} MB")
    print(f"    Saved to: {perf_path}")

    print(f"\nDone. Run 07_evaluate_retrieval.py next to evaluate retrieval performance.")


if __name__ == "__main__":
    main()
