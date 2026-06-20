"""
EcoLens Objective 2 — Step 7: Retrieval Performance Evaluation

Systematically evaluates the retrieval engine from Step 6 using standard
information retrieval metrics. Addresses the project proposal activities:
  - Evaluate retrieval performance across different ecosystem types
  - Analyze retrieval consistency for: Forest, Wetlands, Mangroves
  - Compare retrieval quality across similarity methods

Ground truth definition: a retrieved patch is "relevant" if it belongs
to the same ecosystem category as the query patch.

Prerequisites:
    Run 06_retrieval_engine.py first to generate retrieval results.

Run:
    python 07_evaluate_retrieval.py
"""

import json
import os
import numpy as np

from config import RESULTS_DIR, EVALUATION_K_VALUES


# ---------------------------------------------------------------
# Information Retrieval Metrics
# ---------------------------------------------------------------

def precision_at_k(retrieved, relevant_set, k):
    """
    Precision@K: Of the top-K retrieved items, how many are relevant?

    Example: If top-5 results for forest_001 are
      [forest_003, wetland_001, forest_007, forest_005, mangrove_001]
    and relevant = all forest_* patches, then P@5 = 3/5 = 0.60

    Args:
        retrieved: list of retrieved item dicts (ranked by score)
        relevant_set: set of relevant item IDs
        k: cutoff rank
    Returns:
        float: precision score in [0, 1]
    """
    top_k = retrieved[:k]
    relevant_count = sum(1 for item in top_k if item["id"] in relevant_set)
    return relevant_count / k


def recall_at_k(retrieved, relevant_set, k):
    """
    Recall@K: Of all relevant items, how many appear in the top-K?

    Example: If there are 14 other forest patches and 4 appear in top-5,
    then R@5 = 4/14 = 0.286

    Args:
        retrieved: list of retrieved item dicts (ranked by score)
        relevant_set: set of relevant item IDs
        k: cutoff rank
    Returns:
        float: recall score in [0, 1]
    """
    if len(relevant_set) == 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_count = sum(1 for item in top_k if item["id"] in relevant_set)
    return relevant_count / len(relevant_set)


def average_precision(retrieved, relevant_set):
    """
    Average Precision (AP): Measures the quality of the entire ranking.
    Computes precision at every rank position where a relevant item
    is found, then averages those values.

    Higher AP means relevant items are ranked higher in the list.

    Args:
        retrieved: list of retrieved item dicts (ranked by score)
        relevant_set: set of relevant item IDs
    Returns:
        float: AP score in [0, 1]
    """
    if len(relevant_set) == 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0

    for rank, item in enumerate(retrieved, 1):
        if item["id"] in relevant_set:
            hits += 1
            sum_precisions += hits / rank

    return sum_precisions / len(relevant_set)


def reciprocal_rank(retrieved, relevant_set):
    """
    Reciprocal Rank (RR): 1 / (rank of the first relevant item).

    Example: If the first same-ecosystem result is at rank 2,
    then RR = 1/2 = 0.50

    Args:
        retrieved: list of retrieved item dicts (ranked by score)
        relevant_set: set of relevant item IDs
    Returns:
        float: RR score in (0, 1]
    """
    for rank, item in enumerate(retrieved, 1):
        if item["id"] in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------

def evaluate_method(retrieval_results, catalog_lookup, method_name, k_values):
    """
    Evaluate a single retrieval method across all queries.

    Args:
        retrieval_results: dict mapping query_id -> ranked result list
        catalog_lookup: dict mapping patch_id -> catalog entry
        method_name: name of the method (for display)
        k_values: list of K values for P@K and R@K

    Returns:
        dict with overall and per-category metrics
    """
    # Group patches by ecosystem category
    ecosystem_patches = {}
    for pid, entry in catalog_lookup.items():
        eco = entry["ecosystem"]
        if eco not in ecosystem_patches:
            ecosystem_patches[eco] = set()
        ecosystem_patches[eco].add(pid)

    # Collect per-query metrics
    all_ap = []
    all_rr = []
    per_k_precision = {k: [] for k in k_values}
    per_k_recall = {k: [] for k in k_values}

    # Per-category metrics
    category_metrics = {}
    for eco in ecosystem_patches:
        category_metrics[eco] = {
            "ap": [],
            "rr": [],
            "precision": {k: [] for k in k_values},
            "recall": {k: [] for k in k_values},
        }

    # Confusion matrix: for each query category, count how many results
    # from each category appear in top-K
    max_k = max(k_values)
    confusion = {}
    for eco in ecosystem_patches:
        confusion[eco] = {other: 0 for other in ecosystem_patches}

    for query_id, results in retrieval_results.items():
        query_eco = catalog_lookup[query_id]["ecosystem"]

        # Relevant set: all patches of the same ecosystem (excluding query itself)
        relevant_set = ecosystem_patches[query_eco] - {query_id}

        # Compute metrics
        ap = average_precision(results, relevant_set)
        rr = reciprocal_rank(results, relevant_set)
        all_ap.append(ap)
        all_rr.append(rr)
        category_metrics[query_eco]["ap"].append(ap)
        category_metrics[query_eco]["rr"].append(rr)

        for k in k_values:
            p_at_k = precision_at_k(results, relevant_set, k)
            r_at_k = recall_at_k(results, relevant_set, k)
            per_k_precision[k].append(p_at_k)
            per_k_recall[k].append(r_at_k)
            category_metrics[query_eco]["precision"][k].append(p_at_k)
            category_metrics[query_eco]["recall"][k].append(r_at_k)

        # Confusion matrix (top max_k results)
        for item in results[:max_k]:
            confusion[query_eco][item["ecosystem"]] += 1

    # Aggregate metrics
    overall = {
        "method": method_name,
        "num_queries": len(retrieval_results),
        "mAP": float(np.mean(all_ap)),
        "MRR": float(np.mean(all_rr)),
    }

    for k in k_values:
        overall[f"P@{k}"] = float(np.mean(per_k_precision[k]))
        overall[f"R@{k}"] = float(np.mean(per_k_recall[k]))

    # Per-category aggregation
    per_category = {}
    for eco in sorted(ecosystem_patches.keys()):
        cat = category_metrics[eco]
        num_queries = len(cat["ap"])
        cat_result = {
            "ecosystem": eco,
            "num_queries": num_queries,
            "num_relevant_per_query": len(ecosystem_patches[eco]) - 1,
            "mAP": float(np.mean(cat["ap"])) if cat["ap"] else 0.0,
            "MRR": float(np.mean(cat["rr"])) if cat["rr"] else 0.0,
        }
        for k in k_values:
            cat_result[f"P@{k}"] = float(np.mean(cat["precision"][k])) if cat["precision"][k] else 0.0
            cat_result[f"R@{k}"] = float(np.mean(cat["recall"][k])) if cat["recall"][k] else 0.0
        per_category[eco] = cat_result

    # Normalize confusion matrix (per-query-category percentages)
    confusion_normalized = {}
    for eco in confusion:
        total = sum(confusion[eco].values())
        if total > 0:
            confusion_normalized[eco] = {
                other: round(count / total, 4)
                for other, count in confusion[eco].items()
            }
        else:
            confusion_normalized[eco] = confusion[eco]

    return {
        "overall": overall,
        "per_category": per_category,
        "confusion_matrix": confusion_normalized,
    }


# ---------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------

def print_overall_table(evaluations):
    """Print a comparison table of all methods."""
    methods = list(evaluations.keys())
    k_values = EVALUATION_K_VALUES

    print("\n" + "=" * 80)
    print("OVERALL RETRIEVAL PERFORMANCE — Cross-Method Comparison")
    print("=" * 80)

    # Header
    header = f"{'Method':<12} {'mAP':<8} {'MRR':<8}"
    for k in k_values:
        header += f" {'P@'+str(k):<8}"
    for k in k_values:
        header += f" {'R@'+str(k):<8}"
    print(header)
    print("─" * len(header))

    for method in methods:
        overall = evaluations[method]["overall"]
        row = f"{method:<12} {overall['mAP']:<8.4f} {overall['MRR']:<8.4f}"
        for k in k_values:
            row += f" {overall[f'P@{k}']:<8.4f}"
        for k in k_values:
            row += f" {overall[f'R@{k}']:<8.4f}"
        print(row)


def print_category_table(evaluations, method):
    """Print per-category metrics for a given method."""
    k_values = EVALUATION_K_VALUES
    per_category = evaluations[method]["per_category"]

    print(f"\n{'='*80}")
    print(f"PER-CATEGORY RETRIEVAL PERFORMANCE — {method.upper()}")
    print(f"{'='*80}")

    header = f"{'Category':<15} {'#Q':<4} {'#Rel':<5} {'mAP':<8} {'MRR':<8}"
    for k in k_values:
        header += f" {'P@'+str(k):<8}"
    print(header)
    print("─" * len(header))

    for eco in sorted(per_category.keys()):
        cat = per_category[eco]
        row = (f"{eco:<15} {cat['num_queries']:<4} "
               f"{cat['num_relevant_per_query']:<5} "
               f"{cat['mAP']:<8.4f} {cat['MRR']:<8.4f}")
        for k in k_values:
            row += f" {cat[f'P@{k}']:<8.4f}"
        print(row)


def print_confusion_matrix(evaluations, method):
    """Print the confusion matrix for a given method."""
    confusion = evaluations[method]["confusion_matrix"]
    categories = sorted(confusion.keys())

    print(f"\n{'='*80}")
    print(f"RETRIEVAL CONFUSION MATRIX — {method.upper()}")
    print(f"(Row = query category, Column = retrieved category, values = proportion)")
    print(f"{'='*80}")

    # Header
    header = f"{'Query \\ Retr':<15}"
    for cat in categories:
        header += f" {cat[:10]:<12}"
    print(header)
    print("─" * len(header))

    for query_cat in categories:
        row = f"{query_cat:<15}"
        for retr_cat in categories:
            val = confusion[query_cat].get(retr_cat, 0.0)
            if query_cat == retr_cat:
                row += f" {val:<12.4f}"  # Same-category hits
            else:
                row += f" {val:<12.4f}"
        print(row)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    results_path = f"{RESULTS_DIR}/retrieval_results.json"

    if not os.path.exists(results_path):
        print(f"Error: Retrieval results not found at {results_path}.")
        print("Run 06_retrieval_engine.py first.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    # Build catalog lookup from the retrieval results themselves
    # Each result entry contains ecosystem, name, lon, lat metadata
    catalog_lookup = {}
    for method_name, method_results in all_results.items():
        for query_id, results in method_results.items():
            # Add query itself to catalog lookup
            if query_id not in catalog_lookup:
                # Infer ecosystem from the query_id prefix
                ecosystem = query_id.rsplit("_", 1)[0]
                catalog_lookup[query_id] = {
                    "id": query_id,
                    "ecosystem": ecosystem,
                }
            # Add all result entries to catalog lookup
            for item in results:
                if item["id"] not in catalog_lookup:
                    catalog_lookup[item["id"]] = {
                        "id": item["id"],
                        "ecosystem": item["ecosystem"],
                        "name": item.get("name", ""),
                    }
        break  # Only need one method to build the lookup

    # Also load catalog.json for richer metadata if available
    from config import METADATA_CATALOG_PATH
    if os.path.exists(METADATA_CATALOG_PATH):
        with open(METADATA_CATALOG_PATH) as f:
            catalog = json.load(f)
        for entry in catalog:
            if entry["id"] not in catalog_lookup:
                catalog_lookup[entry["id"]] = entry
            else:
                catalog_lookup[entry["id"]].update(entry)

    # ---------------------------------------------------------------
    # Evaluate each method
    # ---------------------------------------------------------------

    evaluations = {}

    for method_name, method_results in all_results.items():
        print(f"\nEvaluating {method_name.upper()} retrieval...")
        evaluation = evaluate_method(
            method_results, catalog_lookup, method_name, EVALUATION_K_VALUES
        )
        evaluations[method_name] = evaluation

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------

    # 1. Cross-method comparison table
    print_overall_table(evaluations)

    # 2. Per-category analysis for each method
    for method_name in all_results.keys():
        print_category_table(evaluations, method_name)

    # 3. Confusion matrix for the best method (cosine by default)
    print_confusion_matrix(evaluations, "cosine")

    # ---------------------------------------------------------------
    # Determine best method
    # ---------------------------------------------------------------

    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")

    # Find best method by mAP
    best_method = max(evaluations.keys(), key=lambda m: evaluations[m]["overall"]["mAP"])
    best_map = evaluations[best_method]["overall"]["mAP"]
    print(f"\n  Best overall method by mAP: {best_method.upper()} (mAP = {best_map:.4f})")

    # Per-category insights
    print(f"\n  Per-category retrieval quality ({best_method.upper()}):")
    per_cat = evaluations[best_method]["per_category"]
    sorted_cats = sorted(per_cat.keys(), key=lambda c: per_cat[c]["mAP"], reverse=True)
    for eco in sorted_cats:
        map_val = per_cat[eco]["mAP"]
        quality = "excellent" if map_val > 0.8 else "good" if map_val > 0.5 else "moderate" if map_val > 0.3 else "poor"
        print(f"    {eco:<15} mAP={map_val:.4f} ({quality})")

    # Random baseline context
    categories = list(per_cat.keys())
    avg_relevant = np.mean([per_cat[c]["num_relevant_per_query"] for c in categories])
    total_patches = sum(per_cat[c]["num_queries"] for c in categories)
    random_p5 = avg_relevant / (total_patches - 1)
    print(f"\n  Random baseline P@5 ≈ {random_p5:.4f}")
    print(f"  Achieved P@5 = {evaluations[best_method]['overall']['P@5']:.4f} "
          f"({evaluations[best_method]['overall']['P@5']/random_p5:.1f}x above random)")

    # ---------------------------------------------------------------
    # Save evaluation report
    # ---------------------------------------------------------------

    report_path = f"{RESULTS_DIR}/evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(evaluations, f, indent=2)
    print(f"\n  Evaluation report saved to: {report_path}")
    print(f"\nDone. Run 08_retrieval_dashboard.py next to visualize these results.")


if __name__ == "__main__":
    main()
