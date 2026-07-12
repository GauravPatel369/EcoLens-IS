"""
EcoLens Objective 2 - Step 7: Retrieval Performance Evaluation (Multi-Model)

Evaluates retrieval quality for every foundation model that has been run
through 06_retrieval_engine.py. Automatically discovers all
retrieval_results_<model>.json files in the results/ directory.

Outputs a unified cross-model comparison and saves per-model evaluations
into a single evaluation_report.json.

Run:
    python 07_evaluate_retrieval.py
"""

import glob
import json
import os
import numpy as np

from config import RESULTS_DIR, EVALUATION_K_VALUES, SUPPORTED_MODELS


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

        # Extract base ID (e.g. 'forest_001' from 'forest_001_p0')
        query_base_id = query_id.split("_p")[0]
        
        # Relevant set: all patches of the same ecosystem, excluding any patches from the SAME base location
        relevant_set = {pid for pid in ecosystem_patches[query_eco] if not pid.startswith(query_base_id)}
        
        # Filter retrieved results to exclude patches from the same base location
        filtered_results = [item for item in results if not item["id"].startswith(query_base_id)]

        # Compute metrics
        ap = average_precision(filtered_results, relevant_set)
        rr = reciprocal_rank(filtered_results, relevant_set)
        all_ap.append(ap)
        all_rr.append(rr)
        category_metrics[query_eco]["ap"].append(ap)
        category_metrics[query_eco]["rr"].append(rr)

        for k in k_values:
            p_at_k = precision_at_k(filtered_results, relevant_set, k)
            r_at_k = recall_at_k(filtered_results, relevant_set, k)
            per_k_precision[k].append(p_at_k)
            per_k_recall[k].append(r_at_k)
            category_metrics[query_eco]["precision"][k].append(p_at_k)
            category_metrics[query_eco]["recall"][k].append(r_at_k)

        # Confusion matrix (top max_k results)
        for item in filtered_results[:max_k]:
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
            "num_relevant_per_query": len(ecosystem_patches[eco]) - 10,
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
    print("OVERALL RETRIEVAL PERFORMANCE - Cross-Method Comparison")
    print("=" * 80)

    # Header
    header = f"{'Method':<12} {'mAP':<8} {'MRR':<8}"
    for k in k_values:
        header += f" {'P@'+str(k):<8}"
    for k in k_values:
        header += f" {'R@'+str(k):<8}"
    print(header)
    print("-" * len(header))

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
    print(f"PER-CATEGORY RETRIEVAL PERFORMANCE - {method.upper()}")
    print(f"{'='*80}")

    header = f"{'Category':<15} {'#Q':<4} {'#Rel':<5} {'mAP':<8} {'MRR':<8}"
    for k in k_values:
        header += f" {'P@'+str(k):<8}"
    print(header)
    print("-" * len(header))

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
    print(f"RETRIEVAL CONFUSION MATRIX - {method.upper()}")
    print(f"(Row = query category, Column = retrieved category, values = proportion)")
    print(f"{'='*80}")

    # Header
    header = f"{'Query/Retr':<15}"
    for cat in categories:
        header += f" {cat[:10]:<12}"
    print(header)
    print("-" * len(header))

    for query_cat in categories:
        row = f"{query_cat:<15}"
        for retr_cat in categories:
            val = confusion[query_cat].get(retr_cat, 0.0)
            if query_cat == retr_cat:
                row += f" {val:<12.4f}"  # Same-category hits
            else:
                row += f" {val:<12.4f}"
        print(row)


def build_catalog_lookup(all_results):
    """Build a catalog lookup dict from the retrieval results."""
    catalog_lookup = {}
    for method_name, method_results in all_results.items():
        for query_id, results in method_results.items():
            if query_id not in catalog_lookup:
                # Extract base ID (e.g. 'forest_001' from 'forest_001_p0')
                base_id = query_id.split("_p")[0]
                # Extract ecosystem from base ID (everything before the last '_NNN')
                ecosystem = base_id.rsplit("_", 1)[0]
                catalog_lookup[query_id] = {
                    "id": query_id,
                    "ecosystem": ecosystem,
                }
            for item in results:
                if item["id"] not in catalog_lookup:
                    catalog_lookup[item["id"]] = {
                        "id": item["id"],
                        "ecosystem": item["ecosystem"],
                        "name": item.get("name", ""),
                    }
        break  # Only need one method to build the lookup

    # Enrich with catalog.json metadata if available
    from config import METADATA_CATALOG_PATH
    if os.path.exists(METADATA_CATALOG_PATH):
        with open(METADATA_CATALOG_PATH) as f:
            catalog = json.load(f)
        for entry in catalog:
            if entry["id"] not in catalog_lookup:
                catalog_lookup[entry["id"]] = entry
            else:
                catalog_lookup[entry["id"]].update(entry)

    return catalog_lookup


def evaluate_single_model(results_path, model_key):
    """Evaluate a single model's retrieval results and return the evaluation dict."""
    label = SUPPORTED_MODELS.get(model_key, {}).get("label", model_key)

    with open(results_path) as f:
        all_results = json.load(f)

    catalog_lookup = build_catalog_lookup(all_results)

    evaluations = {}
    for method_name, method_results in all_results.items():
        evaluation = evaluate_method(
            method_results, catalog_lookup, method_name, EVALUATION_K_VALUES
        )
        evaluations[method_name] = evaluation

    return evaluations


def print_cross_model_table(all_model_evals):
    """Print a comparison table across all evaluated models (using cosine mAP)."""
    k_values = EVALUATION_K_VALUES

    print(f"\n{'='*90}")
    print("CROSS-MODEL RETRIEVAL PERFORMANCE COMPARISON (Cosine Similarity)")
    print(f"{'='*90}")

    header = f"{'Model':<20} {'mAP':<8} {'MRR':<8}"
    for k in k_values:
        header += f" {'P@'+str(k):<8}"
    for k in k_values:
        header += f" {'R@'+str(k):<8}"
    print(header)
    print("-" * len(header))

    for model_key, evals in all_model_evals.items():
        label = SUPPORTED_MODELS.get(model_key, {}).get("label", model_key)
        cosine_overall = evals.get("cosine", {}).get("overall", {})
        if not cosine_overall:
            continue
        row = f"{label:<20} {cosine_overall.get('mAP', 0):<8.4f} {cosine_overall.get('MRR', 0):<8.4f}"
        for k in k_values:
            row += f" {cosine_overall.get(f'P@{k}', 0):<8.4f}"
        for k in k_values:
            row += f" {cosine_overall.get(f'R@{k}', 0):<8.4f}"
        print(row)


def print_cross_model_category_table(all_model_evals):
    """Print per-category mAP comparison across all models."""
    print(f"\n{'='*90}")
    print("PER-CATEGORY mAP COMPARISON ACROSS MODELS (Cosine Similarity)")
    print(f"{'='*90}")

    # Collect all categories
    all_cats = set()
    for evals in all_model_evals.values():
        cosine_eval = evals.get("cosine", {})
        per_cat = cosine_eval.get("per_category", {})
        all_cats.update(per_cat.keys())
    all_cats = sorted(all_cats)

    # Header
    header = f"{'Category':<15}"
    for model_key in all_model_evals:
        label = SUPPORTED_MODELS.get(model_key, {}).get("label", model_key)
        header += f" {label:<15}"
    print(header)
    print("-" * len(header))

    for cat in all_cats:
        row = f"{cat:<15}"
        for model_key, evals in all_model_evals.items():
            cat_data = evals.get("cosine", {}).get("per_category", {}).get(cat, {})
            map_val = cat_data.get("mAP", 0.0)
            row += f" {map_val:<15.4f}"
        print(row)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    # Discover all model-specific retrieval result files
    result_files = glob.glob(f"{RESULTS_DIR}/retrieval_results_*.json")

    if not result_files:
        # Fallback: try the legacy non-model-specific file
        legacy_path = f"{RESULTS_DIR}/retrieval_results.json"
        if os.path.exists(legacy_path):
            result_files = [legacy_path]
        else:
            print(f"Error: No retrieval result files found in {RESULTS_DIR}/.")
            print("Run 06_retrieval_engine.py --model <model> first.")
            return

    # Parse model keys from filenames
    model_files = {}
    for fpath in sorted(result_files):
        fname = os.path.basename(fpath)
        # Extract model key from 'retrieval_results_<model>.json'
        if fname.startswith("retrieval_results_") and fname.endswith(".json"):
            model_key = fname.replace("retrieval_results_", "").replace(".json", "")
            model_files[model_key] = fpath
        elif fname == "retrieval_results.json":
            model_files["prithvi"] = fpath

    print(f"Found retrieval results for {len(model_files)} model(s): {list(model_files.keys())}")

    # Evaluate each model
    all_model_evals = {}

    for model_key, fpath in model_files.items():
        label = SUPPORTED_MODELS.get(model_key, {}).get("label", model_key)
        print(f"\n{'='*80}")
        print(f"Evaluating {label} ({model_key})")
        print(f"{'='*80}")

        evaluations = evaluate_single_model(fpath, model_key)
        all_model_evals[model_key] = evaluations

        # Print per-model method comparison table
        print_overall_table(evaluations)

        # Print per-category for cosine
        if "cosine" in evaluations:
            print_category_table(evaluations, "cosine")
            print_confusion_matrix(evaluations, "cosine")

    # ---------------------------------------------------------------
    # Cross-model comparison
    # ---------------------------------------------------------------

    if len(all_model_evals) > 1:
        print_cross_model_table(all_model_evals)
        print_cross_model_category_table(all_model_evals)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")

    # Find best model by cosine mAP
    best_model = max(
        all_model_evals.keys(),
        key=lambda m: all_model_evals[m].get("cosine", {}).get("overall", {}).get("mAP", 0)
    )
    best_label = SUPPORTED_MODELS.get(best_model, {}).get("label", best_model)
    best_map = all_model_evals[best_model]["cosine"]["overall"]["mAP"]
    print(f"\n  Best model by cosine mAP: {best_label} (mAP = {best_map:.4f})")

    # Per-category insights for the best model
    per_cat = all_model_evals[best_model]["cosine"]["per_category"]
    sorted_cats = sorted(per_cat.keys(), key=lambda c: per_cat[c]["mAP"], reverse=True)
    print(f"\n  Per-category retrieval quality ({best_label}, Cosine):")
    for eco in sorted_cats:
        map_val = per_cat[eco]["mAP"]
        quality = "excellent" if map_val > 0.8 else "good" if map_val > 0.5 else "moderate" if map_val > 0.3 else "poor"
        print(f"    {eco:<15} mAP={map_val:.4f} ({quality})")

    # Random baseline context
    categories = list(per_cat.keys())
    avg_relevant = np.mean([per_cat[c]["num_relevant_per_query"] for c in categories])
    total_patches = sum(per_cat[c]["num_queries"] for c in categories)
    random_p5 = avg_relevant / (total_patches - 1)
    print(f"\n  Random baseline P@5 ~ {random_p5:.4f}")
    best_p5 = all_model_evals[best_model]["cosine"]["overall"]["P@5"]
    print(f"  Best model P@5 = {best_p5:.4f} "
          f"({best_p5/random_p5:.1f}x above random)")

    # ---------------------------------------------------------------
    # Save unified evaluation report
    # ---------------------------------------------------------------

    report_path = f"{RESULTS_DIR}/evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(all_model_evals, f, indent=2)
    print(f"\n  Evaluation report saved to: {report_path}")
    print(f"\nDone. Run 08_retrieval_dashboard.py next to visualize these results.")


if __name__ == "__main__":
    main()
