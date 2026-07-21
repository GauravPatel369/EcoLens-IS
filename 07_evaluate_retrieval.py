"""
EcoLens Objective 2 - Step 7: Retrieval Performance Evaluation (Multi-Model)

Evaluates retrieval quality for every foundation model that has been run
through 06_retrieval_engine.py. Automatically discovers all
retrieval_results_<model>.json files in the results/ directory.

---------------------------------------------------------------------
EVALUATION METHODOLOGY -- read this before trusting a single number
---------------------------------------------------------------------
02_preprocess_patches.py expands each base location into ~10 sub-crops
via small pixel offsets, and those sub-crops overlap by roughly 60-80%
of their pixels (see that script's docstring). If retrieval relevance
is defined purely as "same ecosystem category", a query's own
sub-crops are counted as valid hits -- and since they're near-
duplicates of the query, the model finds them almost perfectly. That
inflates precision/mAP with a signal that has nothing to do with
recognizing a *different* location as ecologically similar, which is
the actual capability this project is trying to measure.

This script computes TWO evaluations for every method:

  GROUPED (the default / top-level result in evaluation_report.json)
    -- a leave-one-LOCATION-out evaluation. All sub-crops sharing a
    base_id with the query are removed from BOTH the candidate pool
    and the relevant set before any metric is computed -- they are
    not scored as hits or as misses, they simply aren't there. This
    is the honest number: it measures whether the model can find a
    DIFFERENT physical location of the same ecosystem type.

  LEAKED (nested under evaluation_report.json[model][method]["leaked"])
    -- the original patch-level evaluation, where any patch of the
    same ecosystem (including other sub-crops of the SAME base
    location as the query) counts as relevant. Kept as a diagnostic
    so the size of the leakage effect is visible, not because it's a
    number worth reporting on its own.

GROUPED is placed at the top level (not LEAKED) specifically so that
08_retrieval_dashboard.py's existing charts -- which read
ev.<method>.overall / .per_category / .confusion_matrix directly --
automatically display the honest numbers with no dashboard changes
required. Expect GROUPED mAP/P@1 to be noticeably lower than LEAKED;
that drop is the leakage being removed, not a regression.

Run:
    python 07_evaluate_retrieval.py
"""

import glob
import json
import os
import numpy as np

from config import RESULTS_DIR, EVALUATION_K_VALUES, SUPPORTED_MODELS, METADATA_CATALOG_PATH


# ---------------------------------------------------------------
# Information Retrieval Metrics (unchanged -- these are correct;
# only what's fed into them, via group_aware filtering, changes)
# ---------------------------------------------------------------

def precision_at_k(retrieved, relevant_set, k):
    """Precision@K: of the top-K retrieved items, how many are relevant?"""
    top_k = retrieved[:k]
    relevant_count = sum(1 for item in top_k if item["id"] in relevant_set)
    return relevant_count / k


def recall_at_k(retrieved, relevant_set, k):
    """Recall@K: of all relevant items, how many appear in the top-K?"""
    if len(relevant_set) == 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_count = sum(1 for item in top_k if item["id"] in relevant_set)
    return relevant_count / len(relevant_set)


def average_precision(retrieved, relevant_set):
    """Average Precision (AP): quality of the entire ranking."""
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
    """Reciprocal Rank (RR): 1 / (rank of the first relevant item)."""
    for rank, item in enumerate(retrieved, 1):
        if item["id"] in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------

def get_base_id(pid, catalog_lookup):
    """
    Resolve the base location id for a patch id.

    Prefers the explicit "base_id" field written by
    02_preprocess_patches.py. Falls back to the old fragile
    string-split heuristic only for catalogs generated before that
    field existed (audit report Bug #4's concern) -- a safety net,
    not the primary mechanism anymore.
    """
    entry = catalog_lookup.get(pid)
    if entry and "base_id" in entry:
        return entry["base_id"]
    return pid.split("_p")[0]


# ---------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------

def evaluate_method(retrieval_results, catalog_lookup, method_name, k_values, group_aware):
    """
    Evaluate a single retrieval method across all queries.

    group_aware=True:  leave-one-location-out (remove same-base_id
        candidates from both the ranked list and the relevant set).
    group_aware=False: original patch-level evaluation.
    """
    ecosystem_patches = {}
    for pid, entry in catalog_lookup.items():
        eco = entry["ecosystem"]
        ecosystem_patches.setdefault(eco, set()).add(pid)

    all_ap, all_rr = [], []
    per_k_precision = {k: [] for k in k_values}
    per_k_recall = {k: [] for k in k_values}

    category_metrics = {
        eco: {"ap": [], "rr": [], "precision": {k: [] for k in k_values}, "recall": {k: [] for k in k_values}}
        for eco in ecosystem_patches
    }

    max_k = max(k_values)
    confusion = {eco: {other: 0 for other in ecosystem_patches} for eco in ecosystem_patches}

    num_queries_evaluated = 0
    num_queries_skipped = 0

    for query_id, results in retrieval_results.items():
        query_eco = catalog_lookup[query_id]["ecosystem"]
        query_base = get_base_id(query_id, catalog_lookup)

        if group_aware:
            filtered_results = [
                item for item in results
                if get_base_id(item["id"], catalog_lookup) != query_base
            ]
            relevant_set = {
                pid for pid in ecosystem_patches[query_eco]
                if get_base_id(pid, catalog_lookup) != query_base
            }
            if len(relevant_set) == 0:
                num_queries_skipped += 1
                continue
        else:
            filtered_results = results
            relevant_set = ecosystem_patches[query_eco] - {query_id}

        num_queries_evaluated += 1

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

        for item in filtered_results[:max_k]:
            confusion[query_eco][item["ecosystem"]] += 1

    overall = {
        "method": method_name,
        "group_aware": group_aware,
        "num_queries": num_queries_evaluated,
        "num_queries_skipped": num_queries_skipped,
        "mAP": float(np.mean(all_ap)) if all_ap else 0.0,
        "MRR": float(np.mean(all_rr)) if all_rr else 0.0,
    }
    for k in k_values:
        overall[f"P@{k}"] = float(np.mean(per_k_precision[k])) if per_k_precision[k] else 0.0
        overall[f"R@{k}"] = float(np.mean(per_k_recall[k])) if per_k_recall[k] else 0.0

    per_category = {}
    for eco in sorted(ecosystem_patches.keys()):
        cat = category_metrics[eco]
        has_data = len(cat["ap"]) > 0
        cat_result = {
            "ecosystem": eco,
            "num_queries": len(cat["ap"]),
            "num_relevant_per_query": len(ecosystem_patches[eco]) - 1,
            # None (not 0.0) when no query in this category could be
            # fairly evaluated -- e.g. group_aware=True and every patch
            # in the category shares the same base_id, so there is no
            # "other location" to compare against. 0.0 would look like
            # a real (poor) score; None is explicit about "no data".
            "mAP": float(np.mean(cat["ap"])) if has_data else None,
            "MRR": float(np.mean(cat["rr"])) if has_data else None,
        }
        for k in k_values:
            cat_result[f"P@{k}"] = float(np.mean(cat["precision"][k])) if has_data else None
            cat_result[f"R@{k}"] = float(np.mean(cat["recall"][k])) if has_data else None
        per_category[eco] = cat_result

    confusion_normalized = {}
    for eco in confusion:
        total = sum(confusion[eco].values())
        if total > 0:
            confusion_normalized[eco] = {other: round(count / total, 4) for other, count in confusion[eco].items()}
        else:
            confusion_normalized[eco] = confusion[eco]

    return {"overall": overall, "per_category": per_category, "confusion_matrix": confusion_normalized}


# ---------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------

def _eval_block(evaluations, method, key="grouped"):
    """
    key='grouped' -> the top-level (honest, de-leaked) result.
    key='leaked'  -> the nested diagnostic result.
    """
    if key == "grouped":
        return evaluations[method]
    return evaluations[method]["leaked"]


def print_overall_table(evaluations, key="grouped", title="OVERALL RETRIEVAL PERFORMANCE"):
    methods = list(evaluations.keys())
    k_values = EVALUATION_K_VALUES

    print("\n" + "=" * 80)
    print(f"{title} ({key.upper()})")
    print("=" * 80)

    header = f"{'Method':<12} {'mAP':<8} {'MRR':<8}"
    for k in k_values:
        header += f" {'P@'+str(k):<8}"
    for k in k_values:
        header += f" {'R@'+str(k):<8}"
    print(header)
    print("-" * len(header))

    for method in methods:
        overall = _eval_block(evaluations, method, key)["overall"]
        row = f"{method:<12} {overall['mAP']:<8.4f} {overall['MRR']:<8.4f}"
        for k in k_values:
            row += f" {overall[f'P@{k}']:<8.4f}"
        for k in k_values:
            row += f" {overall[f'R@{k}']:<8.4f}"
        print(row)


def print_leaked_vs_grouped_gap(evaluations, method="cosine"):
    """The single most important diagnostic in this script: how much of
    the headline mAP was coming from same-location sub-crop overlap."""
    if method not in evaluations:
        return
    grouped = _eval_block(evaluations, method, "grouped")["overall"]
    leaked = _eval_block(evaluations, method, "leaked")["overall"]

    print(f"\n{'='*80}")
    print(f"LEAKAGE CHECK -- {method.upper()} (this is the number that matters)")
    print(f"{'='*80}")
    print(f"  Leaked  mAP (same-location sub-crops allowed as hits): {leaked['mAP']:.4f}")
    print(f"  Grouped mAP (leave-one-location-out, honest):          {grouped['mAP']:.4f}")
    gap = leaked["mAP"] - grouped["mAP"]
    pct = (gap / leaked["mAP"] * 100.0) if leaked["mAP"] > 0 else 0.0
    print(f"  Gap: {gap:.4f}  ({pct:.1f}% of the leaked mAP came from same-location overlap)")
    print(f"  Leaked  P@1: {leaked['P@1']:.4f}   Grouped P@1: {grouped['P@1']:.4f}")


def print_category_table(evaluations, method, key="grouped"):
    k_values = EVALUATION_K_VALUES
    per_category = _eval_block(evaluations, method, key)["per_category"]

    print(f"\n{'='*80}")
    print(f"PER-CATEGORY RETRIEVAL PERFORMANCE - {method.upper()} ({key.upper()})")
    print(f"{'='*80}")

    header = f"{'Category':<15} {'#Q':<4} {'#Rel':<5} {'mAP':<8} {'MRR':<8}"
    for k in k_values:
        header += f" {'P@'+str(k):<8}"
    print(header)
    print("-" * len(header))

    for eco in sorted(per_category.keys()):
        cat = per_category[eco]
        if cat["mAP"] is None:
            print(f"{eco:<15} {cat['num_queries']:<4} {cat['num_relevant_per_query']:<5} "
                  f"{'N/A':<8} {'N/A':<8}  (no other location in this category to compare against)")
            continue
        row = (f"{eco:<15} {cat['num_queries']:<4} "
               f"{cat['num_relevant_per_query']:<5} "
               f"{cat['mAP']:<8.4f} {cat['MRR']:<8.4f}")
        for k in k_values:
            row += f" {cat[f'P@{k}']:<8.4f}"
        print(row)


def print_confusion_matrix(evaluations, method, key="grouped"):
    confusion = _eval_block(evaluations, method, key)["confusion_matrix"]
    categories = sorted(confusion.keys())

    print(f"\n{'='*80}")
    print(f"RETRIEVAL CONFUSION MATRIX - {method.upper()} ({key.upper()})")
    print(f"(Row = query category, Column = retrieved category, values = proportion)")
    print(f"{'='*80}")

    header = f"{'Query/Retr':<15}"
    for cat in categories:
        header += f" {cat[:10]:<12}"
    print(header)
    print("-" * len(header))

    for query_cat in categories:
        row = f"{query_cat:<15}"
        for retr_cat in categories:
            val = confusion[query_cat].get(retr_cat, 0.0)
            row += f" {val:<12.4f}"
        print(row)


def build_catalog_lookup(all_results):
    """Build a catalog lookup dict from the retrieval results, enriched
    with catalog.json (the authoritative source for the real base_id
    field written by 02_preprocess_patches.py)."""
    catalog_lookup = {}
    for method_name, method_results in all_results.items():
        for query_id, results in method_results.items():
            if query_id not in catalog_lookup:
                base_id = query_id.split("_p")[0]
                ecosystem = base_id.rsplit("_", 1)[0] if "_" in base_id else base_id
                catalog_lookup[query_id] = {"id": query_id, "ecosystem": ecosystem, "base_id": base_id}
            for item in results:
                if item["id"] not in catalog_lookup:
                    catalog_lookup[item["id"]] = {
                        "id": item["id"],
                        "ecosystem": item["ecosystem"],
                        "name": item.get("name", ""),
                        "base_id": item["id"].split("_p")[0],
                    }
        break  # Only need one method to build the lookup

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
    """
    Evaluate a single model's retrieval results.

    Returns, per method:
        {
          "overall": <GROUPED overall metrics>,
          "per_category": <GROUPED per-category metrics>,
          "confusion_matrix": <GROUPED confusion matrix>,
          "leaked": {"overall": ..., "per_category": ..., "confusion_matrix": ...},
        }

    Top level = grouped (honest) results, so 08_retrieval_dashboard.py's
    existing JS (ev.<method>.overall etc.) keeps working unmodified and
    now shows the de-leaked numbers automatically.
    """
    with open(results_path) as f:
        all_results = json.load(f)

    catalog_lookup = build_catalog_lookup(all_results)

    evaluations = {}
    for method_name, method_results in all_results.items():
        grouped = evaluate_method(method_results, catalog_lookup, method_name, EVALUATION_K_VALUES, group_aware=True)
        leaked = evaluate_method(method_results, catalog_lookup, method_name, EVALUATION_K_VALUES, group_aware=False)
        evaluations[method_name] = {
            "overall": grouped["overall"],
            "per_category": grouped["per_category"],
            "confusion_matrix": grouped["confusion_matrix"],
            "leaked": leaked,
        }

    return evaluations


def print_cross_model_table(all_model_evals, key="grouped"):
    k_values = EVALUATION_K_VALUES

    print(f"\n{'='*90}")
    print(f"CROSS-MODEL RETRIEVAL PERFORMANCE COMPARISON -- Cosine Similarity ({key.upper()})")
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
        if "cosine" not in evals:
            continue
        cosine_overall = _eval_block(evals, "cosine", key)["overall"]
        row = f"{label:<20} {cosine_overall.get('mAP', 0):<8.4f} {cosine_overall.get('MRR', 0):<8.4f}"
        for k in k_values:
            row += f" {cosine_overall.get(f'P@{k}', 0):<8.4f}"
        for k in k_values:
            row += f" {cosine_overall.get(f'R@{k}', 0):<8.4f}"
        print(row)


def print_cross_model_category_table(all_model_evals, key="grouped"):
    print(f"\n{'='*90}")
    print(f"PER-CATEGORY mAP COMPARISON ACROSS MODELS -- Cosine Similarity ({key.upper()})")
    print(f"{'='*90}")

    all_cats = set()
    for evals in all_model_evals.values():
        if "cosine" not in evals:
            continue
        per_cat = _eval_block(evals, "cosine", key)["per_category"]
        all_cats.update(per_cat.keys())
    all_cats = sorted(all_cats)

    header = f"{'Category':<15}"
    for model_key in all_model_evals:
        label = SUPPORTED_MODELS.get(model_key, {}).get("label", model_key)
        header += f" {label:<15}"
    print(header)
    print("-" * len(header))

    for cat in all_cats:
        row = f"{cat:<15}"
        for model_key, evals in all_model_evals.items():
            if "cosine" not in evals:
                row += f" {'--':<15}"
                continue
            cat_data = _eval_block(evals, "cosine", key)["per_category"].get(cat, {})
            map_val = cat_data.get("mAP")
            row += f" {'N/A':<15}" if map_val is None else f" {map_val:<15.4f}"
        print(row)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    result_files = glob.glob(f"{RESULTS_DIR}/retrieval_results_*.json")

    if not result_files:
        legacy_path = f"{RESULTS_DIR}/retrieval_results.json"
        if os.path.exists(legacy_path):
            result_files = [legacy_path]
        else:
            print(f"Error: No retrieval result files found in {RESULTS_DIR}/.")
            print("Run 06_retrieval_engine.py --model <model> first.")
            return

    model_files = {}
    for fpath in sorted(result_files):
        fname = os.path.basename(fpath)
        if fname.startswith("retrieval_results_") and fname.endswith(".json"):
            model_key = fname.replace("retrieval_results_", "").replace(".json", "")
            model_files[model_key] = fpath
        elif fname == "retrieval_results.json":
            model_files["prithvi"] = fpath

    print(f"Found retrieval results for {len(model_files)} model(s): {list(model_files.keys())}")

    print(f"\n{'#'*80}")
    print("# A NOTE ON THE NUMBERS BELOW")
    print("# Every method is evaluated two ways: GROUPED (leave-one-location-out,")
    print("# honest -- this is also what's saved at the top level of")
    print("# evaluation_report.json) and LEAKED (original, allows a query's own")
    print("# near-duplicate sub-crops to count as correct matches -- saved under")
    print("# ['leaked'] as a diagnostic). Read this file's module docstring for why.")
    print(f"{'#'*80}")

    all_model_evals = {}

    for model_key, fpath in model_files.items():
        label = SUPPORTED_MODELS.get(model_key, {}).get("label", model_key)
        print(f"\n{'='*80}")
        print(f"Evaluating {label} ({model_key})")
        print(f"{'='*80}")

        evaluations = evaluate_single_model(fpath, model_key)
        all_model_evals[model_key] = evaluations

        print_overall_table(evaluations, key="grouped", title="OVERALL RETRIEVAL PERFORMANCE")
        print_overall_table(evaluations, key="leaked", title="OVERALL RETRIEVAL PERFORMANCE")
        print_leaked_vs_grouped_gap(evaluations, method="cosine")

        if "cosine" in evaluations:
            print_category_table(evaluations, "cosine", key="grouped")
            print_confusion_matrix(evaluations, "cosine", key="grouped")

    if len(all_model_evals) > 1:
        print_cross_model_table(all_model_evals, key="grouped")
        print_cross_model_category_table(all_model_evals, key="grouped")
        print_cross_model_table(all_model_evals, key="leaked")

    # ---------------------------------------------------------------
    # Summary (GROUPED metrics -- the honest ones)
    # ---------------------------------------------------------------

    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY (grouped / leave-one-location-out metrics)")
    print(f"{'='*80}")

    best_model = max(
        all_model_evals.keys(),
        key=lambda m: _eval_block(all_model_evals[m], "cosine", "grouped")["overall"]["mAP"] if "cosine" in all_model_evals[m] else 0
    )
    best_label = SUPPORTED_MODELS.get(best_model, {}).get("label", best_model)
    best_map = _eval_block(all_model_evals[best_model], "cosine", "grouped")["overall"]["mAP"]
    print(f"\n  Best model by grouped cosine mAP: {best_label} (mAP = {best_map:.4f})")

    per_cat = _eval_block(all_model_evals[best_model], "cosine", "grouped")["per_category"]
    evaluable_cats = {c: v for c, v in per_cat.items() if v["mAP"] is not None}
    skipped_cats = [c for c, v in per_cat.items() if v["mAP"] is None]
    sorted_cats = sorted(evaluable_cats.keys(), key=lambda c: evaluable_cats[c]["mAP"], reverse=True)
    print(f"\n  Per-category retrieval quality ({best_label}, Cosine, grouped):")
    for eco in sorted_cats:
        map_val = evaluable_cats[eco]["mAP"]
        quality = "excellent" if map_val > 0.8 else "good" if map_val > 0.5 else "moderate" if map_val > 0.3 else "poor"
        print(f"    {eco:<15} mAP={map_val:.4f} ({quality})")
    if skipped_cats:
        print(f"    (skipped -- no other location to compare against: {', '.join(skipped_cats)})")

    if not evaluable_cats:
        print(f"\n  No category had more than one distinct base location -- grouped")
        print(f"  evaluation isn't meaningful on this dataset. Check your catalog.")
        report_path = f"{RESULTS_DIR}/evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(all_model_evals, f, indent=2)
        print(f"\n  Evaluation report saved to: {report_path}")
        return

    categories = list(evaluable_cats.keys())
    avg_relevant = np.mean([evaluable_cats[c]["num_relevant_per_query"] for c in categories])
    total_patches = sum(evaluable_cats[c]["num_queries"] for c in categories)
    random_p5 = avg_relevant / max(total_patches - 1, 1)
    print(f"\n  Random baseline P@5 ~ {random_p5:.4f}")
    best_p5 = _eval_block(all_model_evals[best_model], "cosine", "grouped")["overall"]["P@5"]
    ratio = best_p5 / random_p5 if random_p5 > 0 else float("nan")
    print(f"  Best model (grouped) P@5 = {best_p5:.4f} ({ratio:.1f}x above random)")
    print(f"\n  NOTE: this random baseline is coarse -- it doesn't account for")
    print(f"  category-size imbalance or missing patches (see README). Treat it")
    print(f"  as an order-of-magnitude sanity check, not a precise figure.")

    # ---------------------------------------------------------------
    # Save unified evaluation report
    # ---------------------------------------------------------------

    report_path = f"{RESULTS_DIR}/evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(all_model_evals, f, indent=2)
    print(f"\n  Evaluation report saved to: {report_path}")
    print(f"  Top-level overall/per_category/confusion_matrix = GROUPED (honest).")
    print(f"  ['leaked'] sub-key = original patch-level diagnostic.")
    print(f"\nDone. Run 08_retrieval_dashboard.py next to visualize these results.")


if __name__ == "__main__":
    main()
