import json
import os
import numpy as np

from config import RESULTS_DIR, SUPPORTED_MODELS

def evaluate_ecological_similarity():
    print(f"\n{'='*70}")
    print("EcoLens: Ecological Characteristic Similarity Evaluation")
    print(f"{'='*70}")
    
    desc_path = f"{RESULTS_DIR}/ecosystem_descriptors.json"
    if not os.path.exists(desc_path):
        print(f"Error: Descriptors not found at {desc_path}. Run 09_explainability_engine.py first.")
        return
        
    with open(desc_path, encoding="utf-8") as f:
        descriptors = json.load(f)
        
    retrieval_dir = RESULTS_DIR
    retrieval_files = [
        f for f in os.listdir(retrieval_dir)
        if f.startswith("retrieval_results_") and f.endswith(".json")
    ]
    
    if not retrieval_files:
        print(f"No retrieval result files found in {RESULTS_DIR}.")
        return

    print("Analyzing Mean Absolute Error (MAE) for physical characteristics (Top-5 Retreivals)")
    print("This ensures retrieved ecosystems are physically/climatically analogous.\n")
    
    for rfile in sorted(retrieval_files):
        model_key = rfile.replace("retrieval_results_", "").replace(".json", "")
        label = SUPPORTED_MODELS.get(model_key, {}).get("label", model_key)
        
        with open(os.path.join(retrieval_dir, rfile), encoding="utf-8") as f:
            results = json.load(f)
            
        cosine_res = results.get("cosine", {})
        
        temp_diffs = []
        rain_diffs = []
        elev_diffs = []
        forest_diffs = []
        protected_agreements = []
        trajectory_jaccards = []

        for qid, analogs in cosine_res.items():
            q_desc = descriptors.get(qid)
            if not q_desc: continue

            # Analyze Top 5
            for analog in analogs[:5]:
                aid = analog["id"]
                a_desc = descriptors.get(aid)
                if not a_desc: continue

                # Exclude identical patches
                if aid == qid: continue

                if q_desc.get("temp_c") is not None and a_desc.get("temp_c") is not None:
                    temp_diffs.append(abs(q_desc["temp_c"] - a_desc["temp_c"]))
                if q_desc.get("rainfall_mm") is not None and a_desc.get("rainfall_mm") is not None:
                    rain_diffs.append(abs(q_desc["rainfall_mm"] - a_desc["rainfall_mm"]))
                if q_desc.get("elevation_m") is not None and a_desc.get("elevation_m") is not None:
                    elev_diffs.append(abs(q_desc["elevation_m"] - a_desc["elevation_m"]))
                if q_desc.get("forest_cover") is not None and a_desc.get("forest_cover") is not None:
                    forest_diffs.append(abs(q_desc["forest_cover"] - a_desc["forest_cover"]))
                # Faculty comment: "evaluate whether retrieved ecosystems
                # exhibit similar ... conservation status." protected_area
                # is a bool (never None -- see 09_explainability_engine.py,
                # it always falls back to the catalog value), so agreement
                # is always computable, unlike the geo_lookups-dependent
                # fields above.
                if q_desc.get("protected_area") is not None and a_desc.get("protected_area") is not None:
                    protected_agreements.append(1.0 if q_desc["protected_area"] == a_desc["protected_area"] else 0.0)
                # Faculty comment: "disturbance pathways ... comparable
                # ecological trajectories over time" -- Jaccard similarity
                # of Hansen loss-years (forest/mangrove patches only; see
                # 09_explainability_engine.py::get_disturbance_history).
                if q_desc.get("loss_years") is not None and a_desc.get("loss_years") is not None:
                    q_years = set(q_desc["loss_years"].keys())
                    a_years = set(a_desc["loss_years"].keys())
                    union = q_years | a_years
                    trajectory_jaccards.append(len(q_years & a_years) / len(union) if union else 1.0)

        print(f"[{label}]")
        print(f"  Temperature MAE      : {np.mean(temp_diffs):.2f} °C" if temp_diffs else "  Temperature MAE      : N/A")
        print(f"  Rainfall MAE         : {np.mean(rain_diffs):.2f} mm" if rain_diffs else "  Rainfall MAE         : N/A")
        print(f"  Elevation MAE        : {np.mean(elev_diffs):.2f} m" if elev_diffs else "  Elevation MAE        : N/A")
        print(f"  Forest Cover MAE     : {np.mean(forest_diffs):.2f} %" if forest_diffs else "  Forest Cover MAE     : N/A")
        if protected_agreements:
            agree_pct = 100.0 * np.mean(protected_agreements)
            print(f"  Protection-status agreement: {agree_pct:.1f}% of top-5 analogs share the query's protected/unprotected status ({len(protected_agreements)} pairs)")
        else:
            print(f"  Protection-status agreement: N/A")
        if trajectory_jaccards:
            print(f"  Disturbance trajectory (Hansen loss-year Jaccard): {np.mean(trajectory_jaccards):.3f} avg "
                  f"(forest/mangrove pairs only, n={len(trajectory_jaccards)}; 1.0 = identical loss-years, 0.0 = no shared loss-years)")
        else:
            print(f"  Disturbance trajectory (Hansen loss-year Jaccard): N/A")
        print()

if __name__ == "__main__":
    evaluate_ecological_similarity()
