import subprocess
import sys
import os
import shutil
import time

def run_cmd(args):
    print(f"\n>>> Running: {' '.join(args)}...")
    start_t = time.time()
    res = subprocess.run(args, capture_output=False)
    elapsed = time.time() - start_t
    if res.returncode != 0:
        print(f"Error: Command {' '.join(args)} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"    [Command finished in {elapsed:.2f} seconds]")
    return elapsed

def main():
    overall_start = time.time()
    venv_python = sys.executable  # Use current Python environment
    print("============================================================")
    print("EcoLens Automation Pipeline — Steps 02 to 09 (with Profiling)")
    print("============================================================")

    # 0. Clean old preprocessed patches and embeddings to force complete regeneration
    print("Cleaning up old preprocessed patches and embeddings directories to force regeneration...")
    for folder in ["patches_processed", "embeddings", "embeddings_vit", "embeddings_resnet"]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"  Removed folder: {folder}")
            except Exception as e:
                print(f"  Warning: could not remove folder {folder}: {e}")

    # 1. Preprocess patches (will compute custom Sentinel-2 stats now!)
    run_cmd([venv_python, "02_preprocess_patches.py"])

    from config import SUPPORTED_MODELS
    # 2. Extract embeddings for all models
    for model in SUPPORTED_MODELS.keys():
        run_cmd([venv_python, "03_extract_embeddings.py", "--model", model])

    # 3. Finalize catalog and run basic sanity check
    run_cmd([venv_python, "04_finalize_and_analyze.py"])

    # 4. Create database and dashboard explorer
    run_cmd([venv_python, "05_create_database_and_dashboard.py"])

    # 5. Run similarity retrieval engine for all models
    for model in SUPPORTED_MODELS.keys():
        run_cmd([venv_python, "06_retrieval_engine.py", "--model", model])

    # 6. Evaluate retrieval engines
    run_cmd([venv_python, "07_evaluate_retrieval.py"])

    # 7. Generate explainability descriptors & explanations
    run_cmd([venv_python, "09_explainability_engine.py"])

    # 8. Build the main visualization dashboard (integrates stats, matrix, and explanations)
    run_cmd([venv_python, "08_retrieval_dashboard.py"])

    total_elapsed = time.time() - overall_start
    print("\n============================================================")
    print(f"SUCCESS: EcoLens pipeline run complete in {total_elapsed:.2f} seconds!")
    print("Open 'retrieval_dashboard.html' in your browser to view results.")
    print("============================================================")

if __name__ == "__main__":
    main()
