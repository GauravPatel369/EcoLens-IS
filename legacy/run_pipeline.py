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

    # 0. Clean old preprocessed patches and embeddings to force complete regeneration.
    #
    # GUARD ADDED 4 Sep (C25 audit). This block used to delete unconditionally, and
    # setup.py advertised this function as the package's console script -- so the one
    # command the packaging offered would wipe a large part of the catalog. Worse, the
    # folder list predates Clay and Satlas, so embeddings_clay/ and embeddings_satlas/
    # survive: the result is not a clean slate but a SILENTLY INCONSISTENT one, where
    # some models' vectors match the current patches and others do not.
    #
    # Prefer run_phase.py, which is resumable and destroys nothing. If you really do want
    # a from-scratch rebuild, pass --force and it behaves as before.
    force = "--force" in sys.argv
    doomed = ["patches_processed", "embeddings", "embeddings_vit", "embeddings_resnet",
              "embeddings_clay", "embeddings_satlas"]
    populated = [f for f in doomed
                 if os.path.isdir(f) and any(os.scandir(f))]
    if populated and not force:
        print("\n  REFUSING TO DELETE a populated catalog.")
        for f in populated:
            print(f"    {f}/  ({sum(1 for _ in os.scandir(f))} entries)")
        print("  Re-running from scratch would discard hours of embedding work.")
        print("  Use run_phase.py to resume, or pass --force to delete anyway.\n")
        sys.exit(1)

    print("Cleaning up old preprocessed patches and embeddings directories to force regeneration...")
    for folder in doomed:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"  Removed folder: {folder}")
            except Exception as e:
                print(f"  Warning: could not remove folder {folder}: {e}")

    # 0. Quality control gate. Runs before anything expensive so empty,
    #    offshore or stale patches are caught before they are embedded
    #    into results that look perfectly normal. Non-blocking by design:
    #    it reports and continues, so a known-imperfect catalog can still
    #    be run deliberately.
    print()
    print('>>> Quality control (00_validate_locations.py)')
    subprocess.run([venv_python, '00_validate_locations.py'], capture_output=False)

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
