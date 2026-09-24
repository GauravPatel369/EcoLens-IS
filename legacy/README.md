# Superseded runners — kept for provenance, do not use

Both predate `run_phase.py` and are retained only because `docs/implementation.md`
references them. Neither is part of the supported pipeline.

| File | Why it is here and not at the root |
|---|---|
| `run_pipeline.py` | **Destructive.** Deletes `data/patches_processed/` and the embedding directories on start. It now refuses to run against a populated catalog without `--force`, but it should not be reached for at all. Its list of directories also predates Clay and Satlas, so a forced run leaves a *silently inconsistent* catalog rather than a clean one. |
| `run_all.py` | Written for the 3-model era (steps 01–11). Not resumable, and unaware of every step added since. |

**Use `run_phase.py` instead** — 31 steps, resumable, state in `outputs/logs/pipeline_state.json`.
