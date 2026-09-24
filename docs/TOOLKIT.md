# EcoLens Toolkit — installation, workflow and API

Packaging and user documentation for review comment **C25** ("package as a modular desktop
application or Python toolkit, with user documentation, workflow diagrams, installation
instructions and example datasets").

---

## 1. Installation

```bash
git clone <repo> && cd EcoLens-IS
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux
pip install -e .
```

`-e` (editable) is recommended: the pipeline reads and writes data directories relative to
the repository root, so keeping the code in place avoids duplicating multi-GB artifacts.

Verify:

```python
import ecolens
print(ecolens.__version__, len(ecolens.stages()), "stages")
```

### Environment notes that will otherwise cost you an afternoon

- **Clay and Satlas need extra setup.** Clay is not on PyPI; it is installed from a source
  checkout registered via a `.pth` file, and its package directory was renamed `src/` →
  `claymodel/` upstream, so both spellings must be tried. Satlas needs
  `pip install satlaspretrain_models`. See `implementation.md` §3.5.
- **Clay's checkpoint pulls a DINOv2 teacher** that is unnecessary for inference and large
  enough to trigger `OSError 1455: paging file too small` on a 16 GB machine. `03` forces
  `pretrained=False` and deletes the teacher after load.
- **Reference datasets are separate downloads** (§3 below). The pipeline runs without them
  but reports the affected fields as *unknown* rather than substituting placeholder values.

---

## 2. Workflow

```
                    ┌──────────────────────────────────────────┐
                    │  PILLAR A — ecosystem analog retrieval   │
                    └──────────────────────────────────────────┘
  00 validate ──► 01 acquire ──► 02 preprocess ──► 03 extract embeddings
  locations       Sentinel-2      BOA offset,       Prithvi │ Clay │ Satlas
  vs RESOLVE      L2A patches     nodata, crop      ViT     │ ResNet
                                        │
                                        ▼
                            04 finalize ──► 05 dashboard
                                        │
                                        ▼
                              06 retrieval engine (FAISS)
                                        │
        ┌───────────────┬───────────────┼───────────────┬──────────────┐
        ▼               ▼               ▼               ▼              ▼
   07 evaluate    07b ecological   07c spectral    07d case      08 dashboard
   (grouped mAP)  agreement        baseline        studies
        │
        ├─ 03b layer ablation   19 dimension sweep   22 patch-size sweep
        ├─ 17 cross-region (leave-one-realm-out)     20 threshold + clusters
        └─ 12 stability (seasonal │ atmospheric │ interannual) ─► 21 normalised

                    ┌──────────────────────────────────────────┐
                    │  PILLAR B — forest-loss risk forecasting │
                    └──────────────────────────────────────────┘
  10 grid tiling + Hansen labels ──► 11 risk model (GB │ RF │ XGB │ LGBM)
        │  (+ optional embedding drift)          │
        │                                        ├─ 18 feature-group ablation
        │                                        ├─ 19 SHAP
        │                                        └─ 23 drift ablation (matched)
        ▼
  24 successional stages     25 horizon sweep

                    ┌──────────────────────────────────────────┐
                    │  THE BRIDGE — what connects the pillars  │
                    └──────────────────────────────────────────┘
  13 analog risk features ──► 14 analog ablation ──► 15 trajectory figures
     retrieve analogs from OTHER landscapes (≥250 km, different region),
     turn their Hansen disturbance history into predictors, and test whether
     they improve forecasting under leave-one-region-out.
```

**The bridge is the project's actual claim.** Stages 13–15 are what make this more than two
unrelated halves; §4 below is the shortest path to reproducing that result.

---

## 3. Reference datasets

All are static, one-time downloads. `download_reference_data.py` fetches the first three.

| Dataset | Purpose | Size | Notes |
|---|---|---|---|
| RESOLVE Ecoregions 2017 | realm / biome / ecoregion labels | ~250 MB | validates every location's declared ecosystem |
| WorldClim v2 (BIO1, BIO12) | temperature, rainfall | ~75 MB | BIO1 is °C×10 in some releases; auto-detected |
| Copernicus / SRTM DEM tiles | elevation, ruggedness | ~1.1 GB | keep **unmosaicked**, one tile per degree |
| Hansen Global Forest Change | loss year, tree cover 2000 | ~27 GB | downloaded on demand, cached |
| WDPA protected areas | `protected_area` feature | 4.2 GB → 4.9 GB layer | `python 26_build_wdpa_layer.py` |

**Design principle, and it matters:** when a dataset is absent, lookups return `None` and
the field is reported as *unknown* — never `False`, never an interpolated placeholder. A
missing file is a loud gap; a fabricated number is a silent, expensive one.

---

## 4. Quick start

### Run the whole pipeline, resumably

```bash
python run_phase.py --list          # 31 steps and their state
python run_phase.py                 # run everything not yet done
python run_phase.py --from 06_retrieval_prithvi
python run_phase.py --redo 07b
```

State lives in `logs/pipeline_state.json`; every step logs to `logs/<step>.log`. An
interrupted run resumes rather than restarting.

Installed as a console script:

```bash
ecolens-pipeline --list
```

> **Do not use `run_pipeline.py`.** It is the older non-resumable runner and it deletes
> `patches_processed/` and the embedding directories on start. It now refuses to run against
> a populated catalog without `--force`, but `run_phase.py` is the supported entry point.

### Reproduce the headline result

```bash
python 13_analog_risk_features.py --model prithvi --cells-per-region 100
python 14_analog_ablation.py --model prithvi --spatial-holdout
```

Expected: **+0.0392 PR-AUC (+11.0 %), 14/16 regions improved, paired Wilcoxon p = 0.0010.**
Swap `--model clay` to reproduce the independent replication (+9.9 %, p = 0.0214).

---

## 5. Python API

```python
import ecolens

# Configuration
ecolens.config.PATCH_SIZE_M          # 2240
ecolens.config.SUPPORTED_MODELS      # the five backbones

# Reference lookups
ecolens.geo_lookups.is_protected(23.8611, 52.7439)          # True  (Białowieża)
ecolens.geo_lookups.get_physical_descriptors(-60.02, -3.10) # climate, elevation, ecoregion

# Retrieval
eng = ecolens.retrieval_engine
diag = ecolens.retrieval_diagnostics
diag.THRESHOLDS                                              # the C22 sweep grid

# The bridge
a13 = ecolens.analog_risk_features
a13.CROP_SIZE, a13.RESIZE_TO                                 # catalog crop geometry
means, stds, meta = a13.load_norm_stats()

ecolens.stages()      # {clean_name: numbered_filename}, in run order
```

### Why the modules are named twice

Pipeline files start with a digit (`13_analog_risk_features.py`) because the numbering
documents execution order and is referenced throughout `implementation.md`, the logs and the
results filenames. Python cannot import those names — an identifier may not begin with a
digit — which is why `pip install .` previously shipped metadata and no code.

`ecolens/` maps clean names onto those files at import time. Both spellings work, the CLI is
unchanged, and 15 existing dynamic-load call sites keep working. Modules load **lazily**:
several stages import `torch`, `geopandas` or `rasterio` at module scope, so `import
ecolens` deliberately loads none of them until you name one.

*A full rename remains the cleaner long-term answer and is recorded as the outstanding half
of C25.*

---

## 6. Example dataset

The repository ships a runnable subset so the pipeline can be exercised without the full
multi-GB acquisition:

| Artifact | What it is |
|---|---|
| `metadata/catalog.json` | 1,260 sub-crops across 126 locations, 10 ecosystems, 6 realms |
| `metadata/norm_stats.json` | per-band means/stds — required to embed anything the way the catalog was embedded |
| `results/*.json` | every evaluation output quoted in `implementation.md` |
| `risk_model/cell_year_features.csv` | 143,000 labelled cell-years across 17 forest regions |
| `risk_model/drift_cache_prithvi.json` | 510 precomputed cell embeddings-drift values |
| `results/analog_cell_features*.csv` | analog features for the Phase 4 result, per model |

To exercise retrieval end to end on the shipped catalog without re-acquiring imagery:

```bash
python 06_retrieval_engine.py --model clay
python 07_evaluate_retrieval.py
python 20_retrieval_diagnostics.py
```

---

## 7. Known limitations

Stated here rather than discovered later. Full detail in `implementation.md` §C24 and
`REVIEW_RESPONSE.md`.

- **Ecosystems do not form separated clusters.** Three of five models score a *negative*
  silhouette. Retrieval works as a **ranking**, not a partition — treat output as ranked
  candidates for expert review, never as an ecological equivalence claim.
- **Prithvi's similarity scale is compressed.** Cosine between *unrelated* patches averages
  0.9231, so its raw scores must never be quoted without a scale, and it needs a 0.96
  threshold where Clay operates at 0.70.
- **Precision at the best operating point is 0.2309** (2.35× chance, Clay). This is a
  shortlisting aid, not an automated decision system.
- **Ground footprint is load-bearing**; embedding dimension is not. Quartering the footprint
  costs ~46 % of mAP, while 64 dimensions match native performance.
- **Narrow-fringe coastal ecosystems cannot be sampled at 2,240 m.** Four locations were
  lost to this, three of them mangroves; the surviving mangrove set is biased toward large,
  landward-fringed systems.
- **Recovery is not measurable from Hansen**, which records loss only. Successional *stage*
  is derivable and is; biomass recovery is not.
- **Nothing here is validated against field data.**
