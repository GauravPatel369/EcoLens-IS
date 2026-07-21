# EcoLens

Satellite-imagery ecosystem similarity retrieval, explainability, and
forest-loss risk forecasting, built on Sentinel-2 imagery and three
foundation models (Prithvi-100M, ViT-Base, ResNet-50).

This is a revised version of an earlier pipeline. This README covers
what changed, why, and what you need to download before running it.
If you're picking this up cold, read **"Known limitations"** before
trusting any number this pipeline produces.

---

## Pipeline overview

```
01_acquire_patches.py         Sentinel-2 patch acquisition (Microsoft Planetary Computer STAC)
02_preprocess_patches.py      Normalize + expand into sub-crops
03_extract_embeddings.py      Prithvi-100M / ViT-Base / ResNet-50 embeddings
04_finalize_and_analyze.py    Catalog validation + same/cross-ecosystem sanity check
05_create_database_and_dashboard.py   PCA projection + standalone HTML dashboard
06_retrieval_engine.py        FAISS retrieval (cosine / euclidean / knn)
07_evaluate_retrieval.py      P@K/R@K/mAP/MRR evaluation -- GROUPED + LEAKED
08_retrieval_dashboard.py     Interactive HTML dashboard (PCA/t-SNE, retrieval, confusion matrix)
09_explainability_engine.py   Real spectral + geospatial descriptors, NL explanations
10_grid_tiling_labels.py      [NEW] Grid-tile forest regions, label with Hansen loss data
11_forest_risk_forecast.py    [NEW] Train + query a forest-loss risk model

geo_lookups.py                [NEW] Real protected-area / climate / elevation / ecoregion lookups
config.py                     Locations, paths, and all tunable constants
config.yaml                   Prithvi-100M's own released model/training config
```

Run in numeric order. Each script reads/writes `metadata/catalog.json`
and files under `results/`, `embeddings*/`, `patches*/`.

---

## Setup

### 1. Python dependencies

```
pip install -r requirements.txt
```

### 2. Reference data (for real, non-fabricated descriptors and risk labels)

The original explainability engine generated elevation/temperature/
rainfall/soil/ecoregion from a hash of the patch ID -- numbers that
looked plausible but were fiction. `geo_lookups.py` replaces that with
real lookups against static datasets you download once:

| Dataset | Used for | Download | Where it goes |
|---|---|---|---|
| WDPA (World Database on Protected Areas) | real protected-area status | [protectedplanet.net](https://www.protectedplanet.net/) -> Shapefile export, merge the 0/1/2 polygon parts | `config.WDPA_POLYGONS_PATH` |
| WorldClim v2 (BIO1, BIO12) | temperature, rainfall | [worldclim.org/data/worldclim21.html](https://worldclim.org/data/worldclim21.html), 30 arc-sec | `config.WORLDCLIM_TEMP_PATH` / `WORLDCLIM_PRECIP_PATH` |
| SRTM 30m or Copernicus GLO-30 | elevation | [earthexplorer.usgs.gov](https://earthexplorer.usgs.gov/) or [registry.opendata.aws/copernicus-dem](https://registry.opendata.aws/copernicus-dem/) | `config.DEM_PATH` (mosaic to one VRT/GeoTIFF) |
| RESOLVE Ecoregions 2017 | ecoregion / biome | [resolve.org/ecoregions](https://ecoregions.appspot.com/) | `config.ECOREGIONS_PATH` |
| Hansen Global Forest Change | forest-loss labels | public GCS bucket, no auth -- see `10_grid_tiling_labels.py`, downloads automatically per-tile | `config.HANSEN_DATA_DIR` |

**None of this is required to run steps 1-8.** `geo_lookups.py`
returns `None` for any field whose dataset isn't downloaded yet, and
every caller is written to treat `None` as "unavailable" rather than
substitute a guess. Step 9's explanations and step 10/11's risk model
need at least WDPA + WorldClim + a DEM to be useful; RESOLVE
Ecoregions is used only for the "shared ecoregion" explanation factor.

**Before trusting any of this at scale**, run:
```
python geo_lookups.py
```
This spot-checks a handful of well-known locations (Sundarbans,
Siberian taiga, Amazon, Punjab farmland) and prints their looked-up
values so you can eyeball them against what you already know. A CRS
mismatch or a marine/terrestrial filtering bug produces
plausible-looking *wrong* numbers, not a crash -- this is the only
real defense against that.

---

## Evaluation methodology -- read this before quoting a number

`02_preprocess_patches.py` expands each of the ~75 base locations into
~10 sub-crops via small pixel offsets. Those sub-crops overlap by
roughly 60-80% of their pixels (the offsets are bounded by patch/crop
geometry -- see that script's docstring for the exact numbers). If
retrieval relevance is "same ecosystem category", a query's own
sub-crops count as valid hits, and since they're near-duplicates the
model finds them almost perfectly. That inflates precision/mAP with a
signal that has nothing to do with recognizing a genuinely different
location as ecologically similar -- which is what this project is
actually trying to measure.

`07_evaluate_retrieval.py` now reports two numbers for everything:

- **GROUPED** (top-level `overall`/`per_category`/`confusion_matrix`
  in `evaluation_report.json`) -- leave-one-location-out. Every
  sub-crop sharing a base location with the query is removed from
  both the candidate pool and the relevant set before any metric is
  computed. This is the number to report.
- **LEAKED** (nested under `["leaked"]`) -- the original patch-level
  evaluation. Kept as a diagnostic to show how large the leakage
  effect was, not because it's a number worth citing on its own.

Expect GROUPED mAP/P@1 to be meaningfully lower than LEAKED. That drop
isn't a regression -- it's the inflation being removed.
`print_leaked_vs_grouped_gap()` prints exactly how much of the old
headline mAP came from this effect.

`08_retrieval_dashboard.py` was deliberately left reading the same
JSON keys as before (`ev.<method>.overall`, `.per_category`,
`.confusion_matrix`) -- since those now hold the GROUPED numbers at
the top level, the dashboard's existing charts show the honest metrics
automatically, no dashboard code changes required. The `["leaked"]`
diagnostic isn't wired into the dashboard UI; if you want it there,
it's sitting in the JSON ready to be charted.

---

## What changed, and why

### Confirmed real, fixed here

- **Prithvi's `num_frames` mismatch (03_extract_embeddings.py).** The
  model was built from `config.yaml`'s `model_args` as-is
  (`num_frames: 3`, since Prithvi-100M was pretrained on 3-timestep
  HLS sequences), then fed single-timestep (T=1) input at inference.
  That's a real shape/semantics mismatch, not cosmetic. The fix --
  confirmed against IBM/NASA's own official Prithvi-100M usage
  example, which explicitly overrides `model_args["num_frames"] = 1`
  before instantiating the model for single-frame use -- is applied
  here. This is a more direct explanation for Prithvi underperforming
  the generic ImageNet models in the original results than the
  HLS-vs-Sentinel-2 calibration mismatch alone, though that mismatch
  is real too (see below) and both apply.
- **`06_retrieval_engine.py` claimed three similarity methods
  (cosine/euclidean/knn) in its docstring but only implemented
  cosine** (`SUPPORTED_METHODS = ["cosine"]`). This wasn't just an
  incomplete docstring -- `evaluation_report.json` and
  `08_retrieval_dashboard.py`'s UI both already expected all three
  (the dashboard has method-selector tabs and chart series for all
  three). Euclidean and kNN are now implemented for real via
  `faiss.IndexFlatL2`. Verified with synthetic embeddings that all
  three methods retrieve correctly.
  - **Worth knowing:** embeddings are L2-normalized before saving
    (`03_extract_embeddings.py`). For unit-normalized vectors, cosine
    similarity and Euclidean distance are related by a fixed monotonic
    transform (`||a-b||^2 = 2 - 2*cos_sim(a,b)`), so cosine/euclidean/
    knn rankings are **identical** on this data -- confirmed
    empirically during testing. They aren't three independent signals;
    the docstring in `06_retrieval_engine.py` says so explicitly now.
- **Fabricated physical descriptors (09_explainability_engine.py).**
  `get_physical_descriptors()` generated elevation/temperature/
  rainfall/soil/ecoregion from `sum(ord(c) for c in patch_id)` --
  numbers with no relationship to reality, sitting in the same output
  and the same generated sentences as the real, patch-derived
  NDVI/NDWI/NDBI spectral descriptors. Replaced with `geo_lookups.py`,
  which reads real WDPA/WorldClim/SRTM/RESOLVE data and returns `None`
  (never a guess) for anything not downloaded. `generate_explanation()`
  now skips any factor built from a `None` field instead of comparing
  against it or fabricating a value. Soil type was dropped entirely
  (no equally simple free global raster) rather than kept fabricated.
- **Duplicated name-string "protected area" heuristic
  (08_retrieval_dashboard.py and 09_explainability_engine.py).** Both
  files independently inferred protection status by checking whether
  a location's name contained "national park", "reserve", etc. --
  false positives (anything literally named "... Reserve Ranch") and
  false negatives (real protected areas with no such word in their
  name) both possible. Removed; both now use the catalog's real
  `protected_area` value (manually curated in `config.py`, or backed
  by a real WDPA point-in-polygon check once `09` has run).
- **Dead code / misleading docstring (02_preprocess_patches.py).** A
  `load_prithvi_norm_stats()` function loaded Prithvi's HLS training
  stats from a config file, with a comment insisting this was the
  *only* correct approach -- but `main()` never called it; it called
  `compute_custom_norm_stats()` (which computes stats from the actual
  Sentinel-2 patches) instead. Removed the dead function and rewrote
  the comment to describe what the code actually does: a deliberate
  domain-adaptation choice, not a mandate to use Prithvi's original
  stats. See the "HLS vs Sentinel-2" caveat below for the residual
  limitation this doesn't fully solve.
- **Fragile base-location parsing (audit report's Bug #4).**
  `07_evaluate_retrieval.py` previously derived a patch's base location
  via `id.split("_p")[0]`, which breaks if a location id itself
  contains `"_p"`. `02_preprocess_patches.py` now writes an explicit
  `base_id` field on every sub-crop entry; `07` and `10` use that
  field directly, with the string-split kept only as a fallback for
  catalogs generated before this field existed.
- **`evaluation_report.json`'s per-category mAP silently showed `0.0`
  (labeled "poor") for any ecosystem category where grouped evaluation
  had no other location to compare against**, rather than
  distinguishing "no data" from "genuinely bad". Fixed to report
  `None`/`N/A` explicitly in that case (found and fixed during testing
  of the grouped-evaluation feature itself).

### Claimed in the prior audit report, but already fixed / not present in the actual uploaded code

Several bugs listed in `analysis_results.md` didn't reproduce against
the actual files in this project -- worth knowing so you don't
"re-fix" something that isn't broken:

- The `"Temperated Semi-Arid"` typo (Bug #9) -- already read
  `"Temperate Semi-Arid"` in the uploaded `config.py`.
- The `embedding_path` vs `prithvi_embedding` key mismatch (Bug #2) --
  `04_finalize_and_analyze.py` already checked for
  `"prithvi_embedding"`, and `05_create_database_and_dashboard.py`
  already had a fallback (`e.get("prithvi_embedding") or
  e.get("embedding_path")`).
- The deprecated `torch.cuda.amp.autocast` call (Bug #3) -- already
  used the non-deprecated `torch.amp.autocast(device_type=..., ...)`
  form in both `03_extract_embeddings.py` call sites.
- The duplicate `out_path` assignment in `01_acquire_patches.py`
  (Bug #8) -- not present; `out_path` is set once and reused.

### Known limitation, not fully fixable in code

- **HLS-vs-Sentinel-2 normalization mismatch.** Prithvi-100M was
  pretrained on NASA HLS data; this pipeline computes normalization
  stats from the actual acquired Sentinel-2 patches rather than using
  Prithvi's HLS stats (a deliberate choice, see the comment block in
  `02_preprocess_patches.py`). This keeps the input distribution's
  scale reasonable but does not fully resolve the fact that Prithvi's
  *learned features* were shaped by HLS statistics during pretraining.
  Combined with the `num_frames` fix above, Prithvi's results should
  improve, but a residual domain gap versus true HLS input remains.
  If you want a clean side-by-side, you could add a second Prithvi
  run using HLS data directly via NASA's LP DAAC.
- **Resolution mismatches in geo_lookups data.** WorldClim (~1km),
  SRTM (30m), and your Sentinel-2 patches (~2.24km, 10m native) are at
  different scales. Climate values describe the region around a point,
  not something specific to a given patch -- appropriate for climate
  (which varies slowly), not something to read as patch-level
  precision. Elevation is averaged over a 3x3 pixel window to reduce
  sensitivity to sloped terrain, but a single ~90m sample still
  under-represents patches spanning real elevation gradients (e.g. the
  Himalaya/Andes/Alps locations).
- **WDPA point-vs-polygon coverage.** Some WDPA records (especially
  small or poorly-mapped reserves) are stored as a point + reported
  area rather than a true boundary polygon. The point-in-polygon check
  in `geo_lookups.is_protected()` under-detects those. Acceptable at
  this project's scale; worth revisiting if protection status becomes
  a load-bearing model feature at larger scale.
- **Hansen "loss" is not the same as "deforestation".** Hansen Global
  Forest Change's `lossyear` layer records any tree-cover loss --
  logging, fire, storm damage, and disease all register the same way.
  `10_grid_tiling_labels.py`'s labels should be read as "loss risk",
  not specifically "human-driven deforestation risk", unless you
  cross-reference a fire dataset (e.g. MODIS/VIIRS burned area) to
  exclude fire-driven loss.
- **Sub-crop sampling geometry is fixed.** The ±32px offset range for
  sub-crops in `02_preprocess_patches.py` is already maximal given a
  160px crop inside a 224px canvas (`r = 32 + dy` must stay in
  `[0, 64]`). You can't get more diverse sub-crops without changing
  the acquired patch size or crop size -- the grouped evaluation in
  `07` is the mitigation for this, not a fix to the crops themselves.
- **The dashboards (05, 08) got surgical fixes, not a rewrite.** Given
  their size (mostly embedded HTML/JS as Python string templates),
  changes here were scoped to the two specific bugs described above
  (name-heuristic removal) plus benefiting automatically from `07`'s
  JSON restructuring. The `["leaked"]` diagnostic values aren't wired
  into any dashboard chart -- that would be a reasonable next step if
  you want the leakage gap visible in the UI, not just the console
  output of `07`.

---

## Objective 4: Forest-loss risk forecasting (new)

### What it is, and isn't

This is **forest-loss RISK forecasting**, validated against real
historical Hansen Global Forest Change records -- a probability
estimate, not a claim to predict the future in any stronger sense.
Read "loss" as "any Hansen-recorded tree-cover loss" (see the fire/
storm caveat above), not specifically deforestation.

### Sample size

The project's ~30-45 forest/mangrove **named locations** are nowhere
near enough to train a supervised classifier -- with maybe 20% showing
loss in a given year, that's single-digit positive examples. Standard
practice in this literature (GLAD/DETER-style systems) works with
thousands of grid cells, not named places.

`10_grid_tiling_labels.py` fixes this by changing the unit of
analysis: it tiles a `GRID_REGION_BUFFER_KM`-radius region around each
forest/mangrove base location into `GRID_CELL_SIZE_M` cells, and pulls
a real Hansen label for each cell at each observation year in
`OBS_YEAR_START..OBS_YEAR_END`. This turns ~30-45 locations into
hundreds-to-thousands of independent cell-year samples from the same
handful of real regions, all still using free data. Target at least a
few hundred positive examples and a comparable number of negatives
(order 1,000-5,000 total cell-year samples) before trusting an AUC
number from `11_forest_risk_forecast.py`; the script warns you if
there are too few positives to train reliably.

### Validation methodology

`11_forest_risk_forecast.py` uses a **temporal** train/test split
(train on `obs_year <= cutoff`, test on `obs_year > cutoff`) rather
than a random split -- randomly splitting would let the model
implicitly learn from geographically-adjacent "future" cells in the
same training fold, overstating real forecasting skill. Given rare
loss events, **Average Precision (PR-AUC)** is the primary reported
metric, not accuracy or plain ROC-AUC, both of which look
deceptively good under class imbalance.

### The ablation this project set out to answer

Does foundation-model embedding drift add predictive power beyond
cheap driver features (baseline tree cover, distance to existing loss
edges, protection status, climate)? `11_forest_risk_forecast.py` runs
this ablation **only if** an `embedding_drift` column is present in
the features CSV -- it does not fabricate one. Wiring up a real
`embedding_drift` feature means re-running the Sentinel-2 acquisition
+ embedding pipeline at the grid-cell level across multiple years and
computing year-over-year cosine distance per cell -- a real extension
beyond `10`'s current scope. See "Next steps" below.

### Usage

```bash
python 10_grid_tiling_labels.py          # build the cell-year feature table (needs Hansen tile downloads)
python 11_forest_risk_forecast.py        # train + evaluate, saves the best model
python 11_forest_risk_forecast.py --predict <lon> <lat>   # score a single location
```

Both scripts were tested against synthetic rasters/CSVs during
development (tile naming, grid geometry, distance-to-prior-loss
computation, temporal split, class-imbalance handling, and the
ablation logic all verified to produce correct, sensible output) since
this sandbox doesn't have network access to Hansen's GCS bucket. Real
execution needs outbound access to `storage.googleapis.com`.

---

## Next steps (not built here, but the natural continuation)

- **Wire in a real `embedding_drift` feature** for the ablation above:
  extend `10_grid_tiling_labels.py` to acquire + embed each grid cell
  at 2+ points in time and compute cosine distance between them.
- **Fire-dataset cross-referencing** to separate deforestation-driven
  loss from fire/storm-driven loss in the Hansen labels.
- **Wire the `["leaked"]` diagnostic into `08`'s dashboard** so the
  leakage gap is visible in the UI, not just `07`'s console output.
- **Query-by-upload**: let a user submit an arbitrary coordinate,
  acquire+embed it live, and retrieve analogs -- turns the retrieval
  database from a closed self-similarity demo into a queryable tool.
- **SoilGrids integration** in `geo_lookups.py` if soil type is worth
  restoring as a real (not fabricated) explanation factor.
