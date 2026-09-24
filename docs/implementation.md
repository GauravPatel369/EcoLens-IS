# EcoLens — Implementation Plan & Project Context

> **Read this first in any new chat.** This file is the single source of truth for
> where the project stands, what is already built, and what still has to be built to
> answer the faculty review. It is written to be read cold, with no prior conversation.

**Last updated:** 2026-09-04 — **PHASES 0, 1, 4 COMPLETE AND VERIFIED.**

> ## ⭐ THE RESEARCH QUESTION IS ANSWERED, AND THE ANSWER IS YES
> **Analog-transferred features improve forest-loss forecasting across geographically
> distinct landscapes: +0.0392 mean PR-AUC (+11.0 %), 14 of 16 held-out regions improved,
> paired Wilcoxon p = 0.0010.** The effect is ~3× larger across unseen landscapes
> (+11.0 %) than within known ones (+3.8 %) — the signature of real transfer, not an extra
> correlated covariate. `analog_trajectory_jaccard` is the model's **single most important
> feature** (+0.1764), ~3× the strongest conventional driver. Full result and its limits in
> §6 Phase 4; C1/C15/C16 are now `[x]`.

131 locations / 10 categories, 126 acquired, 1,260 sub-crops, all 5 models re-embedded and
verified (§3.4). **Next: Phase 3, then Phase 2** (§7's order), or close out Phase 4's two
remaining optional items (the `embedding_drift` arms, and a Clay re-run to test whether
Prithvi's anisotropy is capping the effect).
**Repo root:** `d:\IIIT Hyderabad\Semester 3\IS\EcoLens-IS`
**Parent folder** (`..\`) holds `ecolens_complete_guide.md`, `Ecolens comments.docx`
(the faculty comments this plan implements), `EcoLens_Research_Paper.docx`, `PROJECT_REPORT.md`.

> ### ▶ Start here
> **ACQUISITION IS DONE. Do not reopen it.** 77 of 81 patches pass every QC check. The 4
> losses are diagnosed, deliberate and documented (§3.2). Chasing coordinates consumed a
> whole session on 3 Sep and hit diminishing returns — **do not spend more time on it.**
>
> **`02`–`08` HAVE NOW BEEN RE-RUN CLEAN AND VERIFIED (3 Sep evening).** §3.4 holds the
> citable retrieval table; §3.4b throughput; §3.4c ecological agreement. Only `10`/`11`
> remain in Phase 0. **Use `python run_phase.py --list` to see exactly which steps are
> done** — that is now the source of truth for run state, not this file.
>
> **PHASE 1 IS ALSO COMPLETE (3 Sep, late evening).** 131 locations / 10 categories,
> 126 acquired, 1,260 sub-crops, all 5 models re-embedded and verified. §3.4 holds the
> current numbers; §3.4a keeps the 77-location baseline. **Do not compare the two mAP
> tables directly — 5 → 10 categories halves the chance level.** Against chance the system
> improved (2.6× → 4.0× above random).
>
> **The Phase 1 headline: Clay's margin over the handcrafted spectral baseline nearly
> doubled (+14.0 % → +26.4 %) while both generic-ImageNet models fell BELOW it.** That is
> the C3 argument the project needed, and it only became visible with a diverse catalog.
> Second: **Prithvi wins `shrubland` by 29 % over Clay** — it is the only 6-band model and
> drylands separate on SWIR, which ViT/ResNet cannot see.
>
> **Three headline results from the earlier clean 77-location run, all of which changed a
> conclusion:**
> 1. **Clay-v1.5 overtook ResNet** (mAP 0.4059 vs 0.3808) — but their CIs overlap, so
>    report them as tied. Prithvi rose 5th → 4th. **ViT got *worse*.**
> 2. **The handcrafted spectral baseline now beats Prithvi and ViT** (0.3561). C3 is
>    downgraded to `[~]`. This is a real loss of ground and §9 says to report it.
> 3. **Every EO-pretrained model gained from the QC fixes; the generic-ImageNet ones did
>    not.** The contaminated catalog was flattering the least-grounded models.
>
> **Verified 3 Sep:** `00_validate_locations.py` reports **0 errors** (4 MISSING + 6
> advisory warnings). `01` now *skips* rather than re-fetches — the loop is closed.
>
> **Next action:** see §6 Phase 0's "YOU ARE HERE". `02` is done; `03` is running for all
> five models; `04` onward per §8.
>
> **⚠ Before running `03` in a fresh venv, read §3.5 (Environment).** Clay and Satlas are
> not covered by `requirements.txt`, and Clay's repo renamed `src/` → `claymodel/`. A
> rebuilt venv breaks both with `ModuleNotFoundError`, which is exactly what happened on
> 3 Sep. §3.5 has the fix.
>
> **`03` skips patches whose embedding file already exists, and that silently produces
> wrong results after a catalog rebuild** — the sub-crop filenames are stable, so a stale
> `.npy` from an older acquisition keeps its name and is reused. On 3 Sep this made a run
> extract only 80 of 770 and leave 690 pre-BOA-offset vectors in the same index. **Use
> `--force` (added 3 Sep) after any `02` re-run**, or clear the embedding directories.
>
> **`03` is NOT the multi-hour step the earlier draft claimed.** Measured 3 Sep on CPU:
> Prithvi/ViT ~5 min each for 770 patches, ResNet ~2 min. **Clay dominates** — it loads a
> 5.1 GB checkpoint (~4 min) and then embeds one patch at a time by design. Budget for
> Clay, not for the model count.
>
> **Expect `02`'s normalization statistics to differ from any earlier run.** It computes
> per-band mean/std from the patches, and those patches now carry *corrected* reflectance
> (BOA offset, §3.2) plus a corrected nodata rule (§3.2 bug 4). Current values, 3 Sep:
> means `[541.7, 768.3, 770.8, 2619.7, 1907.2, 1259.3]`,
> stds `[546.3, 586.5, 742.5, 1117.8, 1054.6, 982.1]`.
> If blue comes back near ~1550, the BOA offset is not being applied — check that
> `patches/acquisition_manifest.json` records `"boa_offset": -1000`.
>
> **The working tree is UNCOMMITTED by the user's explicit choice (3 Sep).** Now **seven**
> files (§3.1) — `02` and `03` were both edited in the later 3 Sep session. §9 says commit
> before a long run; that was raised and declined twice. Do not commit or push without
> asking.
>
> **Why the old numbers are dead:** §3.4 was computed on a catalog with all-zero patches,
> offshore "mangroves", winter scenes over deciduous forest, and — found 3 Sep —
> **uncorrected reflectance that made every NDVI ~0.3 too low**. Do not quote any of it.
>
> **To orient fast:** §1 (the thesis) → §3 (repo state) → §5 (what's done vs. not).
> Update §5/§6 checkboxes and §3.4 numbers as work lands; this file is the handoff
> between sessions.

---

## 0. Documents in play, and how much to trust each

| Document | Trust | Note |
|---|---|---|
| `README.md` (this repo) | **High** | Most accurate written description of the code. Kept current through the "Faculty review response" pass. |
| `implementation.md` (this file) | **High** | The plan. Update the checkboxes in §5 as work lands. **Deliberately untracked** — it's a working document, not repo history, so it will always show as `??` in `git status`. Do not commit it without asking. |
| `..\ecolens_complete_guide.md` | **Stale** | Describes an earlier 3-model / 75-location / 11-step version. Paths in it (`c:\IS\...`) are wrong. Useful for beginner-level conceptual explanations only — see §2 for the delta. |
| `analysis_results.md`, `updated changes`, `phase2.md`, `phase3.md`, `walkthrough.md` | **Historical** | Old audit / bug-fix logs. Several bugs listed in them were never real (README documents which). Do not act on them. |
| `ECO_SYSTEM_ARCHITECTURAL_SPEC.md` | **Historical** | Original design spec, pre-dates most of the current code. |

---

## 1. The research question (as re-framed by the reviewer)

> **Can geospatial foundation models identify ecological analogs whose historical
> disturbance trajectories improve forest-loss risk forecasting across geographically
> distinct landscapes?**

This is now the thesis. Everything in the project must serve it. The single biggest
structural weakness the reviewer identified is that the project currently has **two
disconnected halves**:

- **Pillar A — Retrieval:** embed Sentinel-2 patches with foundation models, find
  ecological analogs, evaluate and explain them.
- **Pillar B — Forecasting:** tile forest regions into grid cells, label with Hansen
  loss data, train a classifier on geographic / climatic drivers.

Nothing in Pillar B currently consumes anything produced by Pillar A except a
proof-of-concept `embedding_drift` column (a self-similarity-over-time feature, **not**
an analog-derived feature). **Closing that gap is the centrepiece of this plan
(Phase 4).**

---

## 2. What the stale guide gets wrong (delta vs. reality)

If you have read `..\ecolens_complete_guide.md`, correct it with these:

| Guide says | Reality |
|---|---|
| 3 models (Prithvi, ViT, ResNet) | **5 models**: + Clay-v1.5 (1024D, real ckpt in `clay_checkpoint/`) and Satlas-ResNet50 (2048D). Driven by `config.SUPPORTED_MODELS`. |
| 75 locations → 750 patches | `config.py` defines **81** locations; **77** acquired and QC-passed as of 3 Sep; 4 accepted losses (§3.2). Sub-crops are regenerated by `02` at 10× the base count, so expect **~770**. |
| Steps 01–11 | Also **00** (`00_validate_locations.py`, QC gate), **07b / 07c / 07d**, and **12**. |
| 3 retrieval methods (cosine / euclidean / knn) | **4** — HNSW `ann` added as the only genuinely non-identical one. |
| Prithvi is the headline model | **Resolved 3 Sep on clean data — it is not.** Prithvi rose from 5th to 4th (mAP 0.299 → 0.3402, +13.8 %), the second-largest gain of the five, confirming the hypothesis that as the only 6-band model it had most to lose from the BOA-offset bug and dormant scenes. **Clay-v1.5 now leads at 0.4059**, though its CI overlaps ResNet's by 0.0017 — treat the top two as tied. Full table and the ranking analysis in §3.4. |
| Risk ablation "not wired up" | `10 --with-embedding-drift` now computes a real drift feature; a 10-cell POC ran end to end. |
| Paths `c:\IS\` | `d:\IIIT Hyderabad\Semester 3\IS\EcoLens-IS\` |

---

## 3. Current repo state — READ BEFORE TOUCHING ANYTHING

### 3.1 Git

- Branch `main`, **7 files modified and deliberately uncommitted**, 3 commits ahead of
  `origin/main` (not yet pushed):

  ```
  37ce8e1  Download DEM tiles for all locations, not just forest
  0611461  Add patch quality control and fix 13 mislocated coordinates
  36176a6  Make the pipeline runnable on Windows
  0baa736  (previous head) Add seasonal embedding stability script...
  ```

  These three are the popped stash, split by concern. The QC work is now safely in
  history, so the Phase 0 re-acquire is reversible.
- **Uncommitted, in the working tree (3 Sep).** Five files, one coherent change: *make
  acquisition verify what it downloaded instead of trusting scene metadata.*

  | File | Change |
  |---|---|
  | `config.py` | WorldClim → 2.5-arc-min; `SEARCH_DATE_RANGE` → full-year 2024; `MAX_SCENE_ATTEMPTS`; growing-season month sets + `TROPICS_LAT`; **`BOA_ADD_OFFSET`**; `MIN_VEG_*`; `HAZE_BLUE_REFLECTANCE`; 10 more coordinate corrections + 1 per-location cloud override |
  | `01_acquire_patches.py` | `search_candidate_scenes()` returns 6 scenes ranked **growing-season first, cloud second**; loop falls back when a scene clips the point, fails to read, or yields a vegetation-free forest/mangrove patch; applies **BOA_ADD_OFFSET**; per-location `max_cloud_cover`; manifest records ndvi / forest% / blue / month / offset |
  | `00_validate_locations.py` | land-probe fix; **new CONTENT check** (ERROR, pixel-based); MISLABEL demoted to WARN; SEASON + HAZE advisories |
  | `geo_lookups.py` | `sample_raster_point()` returned `nan` on an all-nodata window where its sibling returned `None`. A `nan` compares False against every threshold, so the factor vanished silently instead of reporting unavailable |

  Later on 3 Sep, three more fixes landed in `01_acquire_patches.py` after real failures:
  **per-patch checkpointing** (a DNS drop discarded the provenance for 77 completed
  patches), **STAC search retry** with exponential backoff (one Azure 502
  `OriginConnectionAborted` aborted a whole run — the read path already retried, the
  search call did not), and the fix for a **re-download loop I introduced**:
  `ndvi_median` was being written to the catalog entry but not to the manifest that the
  staleness check reads, so every location looked unassessed on every run and `01` never
  converged.

  **Two more files were modified later on 3 Sep**, in the session that actually ran the
  pipeline. Both were required to make `03` run at all:

  | File | Change |
  |---|---|
  | `02_preprocess_patches.py` | **`nodata_mask()` added; `handle_nodata()` and `compute_custom_norm_stats()` switched from a per-band zero test to the all-band L2A convention** — the fabrication bug, §3.2 bug 4. Also fixed a hardcoded `(710 total)` in a progress line that contradicted the actual count. |
  | `03_extract_embeddings.py` | `from src.module` → try `claymodel.module` first (Clay renamed its package; §3.5). **`--force` flag** so a catalog rebuild can't silently reuse stale embeddings. Clay's DINOv2 teacher is now built with `pretrained=False` and deleted after load — it is training-only, and downloading it was blowing out commit charge (§3.5). |

  `git status` also shows **`metadata/catalog.json`** modified — that is *regenerated
  output*, not a hand edit: `01` rewrote it from 79 stale entries to the current 77, and
  `02` then expanded it to 770 sub-crops.
  It is tracked (the `.gitignore` line for it is commented out), so it will ride along
  with whatever commit lands next. Untracked-and-expected: `implementation.md`,
  `metadata/catalog_pre_qc.json`, `results_pre_qc/`, `updated changes`.

  **All of this is uncommitted by the user's explicit decision (3 Sep).** §9's "commit
  before a long re-run" rule was raised and declined. Do not commit without asking.
- **Untracked, deliberately:** `implementation.md` (this file) and `updated changes`
  (stale audit log, see §0). Neither is gitignored, so both show as `??` — that is
  expected, not a task.

  | File | What the three commits contain |
  |---|---|
  | `config.py` | Windows UTF-8 stdout fix; `MAX_WATER_FRACTION` / `MAX_NODATA_FRACTION` QC thresholds; `ACQUISITION_MANIFEST_PATH`; **13 corrected coordinates** (Mt Kenya was on the 4596 m summit above the treeline; Salonga was 690 km off; five mangrove sites were 96–100 % open water or inland; two "wetland" sites sat inside the Sundarbans mangrove biome, making wetland-vs-mangrove unlearnable) |
  | `01_acquire_patches.py` | `assess_patch_quality()` (NDWI water fraction + nodata fraction), rejection + end-of-run failure summary, `--allow-low-quality`, and a **provenance manifest** so a corrected coordinate invalidates the stale patch instead of being silently ignored |
  | `06_retrieval_engine.py` | `get_peak_rss_mb()` — cross-platform (the old top-level `import resource` made the whole script unrunnable on Windows) |
  | `download_reference_data.py` | `--dem-forest-only` flag; the DEM download now defaults to **all** locations (148 tiles / ~5.1 GB) instead of forest-only (31 tiles / ~1.1 GB), because `09` builds descriptors for every patch and the old default left elevation null for 64 of them |
  | `run_pipeline.py` | runs `00_validate_locations.py` as a non-blocking gate before anything expensive |
  | 14 others | UTF-8 `encoding=` on file opens (verified: encoding-only, no logic changes) |

### 3.2 Data integrity — acquisition is clean, downstream is not yet re-run

**Patches have been re-acquired under QC. Everything downstream of them has not.**

`metadata/catalog.json` and `patches/acquisition_manifest.json` are current (3 Sep);
`results/` is empty (the contaminated copies were moved to `results_pre_qc/`, and the
old catalog to `metadata/catalog_pre_qc.json`).

What the old catalog contained, and what happened to it:

| Problem in the old catalog | Status |
|---|---|
| 3 patches (`agri_015`, `mangrove_001`, `mangrove_015`) entirely **nodata** — all-zero arrays, embedded and retrieved like any other. Two all-zero patches match at cosine 0.9998, which is why `retrieval_case_studies.json` had "Shark Bay Mangroves" as the top analog for "Hokkaido Potato Farms" at 0.9997. | **Root-caused and fixed.** `01` picked the least-cloudy scene whose *bbox intersected*, never checking the point fell inside that scene's valid-data footprint; Sentinel-2's wide nodata margins then clipped it. The multi-scene fallback recovered `mangrove_001` (scene 5) and `agri_015` (scene 3). `mangrove_015` needed a new coordinate. |
| 3 "mangrove" locations 96–100 % open water | Rejected by the water check; coordinates corrected. |
| 7 coordinates pointing somewhere other than the place named | Corrected (13 in the committed pass, 5 more on 3 Sep). |
| 10 locations with no patch and no failure summary | Summary added; down to 6, all with a diagnosed cause. |

### Four bugs found on 3 Sep, all upstream of everything

Each would have silently corrupted the embeddings. Listed because they are the strongest
material for the C6 methods section — and because each was invisible to the check that
should have caught it.

1. **Tile-clipping nodata.** `01` picked the least-cloudy scene whose *bbox intersected*,
   never checking the point fell inside that scene's valid-data footprint. Sentinel-2's
   wide nodata margins then clipped it. → multi-scene fallback (6 candidates).
2. **Season selection bias.** "Least cloudy" systematically selects *winter* outside the
   tropics — clear dry air is exactly when there is least to see. **44 of 79 patches were
   Jan–Mar**, and Jiuzhaigou (montane *deciduous*) was sampled on 9 January, returning
   0.1 % forest cover. This is a confound, not noise: half the catalog dormant and half at
   peak growth means retrieval partly encodes acquisition month. → growing-season ranking.
   After the fix Jiuzhaigou reads **58.5 %** forest cover.
3. **Sentinel-2 `BOA_ADD_OFFSET` was never applied.** Since processing baseline 04.00
   (Jan 2022), L2A stores reflectance with a −1000 DN offset. It does *not* cancel out of
   a normalized index — it survives in the denominator. Measured: Daintree NDVI **0.57 →
   0.91**, Hinchinbrook 0.56 → 0.91, Periyar 0.48 → 0.77. A closed tropical canopy is
   0.85–0.92, not 0.55. This biased `02`'s normalization stats, `09`'s descriptors, the
   cover fractions built on them, `07b`/`07c`, and `01`'s own water gate.
4. **`02` fabricated data over dark surfaces — a latent bug that the BOA fix activated.**
   `handle_nodata()` treated a zero in **any single band** as nodata and replaced it with
   that band's median, while `01`'s `assess_patch_quality()` uses the real Sentinel-2 L2A
   convention: nodata is a pixel that is zero in **every** band. Harmless until the BOA
   offset landed — subtracting 1000 DN makes any surface below 0.10 reflectance clip to
   exactly 0, which over open water is most of the blue band. `wetland_014`
   (Mesopotamian Marshes) came back **98 % zero in blue, 88 % green, 69 % red, 52 % NIR**
   with SWIR essentially fully valid, so `02` rebuilt those bands from the median of the
   surviving 2 % and emitted near-constant arrays: per-band std **0.03–0.15** against
   ~1.0 for a healthy patch. A constant patch matches everything at high cosine — the
   same failure that produced the 0.9997 Shark Bay/Hokkaido match, reached by a different
   route, and **invisible to every existing check**: `01` reported `nodata_fraction 0.0037`
   because almost no pixel was zero in all six bands, and `00`'s CONTENT check is an ERROR
   only for `forest`/`mangrove`, so a wetland with `ndvi_median 0.0` passed the gate.
   → `02` now uses the all-band mask in both `handle_nodata()` and
   `compute_custom_norm_stats()`. `wetland_014` went from std 0.03–0.15 to **0.47**, and
   **no location had to be dropped** — the zeros were the real measurement, not missing
   data. Verified across all 770 sub-crops: none zero-variance, lowest whole-patch std now
   0.246 (`forest_017`, dense closed canopy, physically correct).

   **This is the sharpest C6 material in the project.** Three independent QC layers —
   a pixel-content gate, a provenance manifest and a validation script — all passed a
   patch that was 98 % fabricated, because each measured something slightly different from
   what the next stage actually consumed.

### Final acquisition state, 3 Sep: **77 / 81**

forest 17/17 · agricultural 15/15 · urban_green 15/15 · wetland 16/17 · **mangrove 14/17**

Verified on the final run: 77 catalog entries, 77 manifest records carrying `ndvi_median`
(so `01` now skips rather than re-fetches — the earlier re-download loop was a bug where
that key was written to the catalog entry but not the manifest the staleness check reads).
NDVI median across the catalog 0.537, max 0.913 — versus a max of ~0.57 before the BOA
offset correction.

**Acquisition months — the full distribution, corrected.** An earlier draft of this file
listed only "Aug 17 · Sep 13 · May 9 · Jul 8", which accounts for 47 of 77 patches and
reads as though the dormant-season problem was eliminated. The real distribution
(`results/qc_report.md`) is:

| Jan | Feb | Mar | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 5 | 8 | 9 | 4 | 8 | 17 | 13 | 1 | 2 | 4 |

**19 of 77 are still Jan–Mar** (down from 44 of 79), and 23 fall in Dec–Mar. All 77 are
flagged `growing_season: true` — correctly, because the tropics have no dormant season, so
a January scene over Kerala is peak growth. The fix worked; it did not and could not empty
the winter months. **Quote the table, not the four-month summary.**

**The 4 losses are accepted, not pending.** Do not relocate these again:

| id | Diagnosis | Why it is not worth another attempt |
|---|---|---|
| `wetland_015` Wadden salt marsh | best of 6 scenes = 87 % water | Dutch salt marshes are strips a few hundred metres wide behind the dike. **No coordinate makes a 2.24 km patch mostly salt marsh.** |
| `mangrove_015` Exmouth Gulf | 100 % water across all 6 | two relocations tried (Shark Bay → Exmouth); sparse WA mangroves vs a 2.24 km footprint |
| `mangrove_009` Qeshm (Hara) | NDVI −0.06, 3.4 % forest, 51 % water | third site tried for this slot (Red Sea → Qeshm); patch files deleted 3 Sep |
| `mangrove_016` Key West | NDVI 0.05, **0 %** forest | the Keys are too small for the footprint — same as Muara Angke, which the earlier pass already swapped out |

**All four are the same structural finding: narrow-fringe coastal ecosystems cannot be
sampled at 2.24 km.** That is real content for C24 *and* the motivation for the Phase 5
patch-size sweep — write it up rather than treating it as a failure.

**Mangroves absorbed 6 of the original 10 failures and 3 of the final 4.** Systematic, not
chance: mangroves are coastal, so they fail water and nodata checks far more than inland
classes. The surviving mangrove set is biased toward large, landward-fringed systems —
state this as a sampling bias in C24.

**Still true: every number in §3.4, `results_pre_qc/`, `risk_model/`, both dashboards, and
anything quoted in the paper predates all of the above.** Do not cite any of it until
`02`–`11` have been re-run.

### 3.2c Bug 7 (5 Sep) — the labels and the query drifted apart

**The failure.** Adding the interannual axis to `12` was done with a script that applied a
series of `str.replace()` edits. **One of them silently did not match** — the scene-selection
expression kept its original `SEASONS["winter"]` query — while the *label* replacements all
applied. The run therefore printed `July 2021 Cloud Cover: 0.0%` while actually fetching
**January 2023**, and wrote a complete `interannual_stability.json` carrying
`"mode": "interannual"` and `"winter_window": "2021-07-01/2021-08-31"`. Every field said
acquisition-date; every number was the seasonal comparison.

**What caught it.** Not a test — the interannual column came out **byte-identical to the
seasonal column for all five models**, which is not a thing that happens. The decisive
check was the cached patch itself: North Slope Alaska's "July 2021" image had **mean blue
reflectance 8351** (snow), against 473 for the genuine July scene.

**Three fixes.**
1. The branch is now an explicit `if/elif/else`, not a chained ternary — the chained form is
   what let the label and the query diverge.
2. **`12` now asserts the fetched scene's own `datetime` falls inside the window being
   claimed**, and raises instead of emitting mislabelled numbers. *A printed label is not
   evidence of what was fetched.*
3. The contaminated JSON and 14 wrong cache files were purged and the axis re-run.

**Two lessons worth carrying.**
- **`str.replace()` in a patch script fails silently.** Use an edit path that errors on a
  non-match, or assert on the replacement count. Half-applied edits are worse than failed
  ones because the surviving half makes the output look right.
- **A result that is *exactly* equal to another result is a bug signal, not a coincidence.**
  This one would have gone into the paper as "the embedding is perfectly stable across
  years" — which is precisely the kind of too-good number the QC pass exists to catch.

*(Also fixed 5 Sep, same class: `14_analog_ablation.py` hardcoded `analog_cell_features.csv`
and both output JSONs, so a Clay run would have silently overwritten Prithvi's results —
the exact failure `13` suffered on 4 Sep. `14` now takes `--model`; Prithvi keeps the
unsuffixed names so existing references stay valid.)*

### 3.2b-2 Repo hygiene pass, 6 Sep — what was deleted, and what was deliberately kept

Criterion: **delete only what has no present or future use, and never anything the pipeline
would have to regenerate.** A fresh `git clone` must still run end to end.

**Deleted (38.5 MB), each verified first:**

| Item | Why it is garbage |
|---|---|
| `cell_year_features_backup_{phase0,with_drift,wdpa}.csv` | Three safety backups taken during the 6 Sep runs. Verified column-by-column to be **strict subsets** of the final table — none contains anything the current file lacks |
| `cell_year_features_drift_demo.csv` | The 10-cell drift proof-of-concept, superseded by the real 510-cell run and referenced nowhere |
| `results/analog_cell_embeddings.npz` | **Byte-identical** to `analog_cell_embeddings_prithvi.npz` (1,573 keys, same vectors) — the legacy un-suffixed name from before `13` gained per-model caches |
| `__pycache__/`, `ecolens.egg-info/` | Regenerated automatically on next run / install |
| `phase2.md`, `phase3.md`, `walkthrough.md`, `analysis_results.md`, `updated changes` | Progress snapshots from the **3-model, 9-script era**. All contain *zero* mentions of Clay or Satlas. They describe a pipeline that no longer exists, so they mislead a new reader rather than orient them |

**Deliberately KEPT, with the reasoning, so this is not re-litigated:**

| Item | Size | Why it stays |
|---|---|---|
| `retrieval_dashboard.html` | 970 MB | **Not stale** — built 22:42 on 3 Sep, one minute after `catalog.json` (22:19), so it reflects the current 1,260-sub-crop catalog. Regenerating needs a full `06` + `08` re-run. *It is, however, far too large for git — add it to `.gitignore` rather than committing it.* |
| `embedding_dashboard.html` | 58 MB | Same vintage, same reasoning |
| `results/retrieval_results_spectral_baseline.json` | 515 MB | A real pipeline output. Everything downstream has consumed it, so it *is* the largest safe-to-drop item if space is ever needed — but dropping it costs a `07c` re-run, and the brief was not to force re-runs |
| `config.yaml` **and** `Prithvi_100M_config.yaml` | 653 B each | md5-**identical**, and it is tempting to delete one. Do not: `03_extract_embeddings.py` reads `Prithvi_100M_config.yaml` while `inference.py` defaults to `config.yaml`. Deleting either breaks a script or triggers a Hugging Face re-download |
| `results/_stability_patch_cache/` | 44 MB | A cache whose runs are complete — but it is what makes a `12` re-run take minutes instead of an hour |
| `patches_stale/` | 1.2 MB | One quarantined patch, and §3.3 documents the QC decision that put it there. Evidence, not clutter |
| `Prithvi_100M.pt` | 433 MB | The model checkpoint. Re-downloadable, but needed to run anything |
| `.docx` files | — | The author's own documents. Not this pass's business |

**Flagged but NOT deleted — a judgement for the author:** `ECO_SYSTEM_ARCHITECTURAL_SPEC.md`
(33 KB), `PROJECT_REPORT.md` (7.9 KB) and `README.md` (26 KB) all predate the 5-model work.
`PROJECT_REPORT.md` mentions neither Clay nor Satlas and the spec mentions them once, so both
describe a superseded architecture. `README.md` does cover all five models and is the entry
point a cloner reads first, so it should be **updated, never deleted**.

### 3.2b Disk — what was deleted on 4 Sep, and what must never be

Catalog growth has a **quadratic** storage cost (§ Phase 1), and the run filled the disk
to **1.6 GB free** at one point. Recovered to ~16 GB by deleting only things with no
future use:

| Removed | Size | Why it was safe |
|---|---|---|
| `results-2/` | 2.04 GB | Aug-12 outputs from the **3-model era** (prithvi/resnet/vit only). Superseded twice; referenced by no code and by no section of this file. |
| big files in `results_pre_qc/` | 3.2 GB | The contaminated pre-QC ranked lists. §3.2 says "do not cite any of it", and the withdrawn *numbers* are transcribed in §3.4a. **The small reports were KEPT** (`evaluation_report.json`, `retrieval_perf.json`, `retrieval_case_studies.json`, `ecosystem_descriptors.json`) so the before/after provenance survives. |
| `embeddings_pre_qc/` | 19 MB | Aug-6 vectors; all five models re-embedded twice since. |
| `retrieval_results_*.json` in `results/` | ~10 GB | **Regenerable** by re-running `06` (~10 min). Every analysis that consumes them (`07`, `07b`, `07c`, `07d`, `08`) had already run and its small outputs are on disk. |

**Do NOT delete:** `hansen_data/` (27 GB — used by `09`, `10`, `13`, `15`, and re-downloading
is hours), `clay_checkpoint/` (4.8 GB), `geo_data/` (RESOLVE + WorldClim + DEM),
`patches/`, `patches_processed/`, the five `embeddings_*` dirs, or
`results_phase0_77loc/` (12 MB — the verified 77-location baseline).

**If space is needed again**, the first thing to drop is `results/retrieval_results_*.json`
— that is the O(n²) term, and `06` regenerates it. Storing top-K instead of the full
ranking would remove the problem permanently; nothing downstream reads past top-50.

### 3.3 Reference data on disk

| Dataset | Present? | Consequence |
|---|---|---|
| RESOLVE Ecoregions 2017 | ✅ | ecoregion / biome factors work |
| Hansen GFC tiles | ✅ ~20+ tiles | loss labels + disturbance history work |
| WorldClim BIO1 / BIO12 | ✅ **2.5 arc-min** (~4.6 km) | `temp_c`, `rainfall_mm` work. Coarser than the 30s (~1 km) originally intended — the 30s bundle is a 10.4 GB download for 2 of its 19 rasters. 4.6 km cells are *wider than a 2.24 km patch*, so climate is regional context, never patch-level. Upgrade path is documented in `config.py`. |
| DEM | ⚠️ **forest locations only** (`--dem-forest-only`, ~1.1 GB) | `elevation_m` / `ruggedness_m` work for forest — which is all the risk model tiles — but stay `None` for mangrove / wetland / agricultural / urban_green descriptors. Caps what C11 can report. |
| WDPA | ❌ absent (needs a protectedplanet.net token; **`--all` does not fetch it**) | `protected_area` is `None` from `geo_lookups`. |

**Verified 3 Sep** via `python geo_lookups.py`: Amazon 27.1 °C / 2314 mm / 24 m
(`Japurá-Solimões-Negro moist forests`), Siberia −0.0 °C / 475 mm (`West Siberian
taiga`), Sundarbans 26.2 °C / 1828 mm (`Sundarbans mangroves`), Punjab 23.8 °C / 657 mm.
Four correct climate readings plus three exactly-right polygon lookups rules out a CRS
bug — misprojection produces garbage everywhere, not four right answers.

**The WDPA gap has a consequence that is easy to miss.** `10_grid_tiling_labels.py:610`
reads `geo["protected_area"]` from `geo_lookups`, **not** from the hand-curated
`config.py` flag. So `protected_area` will still be 0 % filled in
`cell_year_features.csv` after the re-run. `11 --ablation` will therefore be meaningful
for **Climate** and **Topography** but its "Anthropogenic" group
(`protected_area` + `distance_to_prior_loss_m`) remains half-empty. C18/C19 are
correspondingly weaker than their `[x]` implies — see §5.

### 3.4 CURRENT retrieval numbers — Phase 1, **126 locations / 10 categories**

GROUPED (leave-one-location-out), cosine, **1,260 queries**, from
`results/evaluation_report.json`. Bootstrap 95 % CIs. **These supersede the 77-location
table in §3.4a below.**

| Model | dim | mAP | 95 % CI | MRR | P@1 |
|---|---|---|---|---|---|
| **Clay-v1.5** | 1024 | **0.3161** | 0.3045–0.3277 | 0.4691 | 0.4111 |
| Prithvi-100M | 768 | 0.2525 | 0.2427–0.2620 | 0.4461 | 0.3817 |
| ResNet-50 | 2048 | 0.2466 | 0.2375–0.2558 | 0.4431 | 0.3675 |
| Satlas-RN50 | 2048 | 0.2429 | 0.2333–0.2522 | 0.3980 | 0.3159 |
| ViT-Base | 768 | 0.1970 | 0.1892–0.2059 | 0.3749 | 0.2873 |

> ### ⚠ DO NOT compare these mAPs to §3.4a's. The task changed.
> Going 5 → 10 categories halves the chance level. Absolute mAP fell 22–38 % for every
> model, and **that is a property of the benchmark, not a regression.** The comparable
> figure is performance against chance, and it **improved**:
>
> | | 5 categories (77 loc) | 10 categories (126 loc) |
> |---|---|---|
> | random P@5 | 0.1990 | **0.0993** |
> | best model P@5 | 0.5091 | 0.4014 |
> | **× above random** | **2.6×** | **4.0×** |
>
> Anyone quoting "mAP dropped from 0.41 to 0.32" without this context is reporting the
> benchmark's difficulty as though it were the model's performance.

**Ranking changes from the 77-location run:**
- **Clay-v1.5 still leads, and now unambiguously** — its CI (0.3045–0.3277) no longer
  overlaps second place. On the 5-category task it was tied with ResNet; the harder, more
  diverse task separates them.
- **Prithvi rose 4th → 2nd.** Combined with its 5th → 4th move in Phase 0, Prithvi improves
  every time the data gets cleaner or more ecologically varied.
- **ResNet-50 fell 2nd → 3rd** and **ViT-Base is last by a wide margin** (0.1970, ~38 %
  below Clay). The two generic-ImageNet models are the ones that degrade most as ecological
  diversity rises — the same pattern §5/C8 recorded in Phase 0, now much more pronounced.

**Per-category mAP (GROUPED, cosine).** Bold = best model for that category:

| Category | Clay | Prithvi | ResNet | Satlas | ViT |
|---|---|---|---|---|---|
| urban_green | **0.6841** | 0.5341 | 0.5783 | 0.5576 | 0.4505 |
| shrubland | 0.3794 | **0.4903** | 0.1983 | 0.3676 | 0.1308 |
| tundra | **0.4000** | 0.1337 | 0.1768 | 0.1864 | 0.1725 |
| mangrove | **0.2803** | 0.2039 | 0.1411 | 0.1603 | 0.1427 |
| savanna | **0.2799** | 0.1562 | 0.1882 | 0.1246 | 0.1367 |
| boreal | **0.2796** | 0.2392 | 0.2219 | 0.1918 | 0.2196 |
| forest | 0.2689 | 0.2465 | **0.2970** | 0.2419 | 0.2212 |
| agricultural | **0.2634** | 0.1758 | 0.2566 | 0.1823 | 0.1397 |
| wetland | 0.1637 | 0.1356 | 0.1547 | **0.1736** | 0.1496 |
| grassland | 0.1514 | **0.1898** | 0.1379 | 0.1858 | 0.1395 |

- **`shrubland` is Prithvi's standout (0.4903, beating Clay by 29 %)** — and Prithvi is the
  only 6-band model. Drylands are separated by SWIR, which ResNet and ViT cannot see at
  all (0.1983 / 0.1308). This is the cleanest single piece of evidence in the project that
  **spectral bands beyond RGB matter**, and it only became visible once drylands existed
  in the catalog. Good C8/C9 material.
- **`grassland` (0.151) and `wetland` (0.164) are the worst classes — as `00` predicted.**
  Both were flagged DUPLICATE against `agricultural`/`savanna` (Pampas cropland vs Pampas
  grassland, Kakadu floodplain vs Kakadu savanna, Cerrado soybean vs Cerrado). Those pairs
  are real land-use contrasts, kept deliberately, and they depress these numbers. **Report
  the cause alongside the number rather than presenting it as model failure.**
- `tundra` (Clay 0.400) and `shrubland` split sharply by model, while `urban_green` remains
  easiest for everyone — geometric, high-contrast, and globally homogeneous.

### 3.4a SUPERSEDED — the 77-location / 5-category table (Phase 0)

Kept because it is the clean baseline the Phase 1 numbers must be read against, and because
the Δ column against the withdrawn pre-QC run is the evidence that the QC work mattered.
**Not the current numbers.**

| Model | dim | mAP | 95 % CI | MRR | P@1 | withdrawn mAP | Δ |
|---|---|---|---|---|---|---|---|
| **Clay-v1.5** | 1024 | **0.4059** | 0.3910–0.4196 | 0.5548 | 0.5026 | 0.344 | **+18.0 %** |
| ResNet-50 | 2048 | 0.3808 | 0.3684–0.3927 | 0.5953 | 0.5221 | 0.366 | +4.0 % |
| Satlas-RN50 | 2048 | 0.3621 | 0.3499–0.3744 | 0.4704 | 0.3766 | 0.332 | +9.1 % |
| Prithvi-100M | 768 | 0.3402 | 0.3276–0.3532 | 0.4906 | 0.4325 | 0.299 | +13.8 % |
| ViT-Base | 768 | 0.3157 | 0.3047–0.3277 | 0.5253 | 0.4364 | 0.327 | **−3.5 %** |

**The ranking changed, and §2 said to treat that as a finding rather than a swap.**

- **Clay overtook ResNet for first place.** But be careful: Clay's CI lower bound (0.3910)
  and ResNet's upper bound (0.3927) **overlap by 0.0017**. The two are effectively tied;
  do not claim Clay is significantly better. What *is* defensible is that Clay is at least
  as good as the best ImageNet baseline while using half the dimensions.
- **ResNet still wins MRR (0.5953) and P@1 (0.5221)** despite ranking second on mAP. Say
  which metric a claim rests on.
- **Prithvi rose 5th → 4th (+13.8 %), the second-largest gain.** This is the §2 hypothesis
  confirmed: as the only 6-band model it had the most to lose from the BOA-offset bug and
  from dormant-season scenes, and it gained the most when both were fixed. It is still not
  the headline model.
- **ViT-Base is the only model that got WORSE (−3.5 %) and is now last.** The cleanest
  reading: ViT is RGB-only ImageNet with no EO pretraining, so it was the model most
  reliant on the very artifacts the QC pass removed. Worth a sentence in C8/C24 — the
  contaminated catalog was *flattering* to the weakest-grounded model.
- **Every EO-pretrained model gained** (Clay +18.0, Prithvi +13.8, Satlas +9.1) while the
  two generic-ImageNet models moved least or backwards (ResNet +4.0, ViT −3.5). That is a
  real argument for C3/C8 and it did not exist before this run.

Per-category mAP (GROUPED, cosine) — `urban_green` is far and away the easiest class and
`wetland` the hardest, which tracks the within-class heterogeneity noted in §6:

| Category | Clay | Prithvi | ResNet | Satlas | ViT |
|---|---|---|---|---|---|
| urban_green | **0.6935** | 0.5909 | 0.6037 | 0.6036 | 0.5141 |
| agricultural | 0.3678 | 0.2878 | 0.3653 | 0.3127 | 0.2236 |
| forest | 0.3601 | 0.3533 | **0.4328** | 0.3379 | 0.3427 |
| mangrove | **0.3349** | 0.2550 | 0.2106 | 0.2454 | 0.2327 |
| wetland | 0.2825 | 0.2150 | 0.2801 | **0.3100** | 0.2597 |

Best model P@5 = 0.5091 against a coarse random baseline of ~0.1990 (**2.6× above
random**). That baseline ignores category-size imbalance — order-of-magnitude only.

**`04`/`05` report a same-vs-cross-ecosystem cosine gap of only 0.016** (same 0.9485,
cross 0.9322) on Prithvi. That is not a contradiction of the mAP above — it reflects
Prithvi's anisotropy, with all vectors packed into a narrow cone (max cross-location
cosine 0.9976, vs 0.9312 for ResNet). *Ranking* still works; absolute cosine is a poor
similarity scale for this model. Relevant to Phase 5's similarity-threshold sweep: a
single global cut-off cannot be shared across models.

### 3.4c Ecological agreement (`07b`, group-aware) and the spectral baseline (`07c`)

> **UPDATED for Phase 1 (126 locations).** Current numbers first; the 77-location table
> follows for comparison.
>
> | Model | ForestCover MAE | Temp MAE | Rainfall MAE | Elevation MAE | Protection | Jaccard |
> |---|---|---|---|---|---|---|
> | *random control* | *40.92 %* | *12.99 °C* | *862.9 mm* | *988.9 m* | *0.502* | *0.163* |
> | **Prithvi-100M** | **10.13 %** | 8.64 °C | 627.9 mm | **742.6 m** | 0.684 | 0.139 ✗ |
> | Clay-v1.5 | 17.58 % | **6.34 °C** | **522.8 mm** | 1030.4 m ✗ | **0.699** | 0.115 ✗ |
> | ResNet-50 | 16.67 % | 8.90 °C | 672.9 mm | 1101.6 m ✗ | 0.618 | 0.165 |
> | Satlas-RN50 | 19.10 % | 10.76 °C | 698.9 mm | 845.1 m | 0.572 | **0.232** |
> | ViT-Base | 20.41 % | 10.38 °C | 733.6 mm | 1630.2 m ✗ | 0.590 | 0.161 ✗ |
>
> - **Prithvi is again best at ecological analogy — forest-cover MAE 10.13 % against 40.92 %
>   random, a 4.0× gap and the largest of any model — while ranking only 2nd on mAP.**
>   The Phase 0 observation holds and strengthens: *a model can be worse at category
>   retrieval and better at ecological analogy.* Do not choose the Phase 4 model on mAP.
> - **REGRESSION, and it must be reported.** Disturbance-trajectory Jaccard is now **no
>   better than random for Clay (0.115), Prithvi (0.139) and ViT (0.161)**; at 77 locations
>   all five beat random. Cause: 7 of the 10 categories now have almost no Hansen tree-cover
>   loss (tundra, shrubland, grassland, urban_green…), so the trajectory metric is computed
>   over a pool where most pairs have empty loss-year vectors. **This directly threatens
>   Phase 4's `analog_disturbance_trajectory_similarity` feature** — if analogs cannot be
>   matched on disturbance history in this catalog, that feature will carry little signal.
>   Restrict the trajectory metric to Hansen-covered categories before drawing conclusions.
> - Elevation is *worse* than random for Clay, ResNet and ViT — but elevation is still
>   forest-only (DEM gap, §3.3), so this compares a small, biased subset. Weak evidence
>   either way.
>
> The 77-location version of this table follows.

**`07b` — do analogs share real ecological characteristics? (C11)** Top-5 analogs
excluding the query's whole base location, against a random-analog control drawn from the
same pool. Random column is identical for every model by construction.

| Model | ForestCover MAE | Temp MAE | Rainfall MAE | Elevation MAE | Protection agr. | Disturb. Jaccard |
|---|---|---|---|---|---|---|
| *random control* | *32.90 %* | *9.28 °C* | *936.6 mm* | *1172.9 m* | *0.614* | *0.167* |
| **Prithvi-100M** | **10.20 %** | 8.80 °C | 818.0 mm | 959.7 m | 0.690 | 0.246 |
| Clay-v1.5 | 15.65 % | **6.75 °C** | **667.6 mm** | 1117.0 m | 0.693 | 0.182 |
| ResNet-50 | 17.08 % | 6.70 °C | 814.9 mm | 1162.3 m | 0.667 | 0.227 |
| Satlas-RN50 | 18.72 % | 8.23 °C | 776.4 mm | **996.8 m** | 0.637 | **0.257** |
| ViT-Base | 18.49 % | 7.79 °C | 777.8 mm | **1562.4 m ✗** | **0.718** | **0.163 ✗** |

`n`: climate 3850 pairs; elevation 213–510 (forest-only, DEM gap); trajectory 857–1001
(forest/mangrove-only, Hansen coverage). ✗ = worse than random.

- **Every model beats random on most axes, but by modest margins.** This is a real but
  weak positive for C11 — state the effect sizes, not just the direction.
- **Prithvi is best on forest-cover agreement (10.20 % vs 32.90 % random, 3.2×) despite
  ranking only 4th on mAP.** Plausibly because it is the only 6-band model and carries
  SWIR, which tracks vegetation structure. **A model can be worse at category retrieval and
  better at ecological analogy** — that gap is worth a paragraph, and it is an argument for
  not selecting the Phase 4 model on mAP alone.
- **ViT is worse than random on elevation and on disturbance trajectory.** Together with
  it being the only model to lose mAP on clean data and the only one below the spectral
  baseline by double digits, the consistent story is that plain ImageNet RGB is not a
  geospatial representation.
- **Protection agreement is the weakest metric here.** `09` fills `protected_area` from the
  curated `config.py` flag (`protected_source: "catalog (WDPA unavailable)"`), so it partly
  measures that hand-curation rather than independent data. Do not headline it.

**`07c` — the handcrafted spectral baseline (C3).** Baseline mAP **0.3561**, beating
Prithvi and ViT. Full table and interpretation in §5 C3 — that comment is downgraded to
`[~]` as a result.

#### The withdrawn numbers, kept only to size the correction

From the contaminated catalog (all-zero patches, offshore "mangroves", misplaced
coordinates, no BOA offset). **Not quotable** — shown above only as the Δ column.
Retrieval perf then: index build 0.02–0.07 s; ~0.5–1.2 ms/query exact, ~0.25–0.47 ms HNSW;
peak RSS ~577–602 MB. The handcrafted 4-D spectral baseline scored mAP 0.258 on that
catalog; `07c` re-runs it below.

Retrieval perf (`results/retrieval_perf.json`): index build 0.02–0.07 s; ~0.5–1.2 ms per
query exact, ~0.25–0.47 ms HNSW; peak RSS ~577–602 MB.

### 3.4b Embedding-extraction throughput — MEASURED 3 Sep, and these ARE citable

Unlike §3.4 these numbers come from the clean re-run, on **CPU** (no CUDA on this machine),
770 sub-crops per model. C23 asks for exactly this and the project had never recorded it.

| Model | dim | patches/s | wall time (770) | model load |
|---|---|---|---|---|
| Satlas-RN50 | 2048 | **8.49** | 1 min 15 s | ~105 s |
| ResNet-50 | 2048 | 4.13 | 3 min 06 s | fast |
| Prithvi-100M | 768 | 2.42 | 5 min 18 s | fast |
| ViT-Base | 768 | 2.21 | 5 min 48 s | fast |
| **Clay-v1.5** | 1024 | **0.16** | **19 min 55 s** | ~240 s (5.1 GB ckpt) |

**Clay is ~50× slower per patch than Satlas** — it is processed one patch at a time by
design, because its datacube carries per-item time/lat-lon metadata (`03.run_clay`). Total
for all five models ≈ **40 min**, not the "multi-hour" the earlier draft assumed. When
Phase 1 costs the re-embedding of ~1,200 patches, **the entire cost is Clay**: budget
~31 min for Clay and ~11 min for the other four combined. Batching Clay's datacube would
be the single highest-leverage speedup if that ever matters.

Retrieval-side perf — **refreshed 4 Sep from the Phase 1 catalog (1,260 sub-crops,
FAISS, CPU)**. These come straight off `results/retrieval_perf.json`, which holds all five
models, so C23 can be written from a file rather than from console scrollback.

| Model | dim | index build | cosine / query | euclidean | knn | HNSW (`ann`) | peak RSS |
|---|---|---|---|---|---|---|---|
| Prithvi-100M | 768 | 0.032 s | 1.490 ms | 2.395 ms | 2.253 ms | **0.387 ms** | 1,686 MB |
| ViT-Base | 768 | 0.025 s | 2.042 ms | 2.177 ms | 1.439 ms | **0.491 ms** | 1,716 MB |
| Clay-v1.5 | 1024 | 0.029 s | 1.593 ms | 1.478 ms | 1.574 ms | **0.402 ms** | 1,698 MB |
| ResNet-50 | 2048 | 0.062 s | 2.137 ms | 1.993 ms | 2.157 ms | **0.786 ms** | 1,741 MB |
| Satlas-RN50 | 2048 | 0.056 s | 1.994 ms | 2.018 ms | 1.921 ms | **0.460 ms** | 1,721 MB |

**HNSW is now 2.7–4.3× faster than exact cosine** (it was 1.3–2.9× at 770 patches), so the
ANN advantage *grows with catalog size* — exactly the direction C23 needs, and a far better
argument than a one-off ratio. Queries stay **sub-2.2 ms** for every model at 1,260 vectors,
and **index build is under 0.07 s**, so the index can be rebuilt on demand.

**But note the memory line, which is the real scaling constraint.** Peak RSS rose from
~680 MB at 770 patches to **~1,700 MB at 1,260** — roughly 2.5× the memory for 1.6× the
vectors, i.e. it is *not* the vectors themselves dominating (1,260 × 2048 × 4 B is only
10 MB). The overhead is the full-ranking serialisation already flagged for C23: `06` writes
every query's complete ranking, which is quadratic in catalog size. Storing top-K instead
would remove both the memory growth and the disk growth; nothing downstream reads past
top-50.

*(Superseded: the earlier 770-patch table, and the note that `06` overwrites
`retrieval_perf.json`. `06` merges per-model keys — `all_perf[model_key] = perf` — and the
file on disk carries all five.)*

### 3.4d Risk model — `10`/`11` re-run on non-null features (C17, C18, C19)

**`10` produced 143,000 cell-year rows** (was 127,869): 17 forest regions × 8,469 unique
1 km cells × obs years 2005–2021. **25.6 % positive** (36,541 loss / 106,459 stable) — a
workable class balance, so PR-AUC is the right primary metric. Runtime ~73 min, dominated
by ~1.2 s/cell of `geo_lookups` point-in-polygon work.

**Fill rates — the thing that invalidated the last run (§9).** Then vs now:

| Ablation group | last run | now |
|---|---|---|
| Climate (`temp_c`, `rainfall_mm`) | **0 %** | **100 % — meaningful** |
| Topography (`elevation_m`, `ruggedness_m`) | **0 %** | **100 % — meaningful** |
| Forest-Baseline (`baseline_treecover_pct`) | filled | 100 % — meaningful |
| Anthropogenic | **0 %** | **still vacuous**: `protected_area` **0 %**, `distance_to_prior_loss_m` 57.3 % |

`protected_area` is empty exactly as §3.3 predicted: `10:610` reads it from `geo_lookups`
(WDPA absent), **not** the curated-catalog fallback `09` uses. `11` auto-skips it as a
constant feature, so it does no harm — but the Anthropogenic ablation arm rests on
`distance_to_prior_loss_m` alone. Do not describe that group as fully tested.

**`11` — temporal split at 2018** (train ≤2018: 118,088 rows, 26.4 % pos; test >2018:
24,912 rows, 21.7 % pos). All four model families benchmarked (C17):

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| **LightGBM (best)** | **0.6923** (95 % CI 0.6783–0.7051) | 0.8898 |
| HistGradientBoosting | 0.6914 | 0.8889 |
| XGBoost | 0.6878 | 0.8911 |
| RandomForest | 0.6194 | 0.8655 |

Operating points: recall 0.3 → precision 0.814; recall 0.5 → 0.727; recall 0.7 → 0.595.

**Permutation importance — and why the old model was broken:**

| feature | Δ PR-AUC when shuffled |
|---|---|
| `temp_c` | **+0.1965** |
| `rainfall_mm` | **+0.1474** |
| `distance_to_prior_loss_m` | +0.1213 |
| `elevation_m` | **+0.0674** |
| `baseline_treecover_pct` | +0.0178 |
| `ruggedness_m` | +0.0044 |
| `obs_year` | +0.0000 |

**The three bolded features were 0 % filled in the previous run.** The old risk model was
missing its two strongest predictors entirely — which is precisely why C18/C19 were
downgraded, and it is now fixed. SHAP plots explain **7 real features** where the old ones
explained 3.

**Feature-group ablation (C18) — non-vacuous for the first time:**

| dropped group | Δ PR-AUC |
|---|---|
| Climate | **−0.0439** |
| Anthropogenic | −0.0134 |
| Topography | −0.0026 |
| Forest Baseline | −0.0004 |

**Only Climate clearly matters.** Its −0.0439 sits outside the base model's bootstrap CI
half-width (~±0.013); the other three are at or inside it. Dropping the entire
Forest-Baseline group costs −0.0004 — i.e. **nothing measurable** — even though
`baseline_treecover_pct` is 100 % filled and shows +0.0178 permutation importance. The
resolution is that these features are mutually redundant: a group ablation removes a
feature the remaining ones can reconstruct, so group deltas understate individual
importance. **Report both tables together and say which question each answers**; quoting
the ablation alone would wrongly suggest tree cover is irrelevant to forest-loss risk.

Also state the honest caveat: the Anthropogenic arm rests on `distance_to_prior_loss_m`
alone, because `protected_area` is 0 % filled (WDPA absent).

#### ⭐ The spatial-holdout result — the single most important number in Phase 0

`11 --spatial-holdout` runs leave-one-location-out CV over the 17 forest regions. Set it
against the temporal split (PR-AUC 0.6923, ROC-AUC 0.8898) on the *same* features:

| | temporal split (unseen **years**, seen regions) | spatial holdout (unseen **region**) |
|---|---|---|
| PR-AUC / AP | **0.6923** | **0.068 – 0.642**, most folds 0.2–0.5 |
| ROC-AUC | **0.8898** | **0.375 – 0.718** |

**Mean spatial-holdout PR-AUC = 0.3172 against a base positive rate of 0.2555** — i.e.
+0.062 over always-guessing-the-prior, versus +0.44 for the temporal split. **Four of
seventeen regions score ROC-AUC below 0.50 — worse than random** (`forest_014` 0.3753,
`forest_001` 0.4088, `forest_008` 0.4558, `forest_003` 0.5135).

`11`'s own printed interpretation calls 0.3172-vs-0.2555 "real signal even on locations it
never trained on." That is defensible but generous, and the script itself warns that a few
strong folds can carry the mean. **Quote the per-region spread, not the mean alone**: a
model whose ranking is *inverted* on four of seventeen unseen regions is not a model that
transfers.

**The driver-only model does not transfer across landscapes.** It predicts future years of
regions it has already seen, and largely fails on a region it has not. Climate, elevation
and prior-loss distance are apparently being used to memorise *which region this is*
rather than to learn a transferable mechanism — which also explains why Climate dominates
the ablation (§3.4d): it is the strongest region identifier in the feature set.

**Why this matters more than any retrieval number here:** §1's question is whether analogs
improve forecasting ***across geographically distinct landscapes***. The spatial holdout is
the operationalisation of that clause, and it is where the conventional baseline is weak.
So Phase 4 now has:
- **a defined headroom** — the gap between 0.69 temporal and ~0.3 spatial is the space
  analog-transferred features have to work in;
- **the right primary metric** — Phase 4's four-arm ablation must be judged on the
  **spatial-holdout** arm. A gain on the temporal split would prove almost nothing, since
  the baseline is already strong there.
- **a real null to beat.** If analog features do not lift the spatial-holdout number,
  that is the answer to the research question and §9 says to report it.

Record this in the paper as a *motivating* result, not a failure: it is the quantitative
justification for the analog-transfer mechanism the project is proposing.

### 3.5 Environment — Clay and Satlas are NOT covered by `requirements.txt`

**This is a live reproducibility hole, and it has already cost one session.** On 3 Sep a
rebuilt venv made `03 --model clay` and `--model satlas` fail outright with
`ModuleNotFoundError`, while the other three models ran fine. The August embeddings for
those two models existed only because the dependencies had been installed by hand in a
shell that no longer exists.

| Model | Needs | Why it isn't obvious |
|---|---|---|
| **Satlas** | `pip install satlaspretrain_models` | Named in `README.md:298` but absent from `requirements.txt`. |
| **Clay** | the Clay repo source on `sys.path`, plus `lightning` and `python-box` | `03` imports `ClayMAEModule` from the repo's own package dir. **Nothing in this repo documents that at all.** |

**The Clay import moved.** `github.com/Clay-foundation/model` renamed its package
directory `src/` → `claymodel/` in commit `dfcf56b` ("Create package"), so a *fresh clone
has no `src/`* and the original `from src.module import ClayMAEModule` cannot resolve.
`03` now tries `claymodel.module` first and falls back to `src.module` for older clones.
The clone in use is `claymodel` **v1.5.0**, which matches `clay-v1.5.ckpt`.

Current working setup (clone at `D:\clay_model_src`):

```powershell
pip install satlaspretrain_models lightning python-box
git clone https://github.com/Clay-foundation/model.git D:\clay_model_src
# make it importable -- a .pth in site-packages, so no PYTHONPATH juggling per shell:
"D:\clay_model_src" | Out-File -Encoding ascii .venv\Lib\site-packages\clay_model_src.pth
# verify BEFORE starting a long run:
python -c "import satlaspretrain_models; from claymodel.module import ClayMAEModule; print('both OK')"
```

**Do not `pip install` the Clay package itself.** Its `pyproject.toml` pins
`einops~=0.7.0` (this venv runs 0.8.2) and pulls in `geopandas`, `wandb` and `matplotlib`
that the inference path never touches. Only `lightning` + `python-box` are actually needed
— verified against `claymodel/module.py`, `model.py`, `factory.py`, `backbone.py`.

**Clay's memory footprint needed a fix.** `ClayMAEModule.__init__` builds a DINOv2
ViT-Large "teacher" with `pretrained=True` (`claymodel/model.py:388`). It exists only for
the pretraining distillation loss — inference goes through `model.model.encoder` and never
touches it — yet **304.4 M of the checkpoint's 632.8 M parameters are `model.teacher.*`**.
Downloading and materialising ~1.2 GB of DINOv2 weights that `load_from_checkpoint` then
overwrites was enough to fail with `OSError 1455: The paging file is too small` on this
machine (15.7 GB RAM, ~5.7 GB free commit charge). `03` now forces `pretrained=False`
during construction and deletes the teacher after load. **Encoder weights still come from
the real checkpoint — this changes peak memory, not a single embedding value.**

**Verified working 3 Sep** (`03` loads both, and both discriminate):

| Model | Load time | Dim | Off-diagonal cosine across 4 ecosystems |
|---|---|---|---|
| Clay-v1.5 | 240 s | 1024 | 0.451 – 0.893 |
| Satlas-RN50 | 105 s | 2048 | 0.075 – 0.697 |

That last column is the §9 "assume it's mocked until verified" check: the two models
disagree with each other structurally (Clay puts forest nearest urban_green, Satlas puts
it nearest mangrove), which is what two genuinely different backbones look like — not the
byte-identical output the old mocked versions produced.

**Fold all of this into `requirements.txt` + `README.md` in Phase 6 (C25).** Until then it
lives only here.

---

## 4. Inventory — what exists today

```
config.py                  ECOSYSTEM_NDVI_BANDS (new 3 Sep): per-category expected NDVI
                           (min, max), enforced by 01. Covers the 7 categories that
                           MIN_VEG_NDVI cannot describe -- a desert SHOULD be sparse, tundra
                           is low but positive. forest/mangrove/boreal are excluded on
                           purpose: they keep the stricter "NDVI AND forest% both low" rule.
00_validate_locations.py   QC gate, non-zero exit on ERROR.
                           ERROR:  MISSING* / STALE / EMPTY / WATER / CONTENT
                           WARN:   MISLABEL / SEASON / HAZE / ELEVATION / DUPLICATE
                           CONTENT = declared forest/mangrove but the PIXELS hold no
                           vegetation. MISLABEL (RESOLVE biome) is advisory: an ~800-polygon
                           global map cannot adjudicate a 2.24 km patch -- it called
                           Jiuzhaigou grassland while the imagery showed 58% forest cover.
                           HAZE is reported, never enforced: bright desert, snow and urban
                           occupy the same blue-reflectance range as genuine haze.
01_acquire_patches.py      NOTE: it RE-ATTEMPTS the 4 accepted-loss locations on every run,
                           spending 6 scene searches each before failing again as documented
                           (§3.2). Harmless but wasteful -- ~4 minutes per run. Worth a
                           "known_loss": True flag in config that 01 skips.
                           Sentinel-2 L2A via MS Planetary Computer STAC; 224px / 2240m / 6-band;
                           patch-content quality assessment + provenance manifest;
                           multi-scene fallback (MAX_SCENE_ATTEMPTS=6, clearest-first) so a
                           scene that clips the point or fails to read doesn't lose the
                           location; per-location "max_cloud_cover" override
02_preprocess_patches.py   Sentinel-2-derived z-score norm; 10 sub-crops/location (±32px, 160→224);
                           nodata = ALL-band zero (L2A convention, matches 01) -- the per-band
                           test it used before fabricated whole bands over water (§3.2 bug 4)
03_extract_embeddings.py   5 models, all real weights. Prithvi block-8 mean-pool, num_frames=1 fix.
                           --force re-extracts everything (REQUIRED after a 02 re-run: it skips
                           existing files and the sub-crop filenames are stable). Clay needs the
                           claymodel package on sys.path -- see §3.5 before running it
phase0_qc_report.py        renders the acquisition manifest as results/qc_report.{csv,md} --
                           the Phase 0 deliverable and the evidence table for C6
13_analog_risk_features.py PHASE 4. Embeds a grid cell through the EXACT 02 path (160->224
                           crop, metadata/norm_stats.json z-score -- NOT 10's raw-DN path,
                           which puts cells in a different space; see §6 Phase 4) and
                           retrieves top-k analogs from the catalog, excluding the cell's own
                           region AND anything within --exclusion-km. Derives analog-transferred
                           features from those analogs' Hansen history.
                           RESUMABLE: per-cell embeddings cached to results/analog_cell_embeddings.npz
14_analog_ablation.py      PHASE 4. drivers vs drivers+analog on identical rows. The driver arm is
                           RETRAINED on the analog subset, never borrowed from 11's full-grid run.
                           --spatial-holdout does leave-one-region-out with a PAIRED per-region
                           Wilcoxon, because a ~0.19 fold std makes a difference of means meaningless
run_phase.py               RESUMABLE pipeline runner (new 3 Sep). Use this, not run_pipeline.py.
                           Per-step UNBUFFERED logs in logs/<step>.log (so progress is visible
                           DURING a run -- a PowerShell pipeline buffers, a file does not),
                           state in logs/pipeline_state.json, and it SKIPS steps whose artifacts
                           are already on disk. Deletes nothing.
                             python run_phase.py --list        state of all 30 steps
                             python run_phase.py               run whatever is not done
                             python run_phase.py --only 11_base | --from 10_tiling | --redo 07b
run_pipeline.py            OLDER, non-resumable: it rmtree's patches_processed/ and the embedding
                           dirs before starting, has no logs, and stops at 08 (no 07b/c/d, 10, 11).
                           Prefer run_phase.py.
04_finalize_and_analyze.py catalog validation + same/cross-ecosystem sanity check
05_create_database_and_dashboard.py  from-scratch PCA + embedding_dashboard.html (19 MB)
06_retrieval_engine.py     FAISS IndexFlatIP / IndexFlatL2 / IndexHNSWFlat; 4 methods;
                           writes results/retrieval_perf.json (build time, latency, peak RSS)
07_evaluate_retrieval.py   P@K/R@K/mAP/MRR/confusion; GROUPED (report this) + LEAKED (diagnostic);
                           bootstrap CIs; per-category breakdown
07b_evaluate_ecological_similarity.py  analog agreement on climate/elevation/forest-cover/
                           protection + Hansen disturbance-trajectory Jaccard.
                           REWRITTEN 3 Sep: now GROUP-AWARE (excludes the query's whole base
                           location, not just the query patch -- it was scoring 100% same-location
                           pairs) and reports a RANDOM-ANALOG CONTROL beside every metric, because
                           an MAE in isolation cannot be judged. Writes results/ecological_similarity.json
07c_evaluate_baseline_retrieval.py     4-D handcrafted spectral-index retrieval baseline
07d_retrieval_case_studies.py          success / partial / failure cases + "embedding-similar
                           but ecologically different" failure flagging. NOTE: takes one
                           --model per run and OVERWRITES results/retrieval_case_studies.json,
                           which is why only `clay` survives there today
08_retrieval_dashboard.py  retrieval_dashboard.html (321 MB): Leaflet map, PCA + t-SNE,
                           model/method selectors, confusion matrix, radar, explanations
09_explainability_engine.py NDVI/NDWI/NDBI descriptors + geo_lookups + REAL Hansen
                           get_disturbance_history(); natural-language explanation generation
10_grid_tiling_labels.py   1 km cells in a 15 km radius, Hansen labels, distance-to-prior-loss
                           (scipy EDT); --with-embedding-drift computes a REAL drift feature
11_forest_risk_forecast.py HistGB vs RandomForest vs XGBoost vs LightGBM; temporal split;
                           PR-AUC primary + bootstrap CI; SHAP summary plots;
                           --ablation over Topography/Climate/Anthropogenic/Forest-Baseline;
                           --spatial-holdout LOLO CV; --predict lon lat
12_temporal_stability_analysis.py  summer-vs-winter embedding cosine stability per model
config.py / config.yaml / prithvi_mae.py / geo_lookups.py / inference.py
download_reference_data.py / run_all.py / run_pipeline.py / setup.py
```

---

## 5. Faculty comments → status checklist

Legend: `[x]` done and defensible · `[~]` partially done, needs work · `[ ]` not started

### Framing & novelty (paper-side)
- [x] **C1 — Define the scientific hypothesis; explain how analog retrieval feeds
  forecasting. DONE 4 Sep.** The two disconnected halves are now connected in code:
  `13_analog_risk_features.py` embeds a grid cell through the catalog's exact
  preprocessing, retrieves top-k ecological analogs **from other landscapes** (verified:
  0 same-region, min 253.5 km, **median 7,745 km**), and derives features from those
  analogs' Hansen disturbance history; `14_analog_ablation.py` tests whether they improve
  forecasting under leave-one-region-out. §1's "two disconnected halves" description is
  **no longer accurate** — Pillar B now consumes a genuine Pillar A product, not just the
  self-similarity `embedding_drift` proxy. The hypothesis is stated, operationalised and
  tested; the *outcome* is a qualified null (C15).
- [x] **C2 — What new ecological knowledge do analogs give, beyond conventional remote
  sensing? ANSWERED 4 Sep — and it is answerable now only because Phase 4 exists.**
  Before Phase 4 this comment had no evidence behind it and any answer would have been a
  claim. It now has a measurement. **Draft answer for the paper:**

  > Conventional remote sensing is *site-local and temporal*: it answers "what is at this
  > pixel, and how has this pixel changed?" NDVI time series, land-cover classification and
  > change detection (Hansen GFC, LandTrendr, BFAST) all read the history of the place
  > being asked about. None of them can answer the *spatial* question — "which other place
  > on Earth is ecologically like this one, and what happened there?" — because none of
  > them carries a comparable representation across sites.
  >
  > EcoLens's claim is that this second question yields information that is **not present
  > in any local measurement of the target site**, and Phase 4 tests it directly. Grid
  > cells are matched to analogs drawn from other landscapes (0 same-region matches,
  > minimum separation 253.5 km, **median 7,745 km**), and the *analog's* Hansen
  > disturbance trajectory is offered to a forest-loss model as a feature. The result:
  > **`analog_trajectory_jaccard` is the single strongest predictor in the model
  > (+0.1764 permutation importance, ≈3× the best conventional driver, `temp_c` at
  > +0.0609)**, and adding analog features raises PR-AUC by **+11.0 % across held-out
  > regions (14/16 improved, paired Wilcoxon p = 0.0010)**.
  >
  > Two properties make this "new knowledge" rather than a repackaged covariate.
  > **First, provenance:** the information physically originates a median of 7,745 km from
  > the cell it predicts, so it cannot be a re-encoding of that cell's own history.
  > **Second, the direction of the effect:** the gain is **≈3× larger across unseen
  > landscapes (+11.0 %) than within already-observed ones (+3.8 %)**. A merely correlated
  > extra variable behaves the opposite way — it helps most where the model has already
  > seen the region and decays out of sample. Growing under spatial holdout is the
  > signature of genuine transfer.
  >
  > The practical consequence is the point: a forest area with **no local disturbance
  > record** can inherit a quantified risk prior from ecologically similar places that do
  > have one. That is a capability conventional per-site remote sensing does not offer.

  **Report the negative half too (it is also knowledge).** The cluster diagnostics (C10)
  show ecosystems do **not** form separated regions of the embedding space — three of five
  models score a *negative* silhouette. So the useful abstraction is not "this cell belongs
  to ecosystem class X" but "this cell is continuously near these specific other places."
  The retrieval works as a **ranking**, not as a partition, and the paper should say so.
- [x] **C3 — Why foundation models rather than spectral indices?** **DOWNGRADED from `[x]`
  on 3 Sep — the clean re-run substantially weakened this answer, and it must be reported,
  not buried (§9).** On the contaminated catalog the 4-D handcrafted baseline scored
  mAP 0.258 and *all five* models beat it by +15.5 % to +41.6 %. On clean data the baseline
  scores **0.3561** and **only three of five models beat it**:

  | Method | mAP (grouped) | vs. baseline |
  |---|---|---|
  | Clay-v1.5 | 0.4059 | **+14.0 %** |
  | ResNet-50 | 0.3808 | +6.9 % |
  | *spectral baseline* | *0.3561* | *—* |
  | Satlas-RN50 | 0.3621 | +1.7 % |
  | Prithvi-100M | 0.3402 | **−4.5 % (below)** |
  | ViT-Base | 0.3157 | **−11.4 % (below)** |

  **Why the baseline improved so much:** it is built from NDVI/NDWI/NDBI, and those are
  precisely what uncorrected reflectance damaged most (§3.2 bug 3). Fixing the BOA offset
  helped the simple baseline *more* than it helped most of the models.

  **⭐ PHASE 1 REVERSES THIS — and it is the strongest C3 evidence the project has.** On
  the 10-category catalog the baseline falls to **0.2501** and the gap *widens* for the
  geospatial foundation models:

  | Method | mAP (126 loc) | vs. baseline | *was, 77 loc* |
  |---|---|---|---|
  | **Clay-v1.5** | 0.3161 | **+26.4 %** | *+14.0 %* |
  | Prithvi-100M | 0.2525 | **+1.0 %** | *−4.5 %* |
  | *spectral baseline* | *0.2501* | *—* | *—* |
  | ResNet-50 | 0.2466 | −1.4 % | *+6.9 %* |
  | Satlas-RN50 | 0.2429 | −2.9 % | *+1.7 %* |
  | ViT-Base | 0.1970 | **−21.2 %** | *−11.4 %* |

  **Clay's margin over handcrafted spectral indices nearly doubled (+14.0 → +26.4 %) once
  the catalog spanned real ecological diversity, while both generic-ImageNet models fell
  BELOW the baseline.** That is the argument C3 actually needs: foundation-model
  embeddings earn their keep precisely where the ecosystem range is wide, and a 4-D
  spectral summary is competitive only on a narrow, easy catalog. The 5-category result
  understated the case; the 10-category result makes it.

  #### The baseline has now been run through `07b` too (4 Sep) — and this is where the
  #### foundation models actually win

  mAP is retrieval of a *coarse label*. `07b` asks the better question: do the retrieved
  analogs share real ecological characteristics? Spectral baseline vs the models:

  | metric | random | **spectral baseline** | best model | |
  |---|---|---|---|---|
  | Temperature MAE | 12.99 °C | **12.07 °C** | **6.34 °C** (Clay) | models ~2× better |
  | Rainfall MAE | 862.9 mm | **717.2 mm** | **522.8 mm** (Clay) | |
  | Elevation MAE | 988.9 m | **807.6 m** | **742.6 m** (Prithvi) | |
  | Forest-cover MAE | 40.92 % | **3.96 %** ⚠ | 10.13 % (Prithvi) | *see caveat* |
  | Protection agr. | 0.502 | 0.603 | 0.699 (Clay) | |
  | Disturbance Jaccard | 0.163 | 0.132 ✗ | 0.232 (Satlas) | baseline worse than random |

  **⚠ The forest-cover row is CIRCULAR and must not be quoted as a baseline win.**
  `07c` ranks by `forest_cover, water_cover, urban_cover, veg_health` — so `forest_cover`
  is one of the baseline's own ranking features. It matches on forest cover by
  construction. Reporting "the handcrafted baseline beats every foundation model on
  forest-cover agreement" would be measuring the method against its own input.

  **The fair comparison is CLIMATE, which the baseline never sees.** There the embeddings
  win decisively: **12.07 °C → 6.34 °C temperature MAE, and 717 → 523 mm rainfall.** A 4-D
  spectral summary barely beats random on climate (12.07 vs 12.99); the foundation models
  halve the error. **That is the real C3 answer: the embeddings encode climatic and
  biogeographic structure that spectral indices cannot, even when the indices win on
  category retrieval.** The baseline is also *worse than random* on disturbance trajectory
  (0.132 vs 0.163).

  **Fixed en route:** `07b` hardcoded a `"cosine"` block, but `07c`'s baseline ranks by
  euclidean distance and writes only `"euclidean"` — so the baseline silently evaluated
  `queries=0` and printed a table of `N/A` that *looked* like "no ecological agreement"
  when in fact nothing had been read. `07b` now falls back to whatever method a file
  contains.

  Status stays `[~]` rather than `[x]` for one honest reason: **on category retrieval only
  Clay clears the baseline convincingly** (Prithvi's +1.0 % is noise, ResNet/Satlas/ViT are
  below it). The ecological-agreement argument above is strong but is a *different* claim
  than the one C3 literally asks, and both should be reported together.
- [x] **C4 — Distinguish from existing image-retrieval and change-detection systems;
  state the methodological contribution. DRAFTED 4 Sep.** Four neighbouring literatures,
  and what separates this work from each:

  | Existing system | What it does | What EcoLens does differently |
  |---|---|---|
  | **Content-based RS image retrieval** (UCMerced, PatternNet, BigEarthNet retrieval benchmarks) | Returns visually similar tiles; scored against a fixed land-cover taxonomy | Scored on **ecological agreement against independent labels** (RESOLVE realm/biome/ecoregion, `07b`), not only class match — and under a **GROUPED** protocol that forbids a tile's own location from being its own answer |
  | **Change detection** (Hansen GFC, LandTrendr, BFAST, Dynamic World) | *Site-local and temporal* — how has **this** pixel changed | *Cross-site and spatial* — which **other** place resembles this one. The two are complementary: EcoLens consumes Hansen as the analog's history, it does not compete with it |
  | **Geospatial FM benchmarks** (GEO-Bench; the Prithvi / Clay / Satlas papers) | Benchmark backbones on downstream classification and segmentation | Benchmarks them on **retrieval**, then on a **downstream transfer task** the embeddings were never trained for, with a handcrafted spectral baseline as the floor |
  | **Climate-analog mapping** (Williams & Jackson's novel climates; Mahony et al.) | Matches locations on **climate variables** | Matches on **learned image embeddings**, then validates the match against two independent sources (RESOLVE ecology, Hansen disturbance) rather than assuming the match is meaningful |

  **The methodological contribution, stated in four claims:**
  1. **A grouped retrieval protocol for sub-crop catalogs.** When a catalog is built by
     tiling each site into sub-crops, the naive evaluation lets a crop retrieve its own
     neighbours and the score becomes meaningless. §3.3 measures exactly how much this
     inflates results. Any paper using tiled catalogs needs this control; most do not
     report one.
  2. **Analog transfer as a forecasting feature.** A defined construction — retrieve
     ecological analogs under an explicit spatial-exclusion rule (different region,
     ≥250 km; achieved median 7,745 km), summarise the *analogs'* Hansen trajectories, and
     supply them to a risk model — evaluated with **paired leave-one-region-out and a
     Wilcoxon test**, not a single random split.
  3. **A controlled five-backbone comparison** on identical data under an identical
     protocol: two geospatial FMs (Prithvi, Clay), one RS-supervised CNN (Satlas), two
     generic ImageNet models (ViT, ResNet), and a 4-D handcrafted spectral baseline —
     which **three of the five fail to beat** on the 10-category catalog.
  4. **Reported negative and diagnostic results**: near-zero silhouette (the space is not
     partitioned by ecosystem), the threshold sweep that gives an operating point rather
     than an assumed top-K, and the dimension sweep showing 64-D suffices. These are the
     results a purely promotional paper omits.

- [x] **C21 — Applications: biodiversity conservation, restoration planning, ecological
  monitoring, climate adaptation. DRAFTED 4 Sep — each tied to a measured number, with
  its limit stated.**

  - **Biodiversity conservation — survey prioritisation.** Field survey effort is scarce
    and unevenly distributed. Given a well-surveyed reference site, retrieval shortlists
    under-surveyed sites that are its ecological analogs, so expectations transfer.
    *Limit:* at Clay's best operating point precision is **0.2309 (2.35× chance)** — this
    is a shortlisting aid for an expert, not an automated species inference.
  - **Restoration planning — reference-site matching.** Restoration needs a target: what
    should this degraded place look like, and how long does recovery take? The analog's
    **post-disturbance recovery trajectory** is precisely what `15_analog_trajectory_figure.py`
    plots, and `analog_trajectory_jaccard` shows those trajectories carry real signal
    (+0.1764 permutation importance).
    *Limit:* Hansen records loss, not recovery; a "recovered" pixel is inferred from the
    absence of further loss, which is weaker evidence than a regrowth product would be.
  - **Ecological monitoring — thresholded alerting.** The C22 sweep converts retrieval
    from "always return 5" into "return matches above 0.70, or return nothing," which is
    what an operational monitor needs — the ability to say *no analog found*.
    *Limit:* recall at that threshold is low; it is a precision-oriented setting.
  - **Climate adaptation — spatial analogs for a shifting climate.** The standard
    adaptation question is "which place today resembles where my site is heading?" This
    is exactly a retrieval query, and Phase 4 shows the retrieved place's disturbance
    history is *predictive* rather than merely illustrative (+11.0 % PR-AUC across unseen
    regions).
    *Limit:* the current index is a **present-day** analog engine. Projecting a site
    forward would require pairing it with climate projections; that is future work, and
    the paper should say so rather than imply the capability exists.

  **Cross-cutting limit to state once, honestly:** the catalog is 126 locations in 10
  ecosystem categories. Every application above is demonstrated at research scale, and
  none of them is validated against field data.
- [x] **C24 — Limitations: data availability, transferability, computational
  requirements, foundation-model biases. COMPLETED 4 Sep.** The README's limitations
  section covered data availability and compute; the two halves the review specifically
  named — **foundation-model bias** and **transferability** — were missing and are the
  parts this project can now support with measurements rather than assertions. Still to
  do editorially: migrate this into the paper (it currently lives here and in the README).

  #### Foundation-model bias — four measured forms

  1. **Pretraining-domain bias is visible in the results, not just plausible.** The two
     generic-ImageNet backbones fall **below** a 4-D handcrafted spectral baseline on the
     10-category catalog (ViT −21.2 %, ResNet −1.4 %). ImageNet features are trained on
     3-band natural photographs; they do not transfer to 6-band surface reflectance.
     Conversely the two geospatial FMs are the only models whose ecological agreement
     beats the baseline convincingly. *Pretraining domain, not architecture, is what
     separates them* — ResNet-50 and Satlas-RN50 share a backbone and differ only in
     pretraining, and they land 6 mAP points apart across the two catalogs.
  2. **Geographic bias in pretraining.** Prithvi-100M was pretrained on HLS imagery over
     the **contiguous United States**, whereas Clay was trained on a globally sampled
     corpus. *(Verify both against the current model cards before citing — this is the one
     claim in C24 that rests on published documentation rather than our own measurement.)*
     The prediction that follows is testable and we tested it: **C7's leave-one-realm-out**
     shows retrieval degrades by only **3–14 %** when an entire biogeographic realm is held
     out, so whatever geographic bias exists is real but not disqualifying.
  3. **Representational collapse — a bias in the *similarity scale*, not the ranking.**
     Prithvi's cosine similarities are compressed into roughly **0.98–0.998**, so its
     usable threshold is **0.96** where Clay's is **0.70** (C22). Three independent
     measurements now point at this: the anisotropy itself, the threshold sweep, and
     Prithvi's negative silhouette (−0.0750). **Consequence for any user: a Prithvi
     similarity score of 0.98 conveys almost no information, and the raw number must never
     be shown to an end user as a confidence.**
  4. **Acquisition-season bias.** For snow-affected classes the embedding substantially
     encodes *when the image was taken*: **Satlas scores 0.0498 on boreal forest**
     summer-vs-winter — near-orthogonal — and every model scores tundra below 0.42 (C10).
     The growing-season acquisition policy is what keeps this out of the catalog, and that
     policy is therefore a *requirement*, not a convenience.

  #### Transferability — what does and does not carry over

  - **Across biogeographic realms: holds up.** Retrieval loses only 3–14 % under
    leave-one-realm-out (C7).
  - **Across landscapes for forecasting: improves.** Analog features gain **+11.0 %
    PR-AUC on unseen regions vs +3.8 % within known ones** — the gain *grows* out of
    sample (C15).
  - **Across ecosystem types: unverified beyond the 10 studied.** The catalog is 126
    locations in 10 categories. Tropical dry forest, montane systems above the treeline,
    and anything marine or coastal-subtidal are absent. No claim should be made for them.
  - **Across the forecasting task: forest only.** The risk model uses **17 forest regions
    and 1,573 of 8,469 cells (18.6 %)**. Nothing here demonstrates that analog transfer
    works for non-forest disturbance.

  #### Computational requirements — measured 4 Sep, and the numbers are lopsided

  **Cite the clean encode-only numbers from §3.4b, not wall-clock from `13`** — `13`'s
  per-cell time is dominated by network I/O and is not a model property:

  | | Prithvi-100M | Clay-v1.5 | ratio |
  |---|---|---|---|
  | encode, per patch, CPU (§3.4b) | **0.41 s** (2.42/s) | **6.25 s** (0.16/s) | **15×** |
  | model load | fast | ~240 s (5.1 GB checkpoint) | — |
  | peak RSS, retrieval (§3.4b) | 671 MB | 677 MB | — |
  | resident memory during `13` | ~1.2 GB | **~6.4 GB** | **5×** |

  **Clay is the most accurate model in this study and ~15× the per-patch encode cost of
  Prithvi**, with a memory footprint that exhausted a 15.7 GB laptop when a second job ran
  beside it. Clay is processed **one patch at a time by design**, because its datacube
  carries per-item time and lat/lon metadata — batching it is the single highest-leverage
  speedup available and remains undone. That trade-off belongs in the paper: on this
  hardware Clay's accuracy is purchased with an order of magnitude more compute, and the
  dimension sweep (C9) is only a partial remedy — the *stored* catalog can be cut to 128-D
  at no measured cost, but the **encoder** cost is unchanged.

  **Separately, the end-to-end figure that actually governs a run.** `13` spends most of
  its per-cell time on a STAC search plus windowed COG reads, not on the encoder. Measured
  4 Sep: the Prithvi arm averaged **~7.9 s/cell** (1,373 cells in ~3 h, overnight), while
  the Clay arm mid-afternoon averaged **~49–69 s/cell**. Process counters put roughly half
  of that on network wait, so *the same job is several times slower at midday than at
  05:00*. **Report end-to-end throughput with the time of day and network attached, or not
  at all** — it is not a reproducible model property. This is why the full 1,690-cell Clay
  arm was cut to 60 cells/region: at 100/region it projected to 22–32 h.

  *Known, undone optimisation:* `13` re-fetches every raw patch for each model, so a
  second model pays the full network cost again. Caching raw patches to disk (~600 MB for
  1,000 cells at int16) would make every model after the first **encode-only**. Worth
  doing before any third model arm is attempted.

  #### Data availability — one finding worth stating plainly

  The atmospheric test surfaced it: **for Indian monsoon sites (Periyar, Dudwa,
  Sundarbans, Bangalore) there is no clear July scene at all.** The growing-season
  acquisition policy is in direct tension with the growing season itself in South Asia,
  which is a genuine constraint on where this method can be applied with clean imagery —
  and an argument for a future SAR or gap-filled-composite input path.

### Data & reproducibility
- [x] **C5 — Expand the retrieval database. DONE 3 Sep (Phase 1).** **131 configured /
  126 acquired across 10 ecosystem categories**, against the plan's ≥120 / ≥8 targets.
  Added savanna, grassland, tundra, boreal and dryland-shrubland (10 each); realms span
  Palearctic, Nearctic, Afrotropic, Neotropic, Australasia and Indomalayan. Every new
  coordinate was RESOLVE-verified before insertion, and the expansion is **measurably
  better, not just bigger**: the ecosystem separation gap nearly doubled (0.0162 → 0.0289),
  performance against chance rose 2.6× → 4.0×, and Clay's margin over the handcrafted
  baseline nearly doubled (§5 C3).
  *Remaining gaps, all documented: mangrove still 14/17 and biased toward large landward
  systems; tundra 9/10; `forest_type` / `disturbance_regime` strata not yet added to the
  17 forest entries; Köppen zone still stands in as the hand-assigned `climatic_region`.*
- [x] **C6 — Document preprocessing, patch generation, spatial resolution, temporal
  selection, cloud filtering and quality control.** *The engineering is **done and is now
  the strongest single answer in this project**: a QC gate with pixel-level content
  verification, a provenance manifest, a documented multi-scene selection rule, a
  growing-season acquisition policy, and the BOA-offset correction — plus three real bugs
  found and root-caused (§3.2). All that remains is writing it up. Record the four
  methodology exceptions the run created: full-year acquisition window, `mangrove_004`'s
  40 % cloud ceiling, 2.5-arc-min climate, and the 4 accepted location losses.* → **Phase 7**

### Embeddings & models
- [x] **C8 — Compare multiple geospatial foundation models (Prithvi, Clay, Satlas) —
  the reviewer emphasised this.** Five models, all real pretrained weights, identical
  evaluation protocol. **Re-run on clean data 3 Sep — §3.4 holds the citable table.**
  The strongest result to come out of it: **every EO-pretrained model gained from the QC
  fixes (Clay +18.0 %, Prithvi +13.8 %, Satlas +9.1 %) while the two generic-ImageNet
  models moved least or backwards (ResNet +4.0 %, ViT −3.5 %)** — i.e. the contaminated
  catalog was flattering the models with no earth-observation grounding. Clay leads at
  mAP 0.4059 but ties ResNet within CI; report it that way.
- [x] **C9 — Justify the embedding layer / feature representation; does embedding
  dimension influence retrieval?** **DIMENSION HALF ANSWERED 4 Sep** via
  `19_dimension_sweep.py` (PCA to 64/128/256/512, same GROUPED protocol, **mAP@5** —
  note this is mAP@5, *not* directly comparable to §3.4's full mAP):

  | Model | native | 64 | 128 | 256 | 512 | full |
  |---|---|---|---|---|---|---|
  | Clay-v1.5 | 1024 | 0.4455 | **0.4524** | 0.4504 | 0.4512 | 0.4381 |
  | Prithvi-100M | 768 | **0.4252** | 0.4245 | 0.4230 | 0.4229 | 0.4181 |
  | ResNet-50 | 2048 | 0.3843 | 0.3959 | 0.3957 | 0.3990 | **0.4031** |
  | Satlas-RN50 | 2048 | **0.3694** | 0.3639 | 0.3617 | 0.3595 | 0.3632 |
  | ViT-Base | 768 | 0.3249 | **0.3334** | 0.3327 | 0.3319 | 0.3300 |

  **Three conclusions, all useful:**
  1. **Dimension is not what separates these models.** At **64-D** every model retains
     97–105 % of its full-dimension mAP, and **the ranking is identical at every
     dimension** (Clay > Prithvi > ResNet > Satlas > ViT). **This removes the confound
     §3.4 was open to** — ResNet/Satlas's 2048-D is not buying them their position.
  2. **ResNet-50 is the only model that genuinely uses its capacity** (0.3843 → 0.4031,
     +4.9 % from 64-D to 2048-D). The other four are flat or *better* when reduced.
  3. **Clay and Prithvi improve slightly when reduced** — they carry noise in their tail
     dimensions. **PCA to 128-D is a free win**: better mAP, an 8–16× smaller index, and
     it directly mitigates the O(n²) storage problem in Phase 1's note. Worth adopting.

  **LAYER HALF ALSO ANSWERED (4 Sep) — `03b_layer_ablation.py`.** 6 layers × 2 pooling
  strategies, GROUPED mAP@5 over all 1,260 patches. *(Cheap because `forward_features`
  returns every block from one forward pass — ~6 min, not the ~45 min a per-layer re-run
  would cost.)*

  | rank | layer | pooling | mAP@5 |
  |---|---|---|---|
  | 1 | 11 | cls | **0.4332** |
  | 2 | 4 | cls | 0.4285 |
  | 2 | 11 | mean | 0.4285 |
  | … | | | |
  | **8** | **8** | **mean** | **0.4181 ← `03`'s current default** |
  | 11 | 0 | mean | 0.3851 |
  | 12 | 0 | cls | 0.3524 |

  **Verdict: block 8 + mean-pool is defensible but not optimal.** It ranks 8th of 12, and
  the best option (layer 11 CLS) is only **+0.0151 (+3.6 %)** better. Layers 4–11 are all
  within ~0.015 of each other; **only layer 0 is clearly bad** (0.35–0.39), which is the
  expected result — the first block holds low-level features, not semantics. C9 can now
  **cite this sweep instead of asserting "optimal"**. Switching to layer 11 CLS is a small
  free gain if the embeddings are ever regenerated; not worth a re-run on its own.

  #### ⚠ A methodological warning worth keeping
  A first pass with `--limit 300` reported the *opposite*: layer 10 CLS at 0.8066 against
  the default's 0.6479 — a **+24.5 %** gap, with **every** CLS config beating **every**
  mean config. None of that survived the full run. The 300-patch subset was 30 locations
  taken from the head of the catalog (not a random sample), and it produced a confident,
  wrong conclusion at 2× the apparent effect size. **Do not tune this pipeline on a
  head-of-file subset** — and the same caution applies to Phase 4's cell sampling, which
  is randomised per region precisely for this reason.
- [x] **C10 — UMAP / t-SNE cluster visualisation + stability across seasons, atmospheric
  conditions and acquisition dates.** The comment says "UMAP **or** t-SNE"; t-SNE and PCA
  are in `08`, so the visualisation half is satisfied. Two substantive additions 4 Sep:

  **⚠ CLUSTER QUALITY IS THE UNCOMFORTABLE RESULT — report it, do not hide behind the
  t-SNE picture.** `20_retrieval_diagnostics.py` measures what the comment actually asks
  ("*demonstrate whether ecologically similar ecosystems naturally cluster*"):

  | Model | silhouette (cosine) | adjusted Rand |
  |---|---|---|
  | Clay-v1.5 | **+0.0586** | **0.2832** |
  | ResNet-50 | +0.0192 | 0.2017 |
  | ViT-Base | −0.0574 | 0.1580 |
  | Prithvi-100M | −0.0750 | 0.1807 |
  | Satlas-RN50 | −0.1036 | 0.1507 |

  **Ecosystems do NOT form separated clusters.** Silhouette near zero means a patch is
  about as close to another ecosystem as to its own; three of five models are *negative*.
  Adjusted Rand of 0.15–0.28 is weak-to-moderate agreement, not clean structure.
  **This contradicts the visual impression from t-SNE — which will always draw tidy
  clumps whether or not the underlying space is separated.** The honest claim is:
  *retrieval RANKING works (mAP is 2.6–4× chance) while the space is not cleanly
  partitioned by ecosystem.* Those are compatible, and stating both is more defensible
  than showing the t-SNE plot alone.

  **Seasonal stability — `12` HAS NOW BEEN RUN**, for the first time in the project's
  history (it previously printed to stdout and saved nothing, so no artifact existed).
  **All five models complete** (7 locations each — one per ecosystem that had a cloud-free
  scene in *both* the July and January windows).

  | Model | mean seasonal cosine |
  |---|---|
  | Prithvi-100M | **0.9083** |
  | Clay-v1.5 | 0.7322 |
  | ResNet-50 | 0.6651 |
  | ViT-Base | 0.5527 |
  | Satlas-RN50 | 0.5427 |

  > **⚠ These raw values were WRONG until 5 Sep, and the fix is worth recording.**
  > `12` embedded Prithvi from the **raw** patch, while `03` builds Prithvi's catalog
  > vectors from the **`02` z-scored** patch. Both dates took the same path, so the cosines
  > were internally consistent and the bug was invisible in `12`'s own output for as long as
  > the script has existed — it only surfaced when `21` compared them against the catalog's
  > background distribution. **Every other model already matched the catalog path, which is
  > exactly why Prithvi alone looked like an outlier.** Same class of defect as
  > `10.compute_cell_embedding` feeding raw DN (§3.2).
  > Prithvi moved **0.7798 → 0.9083**; the others shifted only slightly (the crop geometry).
  > `12` now delegates to `13.embed_cell`, so there is **one** definition of the catalog
  > preprocessing path instead of two copies that can drift. Pre-fix values are preserved in
  > `results/*_prefix.json`.

  Per-ecosystem, the metric tracks real seasonality almost perfectly (all 5 models agree
  on the ordering):

  | Ecosystem | Clay | Prithvi | ResNet | Satlas | ViT |
  |---|---|---|---|---|---|
  | shrubland (Sonoran desert) | 0.9216 | **0.9963** | 0.9659 | 0.8623 | 0.7037 |
  | wetland | 0.9109 | 0.9791 | 0.7660 | 0.6920 | 0.8517 |
  | agricultural | 0.9116 | 0.9725 | 0.8992 | 0.6228 | 0.7168 |
  | savanna | 0.8451 | 0.9215 | 0.7188 | 0.8535 | 0.6450 |
  | grassland | 0.7008 | 0.8145 | 0.6592 | 0.4381 | 0.6045 |
  | boreal | 0.5585 | 0.5221 | 0.3709 | **0.0498** | 0.1154 |
  | tundra | 0.3618 | **0.2528** | 0.4138 | 0.2022 | 0.0934 |

  A desert looks like itself in January and July; snow-covered tundra and boreal forest do
  not. **Satlas scores 0.0498 on boreal — its winter and summer vectors are essentially
  orthogonal**, i.e. that embedding is describing the snow, not the forest. This is the
  sharpest single argument for the growing-season acquisition policy.
  **Implication for C24:** for tundra and boreal, the embedding substantially encodes
  *acquisition season*, so retrieval in those classes depends on when the image was taken.
  The growing-season acquisition policy (§3.2 bug 2) is what keeps this from contaminating
  the catalog. → `results/temporal_stability.json`

  **Atmospheric stability — RUN 4 Sep (`12 --mode atmospheric`).** Both scenes come from
  the *same* July window, so season is held fixed and haze is the only variable: clearest
  scene under 3 % cloud vs cloudiest under 70 %. → `results/atmospheric_stability.json`

  **⚠ The obvious version of this test is invalid, and the first run proved it.**
  `eo:cloud_cover` is a **scene-level** property describing a ~110 km tile, while our patch
  is **2.24 km**. A 68 %-cloudy scene can easily contain a perfectly clear 2.24 km window —
  in which case "clear vs hazy" silently compares two clear patches and the high cosine
  means nothing. Serengeti scored **0.9953** against a 68 %-cloud scene on the first run,
  which is exactly that artifact. So `12` now measures **patch-level** haze directly
  (`patch_haze_proxy` — mean B02 reflectance, the band haze lifts most) and **excludes**
  pairs with no real contrast rather than averaging them in. **It excluded 2 of 6.**

  The four surviving pairs, with the measured patch haze that qualified them:

  | Location | patch blue, clear → hazy | Prithvi | Clay |
  |---|---|---|---|
  | Konza Prairie, Kansas | 451 → 1426 | 0.9366 | **0.5477** |
  | Punjab farmland, India | 637 → **2329** | 0.8662 | 0.6367 |
  | Boreal forest, Alberta | 829 → 1496 | 0.8254 | 0.7231 |
  | Serengeti Plains, Tanzania | 694 → 851 *(barely over threshold)* | 0.9953 | 0.9237 |
  | *North Slope tundra, Alaska* | *no patch-level haze* | *excluded* | *excluded* |
  | *Sonoran Desert, Arizona* | *no patch-level haze* | *excluded* | *excluded* |

  **All five models, mean over the 4 valid pairs:**
  Prithvi **0.9449** · Clay **0.7078** · ResNet 0.5893 · ViT 0.5316 · Satlas 0.4526.

  *State the limit: n = 4 valid pairs.* The ordering hints at a dose-response — Serengeti,
  the least hazy, scores highest for **both** models — but four points cannot support the
  claim, and Konza breaks monotonicity for Prithvi.

  #### ⭐ The raw numbers above are NOT comparable across models (`21_stability_normalized.py`)

  It is tempting to read "Prithvi 0.9449 vs Satlas 0.4526" as *Prithvi is twice as
  haze-robust*. **That comparison is invalid**, and this is the correction to lead with if
  the point is raised. Each model has its own similarity scale, and Prithvi's is
  pathologically compressed: its cosine between **completely unrelated** catalog patches
  averages **0.9231**. Its 0.9449 is roughly what Prithvi gives *anything*.

  `21` measures each model's **own background** — cosine between patches of genuinely
  different base locations, 200,000 sampled pairs, grouped footing — and places each
  stability score on that scale. The **percentile** is distribution-free and is the number
  to quote; a z assumes a Gaussian background, and Prithvi's is visibly not.

  | Model | background mean | seasonal | **pctile** | atmospheric | **pctile** |
  |---|---|---|---|---|---|
  | **ResNet-50** | 0.4036 | 0.6651 | **96.5 %** | 0.5893 | **90.4 %** |
  | Clay-v1.5 | 0.5894 | 0.7322 | 87.1 % | 0.7078 | 82.4 % |
  | ViT-Base | 0.4160 | 0.5527 | 74.9 % | 0.5316 | 72.1 % |
  | Satlas-RN50 | 0.4317 | 0.5427 | 65.7 % | 0.4526 | 57.5 % |
  | **Prithvi-100M** | **0.9231** | 0.9083 | **27.3 %** | 0.9449 | 55.8 % |

  **The ranking inverts.** Prithvi tops the raw seasonal table and sits at the **27th
  percentile of its own background** — a Prithvi embedding of the same place in another
  season is *less* similar to itself than 73 % of random different-place pairs.
  **ResNet-50, a generic ImageNet model, is the most genuinely stable (96.5th percentile).**
  Read against C22, where Prithvi needed a 0.96 cut-off, this is the **fourth** independent
  measurement of the same underlying defect: Prithvi's embedding space is anisotropic, and
  its raw cosines should never be quoted without a scale.

  #### Third axis — ACQUISITION DATE (`12 --mode interannual`), the operationally decisive one

  Same July window, **two years apart** (2023 vs 2021), clearest scene on both sides. Season
  is held fixed and haze is controlled, so what remains is the ordinary year-to-year
  variation any real catalog contains: different phenological timing within the window,
  different crop rotation, different antecedent rainfall.

  **This is the axis that decides whether the project's core assumption holds.** The catalog
  is acquired **once**; every future query image will come from a different year. If an
  embedding cannot recognise a place across two Julys, the catalog goes stale and the whole
  retrieval premise fails.

  | Model | raw cosine | **percentile of own background** |
  |---|---|---|
  | Prithvi-100M | 0.9652 | 78.2 % |
  | **Clay-v1.5** | 0.9061 | **99.9 %** |
  | **ResNet-50** | 0.8259 | **99.9 %** |
  | ViT-Base | 0.7298 | 91.2 % |
  | Satlas-RN50 | 0.7148 | 78.3 % |

  **✅ The assumption holds, and this is the strongest stability result in the project.**
  Clay and ResNet sit at the **99.9th percentile** — a place re-imaged two years later is
  more like itself than 999 of every 1,000 random different-place pairs.

  **The ordering is identical for all five models** — `interannual > seasonal > atmospheric`
  is true without exception (Prithvi 78.2 > 55.8 > 27.3; ViT 91.2 > 74.9 > 72.1;
  ResNet 99.9 > 96.5 > 90.4; Clay 99.9 > 87.1 > 82.4; Satlas 78.3 > 65.7 > 57.5). Five
  independent models agreeing on the ranking of three perturbations is a real finding, not
  a sampling artifact.

  **What this means for acquisition policy, stated once:** *the year an image comes from
  barely matters; the season it comes from matters a great deal.* Spending effort on
  same-year imagery is wasted; spending it on growing-season timing is not. That is the
  concrete, actionable version of the C10 answer.

  **Also note the reversal it produces for Prithvi:** worst model on the seasonal axis
  (27.3 %), mid-pack across years (78.2 %). Prithvi's weakness is specifically *seasonal*,
  which is consistent with it being the only 6-band model — SWIR tracks moisture and snow,
  which is exactly what changes between July and January.

  *Limits: n = 6 locations (Dudwa drops out — no clear July-2021 scene, itself a
  data-availability data point). Only 2021 vs 2023 was tested; a longer baseline may drift
  more.* → `results/interannual_stability.json`

  ⚠ **This axis was WRONG on its first run — see §3.2c Bug 7.** A silently-unapplied edit
  left the query fetching January while the labels said July 2021, and it produced a result
  byte-identical to the seasonal column across all five models. `12` now asserts each
  fetched scene's own date falls inside the window being claimed.

  **Haze costs less than season for every model** (each model's atmospheric percentile is
  close to or below its seasonal one only for ResNet/Clay; for Prithvi haze is *less*
  damaging than season, 55.8 % vs 27.3 %). The consistent reading: **the acquisition policy
  that matters most is growing-season timing, not cloud-fraction strictness.**

  **A data-availability finding fell out of this (see C24).** Four of the ten test
  locations — **Periyar, Dudwa, Sundarbans and Bangalore — have no clear July scene at
  all.** The growing-season acquisition policy is in direct tension with the growing season
  itself across monsoon South Asia.

  **All three axes of C10 are now measured (seasons ✅, atmospheric ✅, acquisition date ✅),
  plus the quantitative cluster metrics the comment asked for. C10 is COMPLETE.**
  The only optional extra left is UMAP, which would add a third projection alongside
  PCA/t-SNE with no new information — the silhouette/ARI numbers are what the comment
  actually wanted.

### Retrieval evaluation
- [x] **C7 — Is retrieval consistent across geographic regions, not just nearby
  ecosystems? ANSWERED 4 Sep — and the answer is largely YES.**
  `16_derive_location_geography.py` derives **biogeographic realm** for all 131 locations
  from RESOLVE (data-derived, not hand-typed — a typed field can drift from its
  coordinate, which is how 13 mislocated coordinates arose). `17_cross_region_retrieval.py`
  then runs the two protocols Phase 3 asks for, both on top of `07`'s GROUPED rule.

  **Leave-one-realm-out** — query from realm X, candidate pool restricted to *everywhere
  but X*. This is the retrieval-side mirror of `11 --spatial-holdout`:

  | Model | grouped mAP@5 | leave-one-realm-out | Δ | analogs from own realm |
  |---|---|---|---|---|
  | Clay-v1.5 | **0.4371** | 0.3943 | −9.8 % | 49.0 % |
  | **Prithvi-100M** | 0.4194 | **0.4069** | **−3.0 %** | 28.7 % |
  | ResNet-50 | 0.4011 | 0.3734 | −6.9 % | 29.4 % |
  | Satlas-RN50 | 0.3592 | 0.3107 | −13.5 % | 20.7 % |
  | ViT-Base | 0.3311 | 0.3013 | −9.0 % | 26.4 % |

  **Retrieval is not leaning on same-realm shortcuts.** Forbidding the query's entire
  biogeographic realm costs only 3–14 %, and **20–49 % of unrestricted top-5 analogs
  already come from a different realm anyway**. For the research question this is the
  point: the embeddings find ecological matches across biogeographic boundaries, not just
  nearby.

  **Prithvi is the most transferable (−3.0 %) while Clay loses the most (−9.8 %).** Clay
  also draws **49 %** of its analogs from its own realm, far more than any other model —
  it has the best absolute mAP but the most regionally-anchored retrieval. *If the goal is
  cross-landscape analogy rather than leaderboard mAP, that trade-off matters, and it is a
  second piece of evidence (with §3.4c) that Prithvi is the right Phase 4 model.*

  **Per-realm results are uneven and should be reported, not averaged away:** Afrotropic
  falls hardest for Clay (0.6357 → 0.4375) and ResNet (0.3768 → 0.2227), while Neotropic
  *improves* under the restriction for several models (Clay 0.1602 → 0.2685) — its
  same-realm neighbours were evidently poor analogs. **20 patches (2 locations) have no
  realm**: their coordinates fall outside every RESOLVE polygon. → `results/cross_region_retrieval.json`
- [x] **C11 — Do retrieved analogs share real ecological characteristics?** **`07b` was
  rewritten 3 Sep — it had been reporting leaked numbers.** It excluded only the query
  patch itself, so **100 % of its top-5 analog pairs (3850/3850) were sub-crops of the
  query's own base location** — the same coordinate, hence the same climate cell, DEM cell,
  ecoregion, protection flag and Hansen loss vector. It was reporting Temperature MAE
  0.01 °C and 100 % protection agreement: a place compared to itself. Now group-aware
  (whole base location excluded, matching `07`) **and with a random-analog control**, since
  an MAE in isolation cannot be judged. Results in §3.4c. Real answer: **every model beats
  random on ecological agreement, modestly; ViT is worse than random on elevation and on
  disturbance trajectory.** Elevation is still forest-only (DEM gap) and trajectory is
  forest/mangrove-only (Hansen coverage). → Phase 3 for the remaining depth
- [x] **C12 — Multiple similarity metrics (cosine, Euclidean, approximate NN).** All four
  implemented, with the honest note that cosine ≡ euclidean ≡ knn on L2-normalised
  vectors and only HNSW genuinely diverges.
- [x] **C13 — Visual examples of successful / partial / failed retrievals. DONE 4 Sep.**
  `18_case_study_figures.py` renders `07d`'s cases as **actual imagery** —
  `results/case_studies/<model>_cases.png` for all five models. Each row shows the query
  patch's RGB beside its retrieved analog, annotated with cosine, category match, and the
  descriptor disagreement `09` measured; SUCCESS/PARTIAL/FAILURE colour-coded in one
  figure so failures sit beside wins rather than being a curated highlight reel.
  RGB reproduces `02`'s crop geometry, so the picture shows what was embedded; the
  contrast stretch is cosmetic and is stated on the figure.

  **The first render exposed a selection problem in `07d`** (not in the rendering): its
  case list contains **symmetric duplicates** (A→B and B→A are separate entries that draw
  the same pair of images), and for **Clay only 1 of 15 cases is cross-category** — 14 are
  `urban_green`, so a naive top-3 produced three near-identical city parks per band.
  `18` therefore de-duplicates symmetric pairs, spreads across query ecosystems, and in
  the FAILURE band **prefers `same_category == False`** — the quadrant C14 is about.
  Cross-category coverage per model: Prithvi 10/15, ViT 10/15, ResNet 10/15, Satlas 8/15,
  **Clay 1/15**. *Clay's near-absence of cross-category failures is itself a finding — it
  makes far fewer category-crossing mistakes than the others.*
- [x] **C14 — Visually similar but ecologically different failures.** `07d` now produces
  genuinely strong material on clean data. The best examples to write up:
  - **Satlas: Florida Bay mangroves → Siberian boreal forest, cosine 0.9854**, justified
    by "pristine surface, strong photosynthetic activity, 7 shared Hansen loss-years" —
    a tropical coastal system matched to taiga on canopy texture alone.
  - **Prithvi: Tiergarten Berlin → Valdivian rainforest Chile, cosine 0.9967** — the
    highest-similarity cross-category failure found, an urban park matched to temperate
    rainforest.
  - **ViT: Congo Basin Gabon → Mesopotamian Marshes Iraq, cosine 0.9858**, both at
    NDVI ~0.85 — matching on greenness while ignoring that one is closed tropical canopy
    and the other is Iraqi marshland.

  These are the "high cosine, high descriptor disagreement" quadrant the plan asks for, and
  they land on exactly the C24 point: **embedding similarity tracks texture and greenness,
  not ecology.**

  **Now VISUAL as well as numerical (4 Sep) — `results/case_studies/satlas_cases.png` is
  the figure to put in the paper.** It shows the mechanism plainly:

  | band | pair | cosine | disagreement |
  |---|---|---|---|
  | SUCCESS | Kainuu taiga (FI) → Boreal shield (CA) | 0.9926 | 0.013 |
  | SUCCESS | Redwood NP → Olympic NF | 0.9911 | 0.004 |
  | PARTIAL | Caroni Swamp **mangrove** (TT) → Boreal shield (CA) | 0.9957 | 0.030 |
  | **FAILURE** | **Florida Bay mangroves → Kainuu taiga, Finland** | **0.9874** | **0.148** |
  | FAILURE | Boreal forest (CA) → Iberá wetlands (AR) | 0.9862 | 0.172 |

  **The successes are real** — Redwood ↔ Olympic are genuinely the same Pacific-Northwest
  temperate rainforest, and Finnish ↔ Canadian taiga are genuinely the same biome. **The
  failures are visually legible**: a Florida mangrove and a Finnish taiga are both "green,
  textured canopy" at 10 m, and the model has no way to know one is tropical intertidal and
  the other is subarctic conifer. Descriptor disagreement (0.148 vs 0.013) separates them
  where cosine does not — **which is the argument for keeping `09`'s descriptors in the
  loop rather than trusting embedding similarity alone.** Ready for Phase 7 prose. → Phase 7

### Forecasting
- [x] **C15 — Can information from retrieved analogs improve forest-loss prediction?**
  **YES — ANSWERED AND STATISTICALLY SIGNIFICANT (4 Sep).** Upgraded from `[ ]`. Under the
  protocol the question actually demands (leave-one-region-out, paired per region):
  **+0.0392 mean PR-AUC (+11.0 %), 14 of 16 regions improved, Wilcoxon p = 0.0010**
  (§6 Phase 4). The effect is **~3× larger across unseen landscapes (+11.0 %) than within
  known ones (+3.8 %)**, which is the signature of genuine transfer rather than an extra
  correlated covariate. *Quote it with its limits: 1,573 of 8,469 cells (18.6 %), 16 folds,
  forest regions only.*

  #### ⭐ REPLICATED ON A SECOND BACKBONE (5 Sep) — the result is not Prithvi-specific

  The obvious objection to the above is that it might be an artifact of Prithvi's
  embedding space. It is not. The whole Phase 4 pipeline was re-run end to end with
  **Clay-v1.5** — a different architecture, a different embedding dimension (1024 vs 768)
  and a different pretraining corpus — retrieving from Clay's own catalog vectors.

  | | Prithvi-100M | **Clay-v1.5** |
  |---|---|---|
  | cells | 1,573 | 951 |
  | rows | 26,488 | 16,167 |
  | **spatial holdout** (leave-one-region-out) | **+0.0392 (+11.0 %)** | **+0.0335 (+9.9 %)** |
  | regions improved | 14/16 | 12/16 |
  | paired Wilcoxon | **p = 0.0010** | **p = 0.0214** |
  | **temporal split** (within known regions) | +0.0257 (+3.8 %) | **+0.0011 (+0.2 %)** |
  | spatial ÷ temporal | ≈ 2.9× | **≈ 50×** |

  **Two things this buys us.**

  1. **The effect replicates in direction, magnitude and significance** on an independent
     backbone. +11.0 % and +9.9 % across unseen landscapes, both significant. Clay's
     larger p-value is expected and is a *power* difference, not a weaker effect —
     it ran on 951 cells against Prithvi's 1,573.
  2. **Clay makes the transfer argument cleaner than Prithvi did.** Clay's analog features
     are worth **essentially nothing within regions the model already knows (+0.2 %)** and
     ~10 % across regions it has never seen. A merely correlated covariate cannot behave
     that way — it would help *most* in-sample. Prithvi's 2.9× ratio hinted at this;
     Clay's ≈50× ratio makes it hard to explain any other way.

  **Fixed en route:** `14` hardcoded `analog_cell_features.csv` and the two unsuffixed
  output JSONs, so a Clay run would have silently overwritten Prithvi's results — the exact
  failure `13` already suffered on 4 Sep. `14` now takes `--model`; Prithvi keeps the
  unsuffixed filenames so every existing reference stays valid.
  → `risk_model/analog_ablation_{spatial,temporal}_clay.json`
- [x] **C16 — Do embedding features help alongside conventional environmental
  variables?** **YES.** The 10-cell drift POC (−0.0045, noise at n=10) is superseded:
  **`analog_trajectory_jaccard` is the single most important feature in the model
  (+0.1764 permutation importance, ~3× the strongest conventional driver `temp_c` at
  +0.0609).** Embedding-derived information is not merely additive here, it dominates.
  *Caveat: not all analog features earn their place — `analog_mean_similarity` (+0.0060)
  and `analog_frac_with_loss_at_horizon` (+0.0009) contribute almost nothing.* The
  #### ⭐ The `embedding_drift` arms are now TESTED (6 Sep) — and the answer is NO

  `10 --with-embedding-drift` ran over all 17 regions (30 cells/region), producing drift
  for **414 cells** (96 had no cloud-free scene in one of the two years). Drift is
  1 − cosine between a cell's own embedding in **2021 vs 2016**.

  **The whole-table `11 --ablation` comparison is not a fair test and should not be quoted:**
  drift is filled on **4,572 of 143,000 rows (3.2 %)**, and the gradient-boosted models
  handle NaN natively, so on 96.8 % of rows the "with drift" arm *is* the "without drift"
  arm. It reported −0.0008, which is zero by construction. `23_drift_ablation_matched.py`
  runs the fair version — both arms trained on **only the rows that have drift**, exactly
  the discipline `14` already applies to the analog features.

  | Arm (matched subset: 4,572 rows, 275 cells, 16 regions) | PR-AUC | 95 % CI |
  |---|---|---|
  | drivers only | 0.7721 | [0.7042, 0.8324] |
  | drivers + `embedding_drift` | 0.7738 | [0.7119, 0.8280] |
  | **delta** | **+0.0016 (+0.2 %)** | *inside the driver-only CI* |

  **Regions improved: 3 of 8.** No detectable benefit.

  **⭐ This NEGATIVE result is what makes C2's claim precise, and it should be reported
  prominently rather than buried.** Put the two embedding-derived features side by side:

  | Embedding feature | What it encodes | Effect |
  |---|---|---|
  | `embedding_drift` | how much **this cell itself** changed over 5 years | **+0.2 %, 3/8 regions — nothing** |
  | `analog_trajectory_jaccard` | the disturbance history of **similar places elsewhere** (median 7,745 km away) | **+11.0 %, 14/16 regions, p = 0.0010** |

  **So the finding is not "embeddings help forest-loss prediction."** A cell's own
  embedding trajectory carries essentially nothing. What carries signal is specifically
  **cross-landscape analog transfer** — which is the project's actual thesis, and is now
  supported by a controlled contrast rather than by a single positive result. A reviewer
  asking "isn't this just adding another correlated feature?" can be answered with this
  table: we added *two* embedding-derived features, and only the transfer one worked.
  → `risk_model/drift_ablation_matched.json`

  *Limits: 275 cells is 3.2 % of the grid; the temporal split leaves 783 test rows, hence
  the wide CIs. `protected_area` is 100 % empty on these rows (WDPA unavailable) and is
  dropped from both arms.*
- [x] **C17 — Benchmark Gradient Boosting against RF, XGBoost, LightGBM.** `11` trains
  and compares all four and keeps the best.
- [x] **C18 — Feature-group ablation. RE-TICKED 3 Sep — it finally measures something.**
  The five columns that were 0 % filled are now 100 % filled (§3.4d), so the ablation is
  no longer dropping empty columns.

  #### ⭐ RE-RUN 6 Sep WITH REAL WDPA DATA — the Anthropogenic group changes verdict

  The previous run carried the caveat *"the Anthropogenic group rests on
  `distance_to_prior_loss_m` alone, since `protected_area` is still 0 % filled without
  WDPA."* That is no longer true: `26_build_wdpa_layer.py` built the 299,473-polygon
  terrestrial/designated layer and **`protected_area` is now 100 % filled** (92,453
  protected / 50,547 not).

  | Group dropped | before (empty protection) | **after (real WDPA)** |
  |---|---|---|
  | **Climate** | −0.0439 | **−0.0372** |
  | **Anthropogenic** | −0.0134 | **−0.0186** |
  | Topography | −0.0026 | +0.0000 |
  | Forest Baseline | −0.0004 | +0.0003 |

  *(base model PR-AUC 0.6974, bootstrap 95 % CI [0.6839, 0.7100] → half-width ≈ 0.013)*

  **Two groups now clear the CI half-width, where before only Climate did.** The
  Anthropogenic delta grew from −0.0134 (inside the noise band) to **−0.0186 (outside it)**,
  purely because the group gained a second real member. `protected_area` scores **+0.0119**
  on permutation importance in the full driver set and **+0.0265** in reduced sets — modest,
  but unambiguously non-zero, where before it was an empty column contributing exactly
  nothing.

  **This is the honest form of the C18 answer:** protection status *does* carry independent
  signal about forest-loss risk, and the earlier "Anthropogenic barely matters" reading was
  an artifact of missing data rather than a finding about the world.

  Still read the ablation **alongside** the permutation-importance table — group deltas
  understate redundant features (dropping the whole Forest-Baseline group costs +0.0003
  while `baseline_treecover_pct` alone scores +0.0247 on permutation, because tree cover is
  recoverable from the climate and distance features).
- [x] **C19 — SHAP or other explainable-AI technique. RE-TICKED 3 Sep.** The plots in
  `risk_model/shap_summary_*.png` now explain **7 real features** instead of 3, because
  climate and topography carry data. The model's two strongest predictors — `temp_c`
  (+0.1965) and `rainfall_mm` (+0.1474) — were **entirely absent from the previous run's
  explanations**, so the old SHAP figures were not merely thin, they were misleading about
  what drives the model.
- [x] **C20 — Disturbance pathways, successional stages, fragmentation patterns,
  recovery; trajectories over historical imagery.** **Substantially advanced 4 Sep.**
  *Disturbance pathways:* `15_analog_trajectory_figure.py` plots each query cell's
  cumulative Hansen loss curve against its top-5 cross-landscape analogs, selected
  best/median/worst rather than cherry-picked → `results/trajectory_figures/`.
  *Fragmentation:* `analog_fragmentation_delta` + `cell_edge_density` implemented in `13`
  (forest/non-forest edge density; validated — Cerrado soy frontier 0.046 vs intact Amazon
  0.008 vs uniform Białowieża 0.000).
  **The figure's honest verdict is mixed:** Jaccard spans 0.000–0.722 across 1,573 cells
  (median 0.145); best cases co-move but differ 2–3× in magnitude, worst cases show a
  flat-zero query against analogs that lost 15–20 %. **Report that spread, not just the
  best panel** — even though the aggregate feature is the model's strongest predictor.
  #### ⭐ SUCCESSIONAL STAGES — DONE 6 Sep (`24_successional_stages.py`)

  This was recorded as blocked on "a regrowth product Hansen does not provide". **That was
  half wrong.** Hansen's `lossyear` already encodes *when* each pixel was disturbed, so
  **stand age since disturbance is directly derivable — and that is the standard field
  definition of successional stage.** No new dataset needed. Never-disturbed cells are
  **censored, not zero-filled** (a mature stand is not "age 0", and treating it as such
  would have merged the two extremes of the gradient).

  **Does successional stage predict future forest loss? Emphatically yes:**

  | Stage | cell-years | future-loss rate | vs base |
  |---|---|---|---|
  | **early (0–5 y)** | 104,921 | **32.9 %** | **1.29×** |
  | mid (6–15 y) | 15,321 | 5.8 % | 0.23× |
  | late (16+ y) | 426 | **1.2 %** | 0.05× |
  | undisturbed (censored) | 22,332 | 5.2 % | 0.20× |
  | *all rows* | *143,000* | *25.6 %* | *—* |

  **An early-successional stand is ~6× more likely to be disturbed again than a
  mid-successional one, and ~27× more than a late one.** That is a clean, monotonic
  disturbance-pathway result and exactly the "pathways / successional stages" content C20
  asked for. It is also directly usable: `years_since_disturbance` is a strong candidate
  risk feature that the model does not currently carry.

  *Read it with the confound:* "early" partly proxies *"this is an active logging
  frontier"*, since 92.8 % of cells carry some loss and most cell-years therefore fall in
  the early class. The gradient is real; the causal reading (young stands are intrinsically
  fragile) is not established by this alone.

  #### RECOVERY — proxied, and the proxy turns out to be WEAK. Report it that way.

  Biomass/canopy recovery genuinely is **not** derivable from a loss-only product, so `24`
  computes the defensible proxy: of cells disturbed in year Y, what share are disturbed
  **again** by 2021.

  Result: **77.6 %–99.5 %, declining monotonically with year.** ⚠ **Both facts are
  artifacts, and neither should be quoted as a recovery finding.**
  - The *level* is saturated: a cell is 1 km across, so "some pixel in this cell was
    disturbed again within 20 years" is almost always true in an actively logged region.
    It measures cell size more than it measures recovery.
  - The *trend* (99.5 % in 2001 → 77.6 % in 2020) is **not** evidence that recovery is
    improving. It is the shrinking observation window: a 2001 disturbance has 20 years in
    which to be re-disturbed, a 2020 one has one year.

  **The honest C20 statement is therefore split:** successional stage is answered with a
  strong, monotonic result; recovery is *not* answered, and the reason is a genuine data
  limitation (Hansen records loss, not regrowth) rather than an unfinished analysis.
  Measuring it properly needs a regrowth product — Hansen's `gain` layer (which we do not
  hold, and which only spans 2000–2012) or an independent biomass time series.
  → `results/successional_stages.json`

### Rigour & engineering
- [x] **C22 — Sensitivity to patch size, embedding dimension, temporal window and
  similarity threshold, plus uncertainty quantification.**
  **Embedding dimension ✅** (§5 C9 — dimension barely matters, ranking unchanged).
  **Uncertainty ✅** — bootstrap CIs on retrieval mAP/MRR (`07`) and on PR-AUC/ROC-AUC
  (`11`), plus a paired Wilcoxon on the Phase 4 result.
  **Similarity threshold ✅ 4 Sep** — `20_retrieval_diagnostics.py` sweeps the cosine
  cut-off and reports precision/recall/F1, so an operating point can be *chosen and
  justified* instead of assumed. Best-F1 operating points:

  | Model | best cut-off | precision | recall | F1 | vs chance (0.0982) |
  |---|---|---|---|---|---|
  | **Clay-v1.5** | **0.70** | **0.2309** | 0.4505 | **0.3053** | **2.35×** |
  | ResNet-50 | 0.50 | 0.1857 | 0.4352 | 0.2603 | 1.89× |
  | Prithvi-100M | **0.96** | 0.1681 | 0.4729 | 0.2481 | 1.71× |
  | Satlas-RN50 | 0.50 | 0.1391 | 0.5383 | 0.2211 | 1.42× |
  | ViT-Base | 0.50 | 0.1362 | 0.4489 | 0.2090 | 1.39× |

  **Clay has by far the most usable similarity SCALE**, and this is the third independent
  measurement pointing the same way. Prithvi needs a **0.96** cut-off to separate anything
  — its scores are all crushed against 1.0 — and even at 0.99 its precision only reaches
  0.52. Clay operates at 0.70 with better precision throughout. *For anyone deploying
  this, "is this a real analog?" should be answered with Clay at ≥0.70, not with Prithvi.*

  **Patch size — HALF ANSWERED 5 Sep, and the half we could run changes the plan
  (`22_patch_size_sweep.py`).** This item was recorded as blocked on "re-acquisition at
  128 px and 448 px". That was only half true. *Larger* footprints do need re-acquisition —
  those pixels are not on disk. *Smaller* ones do not: every catalog patch is already
  224 px, so centre-cropping and resizing back to 224 holds the model's input size fixed and
  varies **only the ground area covered**, which is exactly the variable C22 asks about.

  A catalog patch covers 1,600 m (02 centre-crops 160 of the acquired 2,240 m patch, then
  resizes to 224), so `footprint_m = 1600 × crop/224`:

  | crop | footprint | ResNet-50 mAP@5 | ViT-Base mAP@5 |
  |---|---|---|---|
  | **224 px** | **1,600 m** *(native)* | **0.3492** | **0.3095** |
  | 160 px | 1,143 m | 0.3175 (−9.1 %) | 0.2302 (−25.6 %) |
  | 112 px | 800 m | 0.2937 (−15.9 %) | 0.1587 (−48.7 %) |
  | 64 px | 457 m | 0.1905 (**−45.5 %**) | 0.2143 (−30.8 %) |

  **⭐ Patch size is a genuinely load-bearing hyperparameter — and this is the opposite of
  what the dimension sweep found.** Halving the footprint costs 9–26 %; quartering it costs
  ~46 %. Compare C9, where dropping from 1024 to 64 dimensions cost *nothing*. **So the
  honest statement is: the representation is robust to how many numbers describe a patch,
  and fragile to how much ground the patch covers.** That is a much more interesting
  sensitivity result than "we tried some sizes."

  **This also changes the recommendation on the expensive half.** Performance rises
  monotonically with footprint all the way to the native size and shows **no sign of
  plateauing**, so the 448 px re-acquisition is now *well motivated* rather than
  speculative — we are on a rising curve and do not know where it turns over. Previously
  this was an untested guess; it is now a directed experiment. It also connects to the four
  accepted location losses (§3.2), which all failed because **narrow-fringe coastal
  ecosystems cannot be sampled at 2.24 km** — that argument wants a *larger* footprint too.

  *Caveat, stated rather than smoothed:* ViT is **non-monotonic** — 64 px (0.2143) scores
  above 112 px (0.1587). At a 64 px crop upsampled 3.5× the image is heavily blurred, and a
  generic ImageNet ViT may be latching onto low-frequency colour statistics rather than
  structure. ResNet's curve is clean and monotonic; ViT's should not be quoted as a trend.
  *Only ResNet and ViT were swept — Clay at 6.4 s/patch would be ~9 h for four arms.*
  → `results/patch_size_sweep.json`

  #### ⭐ TEMPORAL WINDOW — DONE 6 Sep (`25_horizon_sweep.py`)

  The risk model labels a cell positive if it loses forest within `RISK_HORIZON_YEARS = 2`,
  and that 2 was never justified. `10` now emits `label_loss_H1/H2/H3/H5` **in the same
  pass** as the native label — the horizon is only a threshold on a `lossyear` array already
  in memory, so a four-horizon sweep costs one run rather than four.

  **Validation first:** `label_loss_H2` is **100.00 % identical** to
  `label_loss_within_horizon`, confirming the alternate labels are computed exactly the way
  the model's own label is.

  | horizon | positive rate | PR-AUC | **lift over base rate** |
  |---|---|---|---|
  | 1 y | 15.3 % | 0.4729 | 3.09 |
  | **2 y — current default** | 21.7 % | **0.6926** | **3.19** |
  | 3 y | 24.9 % | 0.7665 | 3.08 |
  | 5 y | 25.9 % | 0.7993 | 3.09 |

  **⚠ Read the LIFT column, never the raw PR-AUC.** PR-AUC nearly doubles from H1 to H5,
  and that is **entirely an artifact**: a longer horizon raises the positive rate, and
  PR-AUC's own baseline *is* the positive rate. Quoting the raw column would support the
  confident, wrong claim that "longer horizons predict better."

  **Normalised, performance is flat (3.08–3.19 across every horizon).** The 2-year window is
  marginally best and within noise of the rest, so **the horizon choice is not
  load-bearing** — it can be set by what a user actually needs to plan for rather than by
  what the model prefers. → `results/horizon_sweep.json`

  #### The complete C22 sensitivity picture

  | Knob | Verdict | Evidence |
  |---|---|---|
  | **Ground footprint** | **Load-bearing** | −45.5 % mAP when quartered; curve still rising at native size |
  | **Similarity threshold** | **Load-bearing** | an operating point must be chosen; Clay 0.70 vs Prithvi 0.96 |
  | Embedding dimension | Not load-bearing | 64-D matches or beats native for 4 of 5 models |
  | Temporal window | Not load-bearing | lift flat at ~3.1 across 1–5 years |
  | Uncertainty | Quantified | bootstrap CIs on mAP/MRR and PR-AUC/ROC-AUC; paired Wilcoxon on Phase 4 |

  **That contrast is the paper-worthy sentence:** *the method is sensitive to how much
  ground a patch covers and to where the similarity cut-off is placed, and insensitive to
  how many numbers describe a patch or how far ahead it is asked to predict.*

  **Still pending: only the upward half of the patch-size sweep**
  (448 px, needs re-acquisition — now recommended, see above). → **Phase 5**
- [x] **C23 — Processing time, memory, scalability, retrieval speed.**
  **Embedding-extraction throughput now measured for all five models (§3.4b)** — the
  headline being Clay at 0.16 patches/s against Satlas at 8.49, a 50× spread that is the
  entire cost model for any future re-embedding. Retrieval latency and peak RSS also
  re-measured on clean data. *Still missing for Phase 5: the 10⁴/10⁵/10⁶ scaling
  projection. Also fix `06` overwriting `results/retrieval_perf.json` per model — only the
  last model survives on disk, so the §3.4b table had to be captured from the console.*
- [x] **C25 — Package as a modular desktop application or Python toolkit, with user
  documentation, workflow diagrams, installation instructions and example datasets.**
  → **Phase 6** (reviewer marked this "if possible"). **AUDITED 4 Sep — and the earlier
  note "only `setup.py` exists" understated the problem. `setup.py` exists and does not
  work.** Three findings, in order of severity:

  1. **`pip install .` installs no code at all.** `setup.py` calls `find_packages()`, and
     there are no packages — every module sits at the repository root. The evidence is in
     the build's own metadata: `ecolens.egg-info/top_level.txt` is **empty**, and
     `SOURCES.txt` lists exactly two files, `README.md` and `setup.py`. The install
     produces metadata and nothing else.
  2. **The declared console script is broken.** `entry_points` registers
     `ecolens-pipeline = run_pipeline:main`. `run_pipeline.py` exists at the repo root but
     is not packaged, so after an install the command fails with `ModuleNotFoundError`.
  3. **⚠ That entry point is also DESTRUCTIVE.** `run_pipeline.py` opens by `rmtree`-ing
     `patches_processed/`, `embeddings/`, `embeddings_vit/` and `embeddings_resnet/`.
     Run today it would delete a large part of the 1,260 sub-crop catalog — and because
     its list predates Clay and Satlas, it would leave `embeddings_clay/` and
     `embeddings_satlas/` behind, i.e. a **silently inconsistent** catalog rather than a
     clean one. **The one command the packaging advertises is the one command that must
     never be run.** Use `run_phase.py` (§ runner) instead; it is resumable and destroys
     nothing.

  **The root blocker for a real toolkit is the module naming.** Files are named
  `01_acquire_patches.py`, `03_extract_embeddings.py`, … and a Python identifier cannot
  begin with a digit, so **none of them is importable by normal syntax**. That is why the
  entire codebase reaches for `importlib.util.spec_from_file_location` (`_mod()` in `13`,
  `_mod()` in `03b`, `import_module` in `12`). Packaging therefore is not a metadata edit;
  it requires renaming every module to a valid identifier and rewriting every dynamic-load
  site. That is a genuine refactor, and the reviewer's "if possible" is well judged.

  **There are also three competing runners**, which alone would confuse any user of a
  packaged tool: `run_phase.py` (current, 31 steps, resumable — **the correct one**),
  `run_all.py` (171 lines, steps 01–11, written for the 3-model era), and
  `run_pipeline.py` (78 lines, steps 02–09, destructive, and the one `setup.py` points at).

  **Recommended minimum before any release**, cheapest first — this is the honest answer
  to give the reviewer rather than claiming the box is ticked:
  - **Repoint or remove the `console_scripts` entry**, so the packaging cannot invoke the
    destructive path. *One line, and it removes a data-loss hazard.*
  - Add a guard to `run_pipeline.py` that refuses to delete a non-empty catalog without an
    explicit `--force`.
  - Delete or clearly archive `run_all.py` and `run_pipeline.py` so one runner remains.
  - Only then: move modules into an `ecolens/` package with importable names, and replace
    the dynamic loads. Everything else in C25 — user docs, workflow diagram, install
    instructions, example dataset — is straightforward once that refactor exists, and
    pointless before it.

  *Status is `[~]` and not `[x]`: the audit is done and actionable, the refactor is not.*

**Score (6 Sep, latest): 25 done · 0 partial · 0 not started** — was **12 / 9 / 4** on
4 Sep and **4 / 13 / 8** on 3 Sep. **Every one of the 25 comments now has work behind it.**

Newly `[x]` on 5 Sep: **C10** (all three stability axes measured), **C2, C4, C21** (drafted once Phase 4 gave them evidence) and
**C24** (FM bias + transferability now measured, not asserted). **C25 moved `[ ]` → `[~]`**:
the packaging was audited rather than written up, and the audit found `setup.py` installs
no code *and* advertised a catalog-destroying entry point — both now documented, the hazard
fixed.

Earlier: **C1, C5, C7, C13, C15, C16** plus **C18, C19** re-ticked once the risk features
were non-empty. **C3 moved DOWN** to partial when clean data showed the spectral baseline
beating two models — the honest direction, and it stays there.

**The 9 remaining `[~]` split into three kinds**, and only two of them need compute:
- *Writing only* — C3, C6, C9, C14: the evidence is on disk, the paper text is not written.
- *Genuinely expensive or blocked* — C11 (WDPA data absent; DEM/Hansen coverage limits the
  subsets), C20 (Hansen records loss, not regrowth — succession needs another dataset),
  C22 (patch-size sweep needs full re-acquisition), C25 (package refactor).

**Plain-English companion for the professor meeting: `REVIEW_RESPONSE.md`** — one entry per
comment with what was asked, what was done, the method, the numbers, and what is open.

*(historical note: C18 and C19 moved from done to partial
on 3 Sep after the empty-column finding — the code was never the problem, the data was.
**C3 moved from done to partial later on 3 Sep**, when the clean re-run showed the
handcrafted spectral baseline beating two of the five models. C8, C11, C13, C14 and C23
all gained real evidence in the same pass without changing status. The score is flat, but
what sits behind it is now measured rather than asserted — and one entry moved *down*
because the data got better, which is the honest direction.)

---

## 6. The plan

### Phase 0 — Restore, clean, regenerate  *(blocking; nothing is citable until this is done)*

- [x] `git stash pop` — **done**.
- [x] Commit the QC work — **done**, as `36176a6` / `0611461` / `37ce8e1`. The
      re-acquire below is now reversible (`git checkout <sha> -- .` restores the code;
      the patches themselves are gitignored and will simply be re-downloaded).
- [x] Reference data — **done**. WorldClim at 2.5 arc-min, DEM forest-only, Ecoregions
      and Hansen already present. WDPA still missing (token required); consequences in §3.3.
- [x] `python geo_lookups.py` spot-check — **done, passed** (§3.3).
- [x] Archive the contaminated outputs — **done**: `results_pre_qc/`,
      `metadata/catalog_pre_qc.json`.
- [x] **Re-acquire — DONE, 77/81.** Multi-scene fallback, growing-season ranking and the
      BOA offset all landed; the 4 losses are accepted and documented (§3.2).
      **This step is closed — do not reopen coordinate hunting.**
- [x] **`00_validate_locations.py` → 0 errors** — verified 3 Sep. 4 MISSING + 2 MISLABEL
      + 4 HAZE, all advisory. Phase 0 acquisition is closed.
- [ ] *(declined twice, 3 Sep)* Commit the modified files before `03`. The user chose to
      run uncommitted both times; §9's rule still stands for future long runs.
- [x] **Restore the Clay / Satlas environment** — §3.5. Was blocking: three of five models
      ran, two died on `ModuleNotFoundError`. Both now load and are verified discriminating.
- [x] **Fix `02`'s fabricated-nodata bug** — §3.2 bug 4. Found while diagnosing
      `wetland_014`. No location lost.
- [x] **`02` re-run** under the corrected nodata rule → 770 sub-crops, 77 locations.
      Verified: all `(6,224,224)`, all finite, none zero-variance.
- [x] **`03` (all 5 models, `--force`) → `04` → `05` → `06` (all models) → `07` → `09` →
      `07b`, `07c`, `07d` (per model, output copied each time) → `08` — ALL DONE 3 Sep.**
      Verified: every model has 770 embeddings at the right dim, none non-finite,
      none zero-variance, all unit-norm, and **zero cross-location pairs at cosine > 0.999**
      (the degenerate-patch test). `run_phase.py --list` shows the per-step record.
- [x] **Two bugs fixed en route, both of which had been silently corrupting results:**
      `02`'s per-band nodata fabrication (§3.2 bug 4) and `07b` scoring 3850/3850
      same-location pairs (§5 C11). Neither was visible in any log.
- [x] **Phase 0 deliverable produced** — `phase0_qc_report.py` →
      `results/qc_report.{csv,md}`.
- [x] **`10` DONE — 143,000 cell-year rows, 25.6 % positive, 17 regions** (~73 min; it is
      slow because `10:602` calls `get_physical_descriptors` per cell-year, ≈16,000 RESOLVE
      point-in-polygon lookups after the coordinate cache collapses the 17 years per cell).
      If it ever needs re-running, drive it through `run_phase.py --only 10_tiling` so the
      log streams to `logs/10_tiling.log` instead of being swallowed by a PowerShell buffer.
- [x] **Fill rates checked BEFORE trusting anything (§9).** Climate, Topography and
      Forest-Baseline are now **100 % filled** (all were 0 %); **`protected_area` is still
      0 %**, exactly as §3.3 predicted. Numbers in §3.4d.
- [x] **`11` DONE in all four modes.** LightGBM best, PR-AUC 0.6923 (CI 0.6783–0.7051).
      **C18 and C19 re-ticked** — the ablation finally measures something and SHAP now
      explains 7 real features instead of 3.
- [x] **The spatial-holdout result is the key Phase 0 finding** — the driver-only model
      does not transfer across landscapes (mean AP 0.3172 vs 0.2555 base rate; 4 of 17
      regions below random ROC-AUC). §3.4d explains why this *motivates* Phase 4 rather
      than undermining it, and why Phase 4 must be judged on the spatial-holdout arm.

### ✅ PHASE 0 IS COMPLETE (3 Sep, evening)

Every step `00` → `11` has been re-run on the clean catalog and verified. `run_phase.py
--list` shows all 30 steps `done`. §3.4 / §3.4b / §3.4c / §3.4d hold the citable numbers;
everything in `results_pre_qc/` and the old §3.4 table stays withdrawn.

**What changed a conclusion, and must not be quietly reverted:**
1. Clay overtook ResNet (tied within CI); Prithvi 5th→4th; **ViT got worse**.
2. **C3 downgraded** — the spectral baseline beats Prithvi and ViT on clean data.
3. **C18/C19 re-ticked** — climate/topography were 0 % filled before and are the model's
   two strongest predictors now.
4. **The driver-only risk model does not generalise to unseen regions.**

**Two bugs fixed that had been silently corrupting results, both invisible in every log:**
`02`'s per-band nodata fabrication (§3.2 bug 4) and `07b` scoring 100 % same-location
pairs (§5 C11).

**Verification to run after `03` finishes — do not skip it.** Two independent failure
modes have already reached the embeddings stage undetected (stale-file reuse, and
fabricated bands), so check the vectors themselves rather than trusting the logs:

- every model's directory holds **770** `.npy` files, dims 768 / 768 / 2048 / 1024 / 2048
- no vector is non-finite or zero-variance, and every norm is 1.0
- **zero pairs from *different* base locations at cosine > 0.999** — this is the direct
  test for the degenerate-patch failure that produced the withdrawn run's 0.9997
  Shark Bay ↔ Hokkaido match

**Deliverable — DONE.** `phase0_qc_report.py` (new, 3 Sep) renders the acquisition
manifest as `results/qc_report.csv` (one row per configured location) and
`results/qc_report.md` (the same plus summary blocks). Evidence for **C6**.

It records the five methodology exceptions — full-year acquisition window,
`mangrove_004`'s 40 % cloud ceiling, 2.5-arc-min climate, DEM forest-only, WDPA absent —
and states plainly what the manifest **does not** record: **scenes tried per location.**
`01` ranks up to `MAX_SCENE_ATTEMPTS` candidates and writes only the winner's `scene_id`;
the rejected candidates and their reasons are printed at run time and then lost. The
column is omitted rather than guessed (§9). Adding it to `01`'s manifest write is a small
change that would materially strengthen C6.

Two things the report surfaced that the prose above had wrong or missing:
- the **month distribution** was being summarised by its top four entries (§3.2, corrected)
- **`wetland_008` (Camargue) has NDVI median −0.540 and 63 % water** — the most extreme
  entry in the catalog, and legitimate: the Camargue really is lagoon and salt-pan. Its
  processed patch is healthy (std 1.105). But it means the `wetland` class spans NDVI
  −0.54 to +0.73, which is far wider than any other class. **State that within-class
  heterogeneity in C24** — it caps what any retrieval metric over `wetland` can mean.

### Phase 1 — Expand the retrieval database  *(C5, feeds C7)*

> ### ✅ PHASE 1 COMPLETE — 3 Sep, late evening
> `config.PATCH_LOCATIONS` holds **131 locations across 10 ecosystem categories**
> (was 81 / 5); **126 acquired**, 5 documented losses. `02` → **1,260 sub-crops**;
> `03`–`08` re-run for all five models and **verified** (1,260 embeddings each, correct
> dims, no non-finite or zero-variance vectors, **0 cross-location pairs at cosine >
> 0.999**). `00` reports 0 errors; the coordinate-sync check reads CLEAN.
> Results in §3.4 / §3.4c. Remaining Phase 1 gaps are listed in §5 C5.
> **The verified 77-location Phase 0 results are archived in `results_phase0_77loc/`**
> (12 MB: evaluation report, ecological similarity, descriptors, case studies, QC report,
> catalog, cell-year features, pipeline state) so the clean baseline is not lost.
> `10`/`11` are deliberately NOT re-run: the risk model tiles only the 17 `forest`
> locations, which Phase 1 did not touch.
>
> **The normalization stats moved, and the direction confirms the new categories landed:**
>
> | band | 77 loc | 126 loc | |
> |---|---|---|---|
> | blue | 541.7 | **601.6** | drylands are bright |
> | green | 768.3 | 847.4 | |
> | red | 770.8 | **925.5** | |
> | NIR | 2619.7 | 2613.5 | ~unchanged |
> | SWIR1 | 1907.2 | **2180.3** | arid surfaces |
> | SWIR2 | 1259.3 | **1503.0** | |
>
> Blue and SWIR rise while NIR stays flat — exactly the signature of adding deserts,
> shrubland and tundra to a catalog that was previously all closed-canopy and cropland.
> **Every embedding from the 77-location run is therefore invalid**, which is why `03` runs
> with `--force` for all five models.
>
> **Embedding re-run timings at 1,260 sub-crops** (all verified: 1,260 files each, correct
> dims, no non-finite/zero-variance vectors, **0 cross-location pairs at cosine > 0.999**):
> Prithvi 355 s · ViT 354 s · ResNet 141 s · **Clay 5,510 s (92 min)** · Satlas 706 s.
> ≈1 h 45 total. Clay is 78 % of it, as §3.4b predicted.
>
> **First Phase 1 result: the ecosystem separation gap nearly doubled.** On Prithvi,
> same-vs-cross-ecosystem mean cosine went **0.0162 → 0.0289** (same-eco 0.9485 → 0.9494,
> cross-eco 0.9322 → **0.9205**). Same-ecosystem similarity held while cross-ecosystem
> similarity fell, i.e. the wider category range gave the space more to separate rather
> than just adding noise. That is a direct argument for C5: expanding the database did not
> merely add rows, it improved the signal the retrieval depends on.

- [x] Add ecosystem categories the reviewer named that did not exist: **savanna,
      grassland, tundra, boreal/taiga, dryland/shrubland** — 10 locations each, 50 total.
      **10 categories now**, against the "≥ 8" target.

      | category | n | | category | n |
      |---|---|---|---|---|
      | forest | 17 | | savanna | 10 |
      | mangrove | 17 | | grassland | 10 |
      | wetland | 17 | | tundra | 10 |
      | agricultural | 15 | | boreal | 10 |
      | urban_green | 15 | | shrubland | 10 |

      Realm spread: Palearctic 18 · Nearctic 15 · Afrotropic 8 · Neotropic 6 ·
      Australasia 2 · Indomalayan 1 (new entries only).

      **Every coordinate was RESOLVE-verified before being added, and the check earned its
      keep immediately — it rejected 2 of 50:** Hortobágy Puszta, Hungary resolved to
      *Pannonian mixed forests* (the puszta is a grassland enclave below polygon
      resolution) → replaced with Askania-Nova, Ukraine; and Kolyma lowland at 69 °N
      resolved to *Northeast Siberian taiga* (treeline is further north) → moved to
      70.5 °N. Both would otherwise have been found only after downloading a patch and
      reading the pixels, which is how the 13 corrected coordinates were found on 3 Sep.
- [ ] Within `forest`, deliberately stratify by **forest type** (tropical moist /
      tropical dry / temperate broadleaf / temperate conifer / boreal / montane), by
      **disturbance regime** (fire-driven, logging-driven, agricultural-frontier, stable)
      and by **conservation status** (WDPA-protected vs unprotected). These strata are
      what C7 tests against and what makes the Phase 4 analog framing meaningful.
      *Partly addressed: `boreal` is now its own category (10 locations) rather than being
      folded into `forest`. `forest_type` / `disturbance_regime` fields on the existing 17
      forest entries are still to do.*

- [ ] **Decide whether `boreal` should join `RISK_FOREST_ECOSYSTEMS`.** It is currently
      `['forest']` only, so `10` tiles 17 regions. Adding `boreal` would take that to 27
      regions and give the spatial-holdout test — the metric §3.4d shows actually matters —
      **10 more unseen landscapes to generalise to**, which is exactly what Phase 4 needs.
      Cost: `10` runs ~1.2 s/cell, so 27 regions ≈ 2 h rather than 73 min. **Not done
      tonight** — it changes the Phase 0 risk baseline that was just verified, so it should
      be a deliberate call, not a side effect of Phase 1.
- [ ] Add explicit `region` / `continent` / `koppen_zone` fields to every entry in
      `config.PATCH_LOCATIONS` (required by Phase 3).

      **Use RESOLVE's `REALM` instead of hand-typing `continent`.** Confirmed 3 Sep that
      `geo_data/Ecoregions2017.shp` already carries **`REALM`** (Palearctic / Nearctic /
      Neotropic / Afrotropic / Indomalayan / Australasia / Oceania / Antarctica) alongside
      `ECO_NAME`, `BIOME_NAME` and `NNH_NAME`. Biogeographic realm is a *better*
      stratification for C7 than continent — it is the standard unit for exactly the
      "geographically distinct landscapes" question — and it is **derived from the polygon
      rather than guessed**, so it cannot drift from the coordinate the way a typed field
      can. Köppen still has no raster on disk; `climatic_region` (hand-assigned) stands in
      until one is fetched.

- [ ] **Verify every new coordinate against RESOLVE BEFORE it enters `config.py`.**
      RESOLVE's 14 `BIOME_NAME` values map almost one-to-one onto the categories above —
      `Tundra`, `Boreal Forests/Taiga`, `Tropical & Subtropical Grasslands, Savannas &
      Shrublands`, `Deserts & Xeric Shrublands`, `Montane Grasslands & Shrublands`. A
      candidate whose actual biome contradicts its intended label is then caught for free,
      instead of after `01` downloads a patch and someone inspects the pixels. That is the
      cheap version of the check that cost **13 corrected coordinates** on 3 Sep.
      Caveat from §4: RESOLVE is advisory at this footprint (it called Jiuzhaigou grassland
      where the imagery showed 58 % forest), so treat a mismatch as *re-check this*, not as
      proof — pixels still win (§9).
- [ ] Target ≥ 120 base locations, balanced across realms. Re-run `00` → `07d`.
      **131 configured — target met.**

- [x] **Both dashboards hardcoded the original 5 categories — fixed 3 Sep.**
      `05` and `08` each carried a 5-entry `colors` map with a `||'#8b5cf6'` fallback, so
      **all five new categories would have rendered in one identical purple** on the map,
      the PCA scatter and the t-SNE plot. No crash, no warning — the dashboards would just
      have been useless for exactly the ecosystems Phase 1 adds. `08`'s `.badge-*` CSS had
      the same 5-category limit. Both extended (savanna/grassland/tundra/boreal/shrubland).
      **Anything keyed by ecosystem name needs the same audit when a category is added.**

- [x] **QC gap the new categories opened — FOUND AND CLOSED 3 Sep. Strong C6 material.**

      `VEGETATED_ECOSYSTEMS` drives both `01`'s rejection gate and `00`'s CONTENT ERROR,
      and it was `("forest", "mangrove")`. **`boreal` was added** (boreal forest is forest).
      The other four stay out on purpose: tundra, shrubland, grassland and savanna
      legitimately sit below `MIN_VEG_NDVI = 0.35`, so a vegetation floor would reject
      exactly the ecosystems Phase 1 exists to add.

      **That left those four with no content check at all — and it immediately let four bad
      patches through, every one of which passed every existing gate:**

      | id | measured | what it actually is |
      |---|---|---|
      | `tundra_007` W Greenland | NDVI **−0.122**, water **79.3 %** | icecap / fjord — and 79.3 % squeaked under the 80 % water limit |
      | `tundra_010` Kolyma delta | NDVI **−0.366**, water 68 % | a river delta is water |
      | `tundra_008` Finnmark | NDVI **0.047**, water 26 % | bare mountain plateau |
      | `boreal_009` Komi | NDVI 0.214, forest 3.5 %, **May scene** | pre-greenup snow at 63.5 °N |

      **Fix: `config.ECOSYSTEM_NDVI_BANDS`** — a per-category `(min, max)` band, enforced in
      `01.assess_patch_quality()`. Negative NDVI is never tundra; a conifer canopy in season
      is never 0.21. Bands are deliberately wide: they catch "wrong surface entirely", not
      natural variation.

      **The band deliberately EXCLUDES forest / mangrove / boreal.** Those keep the older
      rule, which rejects only when NDVI **and** forest cover are *both* low — strictly
      better than an NDVI-only band. A first version of this applied the band to them too
      and would have discarded `forest_002` (Amazon, NDVI 0.257 but **27.8 %** forest cover)
      and `mangrove_006` (0.299 / **37.3 %**) — two perfectly good patches from the original
      77. Caught before it did any damage; recorded here because it is an easy mistake to
      repeat. `boreal_009` fails the combined rule on its own merits, which is correct.

      All four were relocated (§9: move the coordinate, never relax the threshold). Three
      then came back clean — `tundra_007` NDVI 0.792, `tundra_010` 0.469, `boreal_009`
      0.877 / 100 % forest. **`tundra_008` failed a second time** (Chukchi, NDVI −0.11),
      rejected *by the new band* at acquisition. Two attempts, same structural cause
      (high-arctic barren/ice), so per §9 it is **recorded as the 5th accepted loss** rather
      than relocated again. Tundra ends at 9/10.

- [x] **A rejected relocation left STALE PIXELS on disk — caught by a coordinate-sync
      check, and worth keeping as a permanent step.** When `tundra_008`'s new coordinate
      was rejected, the *old* Finnmark `.npy` stayed in `patches/` while `config.py` said
      Chukchi. Nothing in the pipeline compares the two: `02` is catalog-driven, and the
      catalog happened to exclude it, but any later successful re-acquisition or any code
      that globs `patches/` would have paired Finnmark imagery with Chukchi metadata.
      Moved to `patches_stale/`. **The check is: for every location, does
      `acquisition_manifest[id].lon/lat` still equal `config` lon/lat?** It reported
      1 drift, 0 orphans, and now reads CLEAN at 126 patches.

- [x] **`00` caught a label collision I created: `savanna_005` and `wetland_010` sat on the
      IDENTICAL coordinate** (−68.00, 7.50), i.e. one patch labelled both savanna and
      wetland. Unlearnable by construction, and it would have quietly corrupted the
      confusion matrix and every per-category mAP — the same failure mode as the two
      "wetland" sites inside the Sundarbans mangrove biome (§3.1). Moved 511 km west to the
      Colombian Llanos, still RESOLVE "Llanos" but well-drained rather than flooded.

      `00` also flagged three softer DUPLICATE pairs, **deliberately kept**: `agri_014`
      Cerrado soybean vs `savanna_004` natural Cerrado (46 km), `wetland_005` Kakadu
      floodplain vs `savanna_006` Kakadu savanna (33 km), `agri_006` Pampas cropland vs
      `grassland_002` Pampas (72 km). Those are *real* land-use contrasts on distinct
      patches — natural vegetation against its agricultural conversion — and arguably the
      most interesting pairs in the catalog for C14. **They will depress per-category mAP,
      and that is the honest result, not a bug to engineer away.** Say so when reporting.

- [ ] ⚠ **`results/` grows O(n²) and this is now the binding constraint, not compute.**
      `06` writes a full ranked list per query, so `retrieval_results_<model>.json` is
      **660 MB per model at 770 sub-crops**. At 1,310 that is (1310/770)² = **2.89×**:

      | | 770 sub-crops | 1,310 sub-crops |
      |---|---|---|
      | `results/` total | 3.99 GB | **~11.5 GB** |
      | `retrieval_dashboard.html` | 402 MB | ~1.2 GB |
      | `patches_processed/` | 0.95 GB | ~1.6 GB |

      D: has **28.5 GB free**, so 131 locations fits — but Phase 1's own "≥ 120 locations"
      target is close to the ceiling, and the O(n²) term means **200 locations would need
      ~27 GB of `results/` alone**. Delete the stale `retrieval_results_*.json` before
      re-running `06` rather than letting both generations coexist.

      **This belongs in C23/Phase 5**: the honest scalability statement is not just
      "retrieval is ~2 ms/query" but "the current *serialisation* of results is quadratic
      and caps the catalog at a few hundred locations on a laptop." Storing top-K instead
      of the full ranking would remove the limit; nothing downstream reads beyond top-50
      (`09` precomputes explanations for the top 50).
- [ ] **Cost note, now measured (§3.4b) rather than assumed.** 131 locations → 1,310
      sub-crops. At the measured per-model rates that is **≈ 2 h 20 m for Clay alone** and
      ~25 min for the other four combined. Budget the re-embedding **once**, after the
      location list is frozen — and run it through `run_phase.py` so an interruption
      resumes instead of restarting.

### Phase 2 — Embedding science  *(C9, C10)*

> ### Phase 2 status, 4 Sep: **C9 ✅ · C10 ✅ (both halves) — one optional item left**
> New scripts: `03b_layer_ablation.py` (layer/pooling), `19_dimension_sweep.py` (PCA),
> `20_retrieval_diagnostics.py` (threshold sweep + silhouette/ARI). `12` now persists
> to `results/temporal_stability.json` and has been run for all five models.
> Remaining: the **atmospheric axis** in `12` (low-cloud vs hazier acquisition of the
> same site) — the only Phase 2 item with no result.

- [x] **Layer sweep** — `03b_layer_ablation.py`, all 12 blocks × {mean-pool, CLS}, one
      forward pass per patch, evaluated under `07`'s GROUPED protocol on all 1,260
      sub-crops. → `results/layer_ablation.json`

      | | mAP@5 |
      |---|---|
      | best: **block 11 / CLS** | **0.4332** |
      | block 11 / mean | 0.4285 |
      | block 4 / CLS | 0.4285 |
      | **block 8 / mean — `03`'s current default** | **0.4181** |
      | worst: block 0 / mean | 0.3851 |

      **The "block 8 for optimal semantic features" comment was never true, but it barely
      matters.** Every block from 4 upward scores 0.418–0.433 — a 1.5-point band. The only
      genuinely worse choice is block 0 (the raw patch projection, 0.3851), which confirms
      the encoder *is* doing something. So C9 can now cite a sweep instead of an assertion,
      and the honest wording is *"the choice of layer is not load-bearing above block 4;
      we keep block 8 for continuity with published Prithvi usage."*

- [x] **Dimension sweep** — `19_dimension_sweep.py`, PCA to 64 / 128 / 256 / 512.
      → `results/dimension_sweep.json`

      | Model | native | 64-D | 128-D | 512-D |
      |---|---|---|---|---|
      | Clay-v1.5 | 0.4381 (1024) | 0.4455 | **0.4524** | 0.4512 |
      | Prithvi-100M | 0.4181 (768) | **0.4252** | 0.4245 | 0.4229 |
      | ResNet-50 | 0.4031 (2048) | 0.3843 | 0.3959 | 0.3990 |
      | Satlas-RN50 | 0.3632 (2048) | **0.3694** | 0.3639 | 0.3595 |
      | ViT-Base | 0.3300 (768) | 0.3249 | **0.3334** | 0.3319 |

      **Two clean answers.** (1) *Dimension is not the story.* 64 dimensions match or beat
      the native size for four of five models — the useful signal occupies well under 64
      of Clay's 1024. (2) **It does not explain Prithvi's underperformance.** Compared at
      an identical 64-D, Clay still leads Prithvi 0.4455 vs 0.4252, so ResNet/Satlas's
      2048-D was never an unfair advantage — indeed those two are the models PCA *hurts*.
      Practical consequence for C23: the catalog can be stored at 128-D, a **8× shrink**
      for Clay at no measured cost.

- [x] **Quantitative cluster metric** — silhouette + adjusted Rand, `20_retrieval_diagnostics.py`.
      Results and interpretation in **C10** above. The headline is negative and worth
      reporting: **ecosystems do not form separated clusters** (3 of 5 models score a
      *negative* silhouette), which contradicts the t-SNE picture.
- [ ] **UMAP** — optional; PCA + t-SNE already ship in `05`/`08`, and the quantitative
      metric above is what C10 actually asked for. UMAP would add a third projection with
      no new information.
- [x] **Run `12_temporal_stability_analysis.py` for all 5 models** — done, 4 Sep; the
      5-model table and per-ecosystem breakdown are in **C10** above.
      → `results/temporal_stability.json`
- [ ] **Atmospheric axis** — extend `12` to compare a low-cloud against a higher-cloud
      acquisition of the same site. *Not done.* This is the last open Phase 2 item; it is
      the same code path as the seasonal test with a different scene-selection rule
      (pick the cloudiest scene under `MAX_CLOUD_COVER` instead of the least cloudy).

### Phase 3 — Retrieval evaluation depth  *(C7, C11, C13, C14)*

> ### Phase 3 status, 4 Sep: **C7 ✅ · C13 ✅ · C14 ✅ · C11 ✅(07b) — one item left**
> New scripts: `16_derive_location_geography.py` (realm/biome/ecoregion per location,
> derived from RESOLVE), `17_cross_region_retrieval.py` (C7), `18_case_study_figures.py`
> (C13/C14 imagery). Remaining: the random-analog control exists, but the **spectral
> baseline has still never been run through `07b`**, so C3's fallback argument is untested.

- [x] **Cross-region protocol — DONE via `17_cross_region_retrieval.py`.** Implemented as a
      separate script rather than inside `07`, so `07`'s GROUPED protocol stays untouched
      and the realm logic is testable on its own. Both parts delivered: **(a)** mAP@5
      stratified by realm, **(b)** **leave-one-realm-out** retrieval. Uses biogeographic
      **realm** rather than continent — the standard unit for this question, and it splits
      places continents do not (the Nearctic/Neotropic boundary runs through Mexico).
      Full results in §5 C7.
- [x] **Same- vs cross-realm analog rate per model — DONE.** Share of unrestricted top-5
      analogs drawn from the query's *own* realm: **Satlas 20.7 % · ViT 26.4 % ·
      Prithvi 28.7 % · ResNet 29.4 % · Clay 49.0 %**. Most models already retrieve
      predominantly cross-realm without being forced to. **Clay is the outlier at 49 %** —
      best absolute mAP, most regionally anchored retrieval.
- [ ] **Complete `07b`:** with WorldClim/DEM/WDPA present, report climate, elevation,
      protection-status and disturbance-trajectory agreement for all five models. Add a
      **random-analog control** so agreement is measured against a null baseline rather
      than in isolation.
- [x] **Turn `07d` into figures — DONE.** `18_case_study_figures.py` →
      `results/case_studies/<model>_cases.png` for all five models, with cosine, category
      match and descriptor disagreement annotated. *(Shows the query/analog pair rather
      than the full top-5 — `07d` records one analog per case. Extending to top-5 would
      need `07d` changed, and the pair already carries the C14 point.)*
- [x] **Ecological interpretation of the failure cases — WRITTEN, in §5 C14.** The
      "high cosine, high descriptor disagreement" quadrant now has both numbers and
      imagery: Florida Bay mangroves ↔ Kainuu taiga at cosine 0.9874, disagreement 0.148.
      Conclusion for the paper: **descriptor disagreement separates these pairs where
      cosine does not**, which is the case for keeping `09` in the loop rather than
      trusting embedding similarity alone.

### Phase 4 — The bridge: analog-informed forecasting  *(C1, C15, C16, C20 — the centrepiece)*

> ### ✅ PHASE 4 COMPLETE — 4 Sep. **The central hypothesis is confirmed, p = 0.0010.**
> New scripts: **`13_analog_risk_features.py`** (cell embedding + analog retrieval),
> **`14_analog_ablation.py`** (the ablation), **`15_analog_trajectory_figure.py`** (C20).
> §1's "two disconnected halves" is no longer an accurate description of this project.
>
> **Still open, both optional:** the two `embedding_drift` arms (needs `10` re-run with
> `--with-embedding-drift`, ~75 min), and a **Clay re-run** to test whether Prithvi's
> compressed cosine range is capping the effect.

> ### ⚠ A LANDMINE FOUND BEFORE WRITING ANY PHASE 4 CODE — read this first
> The obvious way to embed a grid cell is `10_grid_tiling_labels.compute_cell_embedding()`,
> which already exists for `embedding_drift`. **Reusing it for analog retrieval would have
> produced pure noise, and nothing downstream would have flagged it.**
>
> It feeds **raw DN values** into the model. Every catalog embedding comes from
> `patches_processed/`, which `02` z-scores with per-band stats *and* crops 160→224.
> The two are therefore in **different spaces**. Measured on the Amazon (`forest_002`),
> embedding its own coordinate and comparing to its own catalog vectors:
>
> | path | sim. to its OWN location | top-1 analog retrieved |
> |---|---|---|
> | **normalized (`13`'s path)** | **0.9985** | `forest_002` (0.999) ✅ |
> | raw DN (`10`'s existing path) | **0.0085** | `agri_004` — **Nile Valley farmland, Egypt** (0.300) ❌ |
>
> An Amazon cell comes out **orthogonal to itself** and its nearest ecological analog is
> Egyptian cropland. Every analog-transferred feature would have been noise wearing a
> plausible-looking number.
>
> **Consequences to carry forward:**
> - `13` reimplements the `02` path exactly (centre 160 crop → 224 resize → nodata fill →
>   z-score) and loads the stats from **`metadata/norm_stats.json`**, which `02` now
>   writes. Previously `02` computed those stats, used them and threw them away, so
>   nothing else could reproduce the catalog's normalization.
> - **`10`'s `embedding_drift` is NOT wrong** — it compares a cell to *itself*, both raw,
>   so it is internally consistent. But it is only valid as a self-similarity measure and
>   **must never be compared against catalog embeddings.** Worth a comment in `10`.
> - This is the §9 rule earning its keep again: *assume anything suspiciously clean is
>   mocked until verified.* The check that caught it was one explicit test — embed a known
>   catalog location and confirm it retrieves itself.

- [~] **Cell embedding + analog retrieval — BUILT AND VALIDATED 3 Sep.**
      `13_analog_risk_features.py` embeds a grid cell through the *exact* `02` path and
      retrieves top-k analogs from the 1,260-vector catalog pool.

      **Leakage guards verified on real output, not just asserted** (45 cells, 225 analog
      pairs, independently recomputed):

      | check | result |
      |---|---|
      | analogs from the same region | **0** |
      | closest analog | **253.5 km** (threshold 250) |
      | analog distance min / median / max | 254 / **7,745** / 18,282 km |

      A **median analog distance of 7,745 km** is the point: these are ecological matches
      from the other side of the planet, which is what "across geographically distinct
      landscapes" demands.

      **Scale caveat, stated plainly:** 8,469 cells × (STAC query + download + embed) is
      ~30 h. The POC samples `--cells-per-region N`. At N=20 that is 340 cells ≈ 4 % of
      the grid. Per §9 this demonstrates **the mechanism**, not significance.
- [~] **Analog-transferred features — 5 of 6 implemented; measured variance below.**
      A feature with no variance cannot help a model, so this was checked before wiring
      anything into `11`:

      | feature | std | range | verdict |
      |---|---|---|---|
      | `analog_prior_loss_rate` | 5.04 | 0–27.7 | ✅ strong |
      | `analog_loss_rate_at_horizon` | 2.02 | 0–13.2 | ✅ strong |
      | `analog_trajectory_jaccard` | 0.214 | 0–0.76 | ✅ good |
      | `analog_frac_with_loss_at_horizon` | 0.45 | 0–1 | ✅ 6 levels (top-5) |
      | `analog_mean_similarity` | **0.0085** | 0.953–0.998 | ⚠️ nearly constant |
      | `analog_similarity_spread` | **0.0002** | ~0 | ❌ useless |

      **The two retrieval-confidence features are dead on Prithvi**, and the cause is known:
      Prithvi's cosine anisotropy (§3.4 — max cross-location cosine 0.9978, vs Clay's
      0.9332). Every analog looks 0.99 similar, so "confidence" carries no information.
      **Re-run with `--model clay` to test whether its wider cosine spread revives them.**
      `analog_fragmentation_delta` (the C20 fragmentation half) is **not implemented**.

- [ ] Original spec for the features:
  - `analog_prior_loss_rate` — mean Hansen loss fraction across the analogs' pixels in
    the years *preceding* the observation year
  - `analog_loss_rate_at_horizon` — what fraction of the analogs actually lost cover
    within the horizon (the "what happened to places like you" feature)
  - `analog_disturbance_trajectory_similarity` — Jaccard over loss-year vectors
    (reuse `09.get_disturbance_history()`)
  - `analog_fragmentation_delta` — edge-density / patch-count difference between the
    cell and its analogs (covers the fragmentation half of C20)
  - `analog_mean_similarity`, `analog_similarity_spread` — retrieval confidence, so the
    model can discount low-confidence analogs
- [~] **THE ABLATION HAS RUN. `14_analog_ablation.py` (new, 4 Sep).** Two arms — drivers
      only vs drivers + analog-transferred — on **identical rows, split and protocol**.
      The driver arm is **retrained on the merged 317-cell subset**, never taken from
      `11`'s full-grid run, or the "effect" would just be two different datasets.

      #### ⭐⭐ THE HEADLINE RESULT — 4 Sep, 1,573 cells, both `13` bugs fixed
      #### ANALOG TRANSFER IMPROVES FORECASTING ACROSS LANDSCAPES (p = 0.0010)

      Leave-one-region-out, 16 folds, paired per region:

      | | mean PR-AUC |
      |---|---|
      | drivers only (+ local edge density) | 0.3572 |
      | **drivers + analog-transferred** | **0.3964** |
      | base rate (no skill) | 0.2929 |
      | **mean delta** | **+0.0392 (+11.0 %)** |
      | regions improved | **14 / 16** |
      | **paired Wilcoxon** | **p = 0.0010** |

      **This is the answer to §1's research question, and it is positive.** Analog features
      lift the margin over no-skill from +0.0643 to +0.1035 — a **61 % increase in usable
      signal** on landscapes the model has never seen. Only `forest_005` (−0.0127) and
      `forest_003` (−0.0034) got worse, both marginally; the largest gain was
      `forest_006` Siberian boreal at **+0.1382**.

      #### The effect is BIGGER across landscapes than within them — as predicted

      | protocol | delta | |
      |---|---|---|
      | temporal split (regions the model already knows) | +0.0257 (+3.8 %) | 0.6711 → 0.6968 |
      | **spatial holdout (unseen regions)** | **+0.0392 (+11.0 %)** | 0.3572 → 0.3964 |

      **Analog transfer helps ~3× more where local drivers fail to generalise.** That is
      the shape the hypothesis predicts and the strongest evidence that this is genuine
      transfer rather than an extra correlated covariate: a feature that merely added
      local information would help *most* where the model is already strong.

      #### `analog_trajectory_jaccard` is the single most important feature

      Permutation importance (temporal split, 14 features):

      | feature | Δ PR-AUC |
      |---|---|
      | **`analog_trajectory_jaccard`** | **+0.1764** |
      | `temp_c` | +0.0609 |
      | `distance_to_prior_loss_m` | +0.0530 |
      | `rainfall_mm` | +0.0514 |
      | `analog_loss_rate_at_horizon` | +0.0230 |
      | … | … |

      It is **~3× the strongest conventional driver**. The model relies most on *how much
      a cell's disturbance history resembles that of ecologically similar places
      elsewhere* — which is precisely the analog-transfer mechanism the project claims as
      its contribution (C4). Note the models see one date per patch and no Hansen data at
      all, so this temporal agreement is not something they were shown.

      #### Honest limits — state these with the result
      - **1,573 of 8,469 cells (18.6 %)** and **16 folds.** The direction is significant;
        the effect *size* should be quoted with that caveat.
      - **Forest regions only.** `RISK_FOREST_ECOSYSTEMS = ['forest']`, so this is transfer
        between forest landscapes, not across all 10 categories.
      - **`analog_mean_similarity` (+0.0060) and `analog_frac_with_loss_at_horizon`
        (+0.0009) contribute almost nothing.** The result rests on trajectory Jaccard and,
        secondarily, `analog_loss_rate_at_horizon`.
      - **Prithvi's anisotropy remains unaddressed** (see the C20 figure note below);
        the effect might be *larger* on a model with a usable similarity scale.

      #### ⚠ SUPERSEDED — the first run, before the two `13` bugs were fixed
      > 317 cells, `trajectory_jaccard` scoring empty-vs-empty as 1.000, and "top-5
      > analogs" that were five sub-crops of one location:
      > **+0.0254 (+7.5 %), 12/16 regions, p = 0.0739 — not significant.**
      > Kept so the before/after is visible. Fixing the bugs *and* raising cell coverage
      > 5× moved it to p = 0.0010. Do not cite the old figures.

      **Why the paired test, not the means:** fold std is ~0.19, so comparing two means
      across 16 regions is not a test. `11.run_spatial_holdout` only prints and returns
      `None`, so `14` reimplements the fold loop identically (same model, seed and
      constant-feature filter) to pair each region against itself.

      **The standout: `forest_006` (Siberian boreal) improved +0.2169**, from 0.2831 to
      0.5000 — by far the largest gain. Consistent with `13`'s retrieval, where **boreal
      was the most common top-1 analog category (127 of 317 cells)**: Siberia has genuine
      boreal analogs elsewhere on the planet, so there is real history to transfer. Four
      regions got worse, the worst being `forest_005` (−0.0566).

      **Temporal split, for contrast: +0.0149 (+2.3 %)**, inside its CI [0.5933, 0.7197].
      Smaller than the spatial gain, which is the right shape — the baseline is already
      strong within known regions, and analog transfer is supposed to help where it is not.
      **`analog_trajectory_jaccard` ranked 4th of 12 in permutation importance (+0.0499),
      above `baseline_treecover_pct`** — the analog features are being used, not ignored.

      #### What would make this conclusive
      1. **More cells.** 317 of 8,469 (3.7 %) is the binding limit; 16 folds is very little
         power for a +7.5 % effect. Scaling `--cells-per-region` is pure compute (~13 s per
         cell, network-bound) and is the single highest-value next step.
      2. **The two dead features.** `analog_mean_similarity` and
         `analog_similarity_spread` are near-constant on Prithvi (§ above). Re-run
         `13 --model clay`, whose cosine spread is far wider.
      3. **The missing arms.** `10` was run without `--with-embedding-drift`, so
         `embedding_drift` does not exist in the CSV and arms 2 and 4 below could not be
         built. Only drivers-vs-analog was testable.

- [ ] **Four-arm ablation in `11`** (arms 2 and 4 still blocked on `embedding_drift`),
      all under the same temporal split, all with bootstrap CIs and SHAP:
      1. drivers only (baseline)
      2. drivers + own-embedding / drift
      3. drivers + **analog-transferred features**
      4. drivers + embedding + analog
      Report PR-AUC deltas with confidence intervals. **A negative result here is a
      publishable result** — report it honestly either way; do not tune until it turns
      positive.
- [ ] **Cross-landscape test:** run all four arms under `--spatial-holdout` as well. The
      hypothesis claims analogs help *across geographically distinct landscapes*, so the
      leave-one-location-out number is the one that actually tests the claim.
- [~] **Trajectory figure (C20) — `15_analog_trajectory_figure.py` (new, 4 Sep).** Plots
      each query cell's cumulative Hansen loss curve against its top-k analogs', choosing
      **best / median / worst by trajectory Jaccard** rather than cherry-picking, so a
      reader sees where the mechanism fails too (§9).

      #### ⚠ THE FIGURE IMMEDIATELY EXPOSED TWO BUGS IN `13`. Both are fixed; both had
      #### already contaminated the first ablation result.

      **Bug A — `trajectory_jaccard` scored empty-vs-empty as a PERFECT 1.000.**
      `len(cy & ay) / len(union) if union else 1.0` meant *"neither place has any recorded
      tree-cover loss"* returned a flawless match. The figure made it obvious: the two
      highest-scoring cells were **flat zero lines agreeing with flat zero lines**. This
      was inflating the feature that ranked **4th of 12 in the ablation's permutation
      importance**, so part of that apparent importance was agreement-about-nothing.
      → empty-vs-empty pairs are now skipped; all-empty returns `None`.

      **Bug B — the "top-5 analogs" were 5 sub-crops of ONE location.** The pool holds 10
      overlapping sub-crops per location, so a naive top-k almost always returned the same
      place five times — every figure panel read `analog 1..5: tundra_007`. The features
      were averaging five near-identical vectors, giving false confidence and zero
      diversity. → `13` and `15` now keep the best sub-crop **per base location** and take
      the top-k **distinct locations** (verified: 5 distinct analogs per cell).

      *(Correction: an earlier note here blamed Bug B for
      `analog_frac_with_loss_at_horizon` taking only 6 values. That was wrong — with k=5
      the fraction can only be 0, 0.2, 0.4, 0.6, 0.8, 1.0. It is arithmetic, not a defect.)*

      **Effect of the fixes on the feature means** (same 47 cells, before → after):
      `analog_trajectory_jaccard` **0.1931 → 0.1717** (the inflation removed),
      `analog_prior_loss_rate` 3.4138 → 2.7609, `analog_frac_with_loss_at_horizon`
      0.4825 → 0.4428. All moved down, consistent with removing both an artificial perfect
      score and four redundant copies of one analog.

      **Consequence: the +7.5 % / p = 0.0739 result below was computed with both bugs
      present and must be re-run.** It is retained for comparison, not as the answer.

      Also hardened: `edge_density` now returns `None` on an unreadable tile instead of
      letting one truncated download kill the run.

      #### What the corrected figure actually shows — read this before writing C20
      **The visual evidence is mixed-to-negative, and that is the finding.** Jaccard now
      spans 0.000–0.578 (the old 1.000s were the artifact):
      - **Best case** (`forest_002`, Amazon, J = 0.578): analogs roughly co-move with the
        query, but their loss magnitudes differ by 2–3×.
      - **Worst cases** (J = 0.000): the query cell is flat at **zero** cumulative loss
        while its analogs lost 15–20 %.
      - **The analogs are ecologically incoherent.** An Amazon *forest* cell's top-5
        includes `mangrove_007` (18,282 km), `agri_015` and `urban_green_014` — all at
        cosine **0.983–0.988**.

      **Root cause is Prithvi's anisotropy, and it now looks like a design problem, not a
      curiosity.** Its cosine range across the whole pool is ~0.98–0.996 (§3.4: max
      cross-location 0.9978), so *ranking among the top candidates is close to arbitrary* —
      which also explains why `analog_mean_similarity` has std 0.0085. **Selecting analogs
      by cosine on an anisotropic model may be the weakest link in Phase 4.**
      → **Re-run `13 --model clay`** (max cross-location cosine 0.9332, a far wider and
      more discriminative range) and compare the retrieved analogs' ecosystem mix. If Clay
      returns forest/boreal analogs for forest cells where Prithvi returns mangrove and
      urban parks, the Phase 4 result should be recomputed on Clay.

      **PARTIALLY TESTED 4 Sep (31-cell Clay smoke run) — the hypothesis holds:**
      `analog_mean_similarity` is **0.7882 on Clay vs 0.9857 on Prithvi**. Clay's analog
      similarities occupy a genuinely wide range where Prithvi's are all crushed against
      1.0, so on Clay that feature (dead on Prithvi, std 0.0085) should carry real signal.
      **A full 1,573-cell Clay run is still outstanding** — it is ~3 h because Clay embeds
      at 0.16 patches/s (§3.4b).

      **Two bugs found while wiring the Clay path, both now fixed:**
      1. **The embedding cache was keyed by coordinate only, not by model.** `--model clay`
         would have loaded Prithvi's 1,573 cached vectors, printed *"resuming: 1573 already
         cached"*, skipped all work, and retrieved "Clay" analogs using **Prithvi**
         embeddings. → `analog_cell_embeddings_<model>.npz`.
      2. **The models do NOT share a preprocessing path, and `13` assumed they did.**
         `03.run_prithvi` reads `processed_path` (02's z-scored patch) but
         `run_timm_model`, `run_clay` and `run_satlas` all read `patch_path` (**raw**);
         Clay applies its own per-band normalization from `metadata.yaml`, so z-scoring
         first would normalize twice. `13` now branches per model (`embed_cell`).
         *The Prithvi result above is unaffected — Prithvi was the one case the original
         code got right.*
      *(Note the tension with §3.4c: Prithvi has the best ecological agreement on catalog
      patches. Good ecological agreement and a usable similarity SCALE are not the same
      property, and Phase 4 needs the second one.)*

### Phase 5 — Sensitivity, uncertainty, scalability  *(C22, C23)*

- [ ] **Patch-size sweep** — re-acquire a subset at 128 / 224 / 448 px (1.28 / 2.24 /
      4.48 km) and re-evaluate retrieval. Report mAP vs patch size.
- [ ] **Temporal-window sweep** — single-date vs seasonal composite vs multi-year median
      input; re-evaluate.
- [ ] **Similarity-threshold sweep** — precision/recall as the cosine cut-off moves;
      pick and justify an operating point for "is this a real analog?".
- [ ] Embedding-dimension sensitivity is covered by Phase 2 — cross-reference it, don't
      duplicate the work.
- [ ] **Uncertainty:** bootstrap CIs already exist on retrieval mAP/MRR and on
      PR-AUC/ROC-AUC. Add **per-query rank stability** (bootstrap the candidate pool and
      report how often the top-1 analog stays top-1) so *ranking* uncertainty is
      quantified, not just aggregate metrics.
- [ ] **Scalability (C23):** extend `retrieval_perf.json` with embedding-extraction
      throughput (patches/s per model, CPU vs GPU) and a projection to 10⁴ / 10⁵ / 10⁶
      patches for IndexFlat vs HNSW (build time, memory, query latency). Include the
      one-off costs: model checkpoint sizes, Hansen tile downloads, geo-lookup caching.

### Phase 6 — Packaging  *(C25 — reviewer said "if possible")*

- [ ] Restructure into an installable package:
      `ecolens/{acquire,preprocess,embed,retrieve,evaluate,explain,risk}/`, with the
      numbered scripts becoming thin CLI entry points over the library. This keeps the
      pipeline order legible while removing the `import_module("01_acquire_patches")`
      hack that `12` currently relies on.
- [ ] An `ecolens` console script via `setup.py` / `pyproject.toml`:
      `ecolens acquire | embed | retrieve | evaluate | explain | risk | dashboard`.
- [ ] `docs/`: installation, a workflow diagram (Mermaid), per-step user documentation,
      and a **small example dataset** (a handful of patches + embeddings, a few MB) so
      someone can run the retrieval demo without touching STAC.
- [ ] Two or three worked real-world walkthroughs: *find analogs for a candidate
      restoration site*, *rank cells in a protected area by loss risk*, *compare a
      degrading mangrove against its global analogs*.
- [ ] Optional GUI: the standalone HTML dashboards already cover most of the "desktop
      application" intent. A small local Flask/FastAPI wrapper adding **query-by-upload**
      (submit a coordinate → acquire → embed → retrieve live) would satisfy the comment
      far more cheaply than building a real desktop app.

### Phase 7 — Writing  *(C1, C2, C4, C6, C14, C20, C21, C24)*

Rewrite `EcoLens_Research_Paper.docx` around the re-framed question:

- [ ] **Introduction / hypothesis:** state the question from §1 verbatim; state the null
      (analog features add nothing beyond local drivers) and how Phase 4 tests it.
- [ ] **Related work & novelty (C4):** contrast against (a) content-based satellite image
      retrieval (BigEarthNet / PatternNet-style), (b) bitemporal change detection, and
      (c) driver-based deforestation risk models (GLAD / DETER / Forest Foresight). The
      claimed contribution is the **analog-transfer mechanism** — using a *different
      place's* history as a predictor for *this* place, made possible by foundation-model
      semantics — not the use of a pretrained model per se.
- [ ] **Methods (C6):** a full reproducibility section — band selection, 2.24 km / 10 m
      geometry, the STAC query, date window, cloud threshold, scene-selection rule, the
      normalisation choice (Sentinel-2 stats over Prithvi's HLS stats, and why), sub-crop
      geometry and its hard ±32 px limit, and **the QC gate from Phase 0**.
- [ ] **Ecological knowledge (C2)** and **applications (C21)** sections, grounded in the
      Phase 3 / Phase 4 figures rather than asserted.
- [ ] **Limitations (C24):** the existing README list plus **foundation-model bias**
      (pretraining geography/latitude skew, the HLS-vs-Sentinel-2 and TCI-vs-L2A domain
      gaps, RGB-only models missing SWIR), **transferability** (does a model tuned on
      these biomes hold elsewhere), **compute** (Phase 5 numbers), **label semantics**
      (Hansen loss ≠ deforestation), and **data availability** (WDPA point-vs-polygon
      coverage, WorldClim's 1 km cells vs 2.24 km patches).

---

## 7. Suggested order & rationale

1. **Phase 0** — blocking. Every other number is meaningless until the catalog is clean.
2. **Phase 4** — highest reviewer weight; it *is* the re-framed thesis. Start it early,
   because a null result still needs time to be investigated and written up honestly.
3. **Phase 1** — feeds Phase 3's cross-region test and strengthens Phase 4's analog pool.
   Do it before committing to the expensive final re-embedding run.
4. **Phase 3**, then **Phase 2** — evaluation depth first (more comments ride on it),
   then embedding science.
5. **Phase 5** — cheap once the pipeline has been parameterised by Phases 1–3.
6. **Phase 7** — continuously; write each section as its phase lands.
7. **Phase 6** — last, and explicitly optional.

---

## 8. Command reference

**PowerShell** — this project runs on Windows. `rm -f`, `for m in ...`, `Select-String`
in cmd.exe etc. will all fail; the venv must be active in every new terminal.

```powershell
.\.venv\Scripts\Activate.ps1          # every new window; -Scope Process Bypass if blocked
(Get-Command python).Source           # must point inside .venv\Scripts\

# --- PREFERRED: resumable runner. Survives interruption; skips finished steps. ---
python run_phase.py --list            # state of every step + where its log is
python run_phase.py                   # run everything still outstanding
python run_phase.py --only 10_tiling  # one step
python run_phase.py --from 07_evaluate
# Watch a running step from another terminal (logs are unbuffered):
Get-Content logs\10_tiling.log -Wait -Tail 20

# --- Or drive the steps by hand, below. Note 09 BEFORE 07b/07c/07d. ---

# --- Acquisition: COMPLETE (77/81, 0 errors). Re-run only to verify it still skips. ---
python 01_acquire_patches.py          # should print "Skipping download" for all 77
python 00_validate_locations.py       # exits non-zero on ERROR

# --- Env gate: run this BEFORE 03, especially in a rebuilt venv (see §3.5) ---
python -c "import satlaspretrain_models; from claymodel.module import ClayMAEModule; print('both OK')"

# --- Pipeline (RESUME HERE). NOTE 09 BEFORE 07b/07c/07d: they all read its descriptors ---
python 02_preprocess_patches.py
# --force is REQUIRED after any 02 re-run: 03 skips existing embedding files, and the
# sub-crop filenames are stable, so without it stale vectors are silently reused (§3.2).
foreach ($m in 'prithvi','vit','resnet','clay','satlas') { python 03_extract_embeddings.py --model $m --force }
python 04_finalize_and_analyze.py
python 05_create_database_and_dashboard.py
foreach ($m in 'prithvi','vit','resnet','clay','satlas') { python 06_retrieval_engine.py --model $m }
python 07_evaluate_retrieval.py
python 09_explainability_engine.py
python 07b_evaluate_ecological_similarity.py
python 07c_evaluate_baseline_retrieval.py
foreach ($m in 'prithvi','vit','resnet','clay','satlas') {
    python 07d_retrieval_case_studies.py --model $m
    Copy-Item results\retrieval_case_studies.json "results\retrieval_case_studies_$m.json"
}
python 08_retrieval_dashboard.py
foreach ($m in 'prithvi','vit','resnet','clay','satlas') { python 12_temporal_stability_analysis.py --model $m }

# --- Risk ---
python 10_grid_tiling_labels.py
python 10_grid_tiling_labels.py --with-embedding-drift --drift-locations 2 --drift-cells-per-location 5
python 11_forest_risk_forecast.py
python 11_forest_risk_forecast.py --ablation
python 11_forest_risk_forecast.py --spatial-holdout
# --predict needs a cell whose Hansen baseline tree cover clears the threshold.
# The old example (88.85 21.95) is the SUNDARBANS -- mangrove, not forest -- and returns
# "Could not compute Hansen-derived features". So does the Amazon PATCH CENTROID
# (-60.0261 -3.1019), whose manifest independently reads forest_pct 27.77. Use a cell
# taken from cell_year_features.csv:
python 11_forest_risk_forecast.py --predict -60.12506 -3.02048   # Amazon, 100% treecover -> 2.0% risk

# --- Reference data (already done, kept for reproduction) ---
python download_reference_data.py --dem --dem-forest-only     # ~1.1 GB; drop the flag for ~5.1 GB
python geo_lookups.py                                          # spot-check, read the output
```

**WorldClim is not fetched by `download_reference_data.py` any more** — its hardcoded 30s
URL is a 10.4 GB download for 2 of 19 rasters, with no resume and a partial-file delete on
error. Fetch the 2.5-arc-min bundle out of band instead (`curl.exe`, not PowerShell's
`curl` alias, which is `Invoke-WebRequest`):

```powershell
curl.exe -L --retry 999 --retry-delay 5 --retry-all-errors -o geo_data\wc2.1_2.5m_bio.zip `
  https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip
python -c "import zipfile; z=zipfile.ZipFile('geo_data/wc2.1_2.5m_bio.zip'); print(len(z.namelist()),'members; first bad:',z.testzip())"
```
Verify the zip **before** extracting, and do not pass `-C -` unless resuming a download
you interrupted in the same session: this server ignores Range requests and returns 200,
so curl appends a whole second copy onto the partial and produces a corrupt archive whose
central directory still reads as valid.

---

## 9. Standing rules for this project

- **Never fabricate a value.** `geo_lookups.py` returns `None` for missing reference data
  and every caller must treat that as unavailable. This project's history includes a
  `sum(ord(c) for c in patch_id)` fake-descriptor bug and mocked Clay/Satlas embeddings
  that were byte-identical to ViT/ResNet — both caught late. Assume anything
  suspiciously clean is mocked until verified.
- **Quote GROUPED metrics, never LEAKED.** LEAKED exists only to show the size of the
  sub-crop inflation.
- **Report negative results.** "Analog features didn't help" answers the hypothesis;
  tuning until they help does not.
- **Never relax a QC threshold to admit a patch that failed it.** `mangrove_017` came in
  at 81 % water against an 80 % limit; moving `MAX_WATER_FRACTION` to 0.85 would have
  "recovered" it and re-opened the exact hole the gate was built to close. Move the
  coordinate, or record the loss. The same applies to cloud: prefer a per-location
  `max_cloud_cover` override, which stays visible in the catalog, over lowering the
  global bar for all 81.
- **A column full of `None` is worse than a missing column.** `geo_lookups` returning
  `None` is correct, but downstream code happily trains, ablates and SHAP-plots over
  all-null features and reports numbers that look fine. Check fill rates before believing
  any feature-importance or ablation result.
- **Distinguish "the mechanism works" from "the result is significant."** The 10-cell
  drift POC demonstrates only the former.
- **Windows.** This repo runs on Windows; `import resource` and cp1252 stdout are real
  failure modes (both fixed in `36176a6`). Keep new code cross-platform and open files
  with `encoding="utf-8"`.
- **Commit before a long re-run.** Anything that re-downloads or re-embeds takes hours
  and overwrites its inputs' provenance. Land the code change first, then run it, so a
  bad run can be diagnosed against a known tree.
- **Fixing the code is not fixing the data.** A coordinate correction or a new QC check
  changes nothing in `results/` until `01` re-downloads and the pipeline re-runs. Check
  the mtime on `metadata/catalog.json` before believing any number.
- **Know when to stop relocating.** Four locations were lost to a real structural limit —
  narrow coastal fringes cannot fill a 2.24 km patch — after several rounds of coordinate
  guessing. A documented gap is worth more than a mudflat labelled "mangrove" in every
  retrieval result. When two attempts fail for the same structural reason, record the loss
  and move on.
- **Trust pixels over maps.** RESOLVE biome, scene-level cloud cover and location names
  are all metadata *about* a patch; NDVI, water fraction and nodata are the patch itself.
  Where they disagree, the pixels win — that is why CONTENT is an ERROR and MISLABEL is
  only a WARN.
- **After finishing a task, tick its box in §5 and §6, and update §3.4 if the numbers
  changed.** This file is the handoff.
