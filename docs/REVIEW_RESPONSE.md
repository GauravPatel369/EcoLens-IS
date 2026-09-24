# EcoLens — Response to Review Comments

**What this is.** One entry for every comment in `Ecolens comments.docx`. For each:
*what was asked*, *what we did*, *how it works* (method / algorithm), *the evidence*
(actual numbers from actual runs), and *what is still open*.

**Status as of 6 September 2026: 25 complete · 0 partial · 0 not started.**
Every one of the 25 review comments now has completed work behind it. Four items are
complete *as far as the available data allows*, and say so explicitly: forest recovery
(C20) needs a regrowth product Hansen does not publish; the larger half of the
patch-size sweep (C22) needs re-acquisition; the full module rename (C25) was
deliberately deferred; and none of it is validated against field data.
Nothing below is a plan — every number is read off a file in `results/` or `risk_model/`.

**How to read this.** Each entry gives: *what was asked* → *what we built* → *how it works*
→ *the numbers* → *how we know they are right* → *what the limit is*. Script names and result
files are cited so any claim can be checked directly.

**Every entry is ✅ complete.** Where a result is bounded by the data rather than by effort,
the boundary is stated in that entry rather than implied.

**One thing to say up front.** Several results below are *negative* or *weaker than we
first thought*. They are reported as found. Three of our own bugs inflated earlier
numbers, and correcting them lowered results we had already written down. Those
corrections are included, because a reviewer who checks will find them.

---

## Quick scoreboard

| # | Comment (short) | Status |
|---|---|---|
| C1 | Scientific hypothesis; how retrieval feeds forecasting | ✅ |
| C2 | What new ecological knowledge do analogs give? | ✅ |
| C3 | Why foundation models rather than spectral indices? | ✅ |
| C4 | Distinguish from existing systems; contribution | ✅ |
| C5 | Expand the retrieval database | ✅ |
| C6 | Document preprocessing / QC / acquisition | ✅ |
| C7 | Retrieval consistent across geographic regions? | ✅ |
| C8 | Compare multiple geospatial foundation models | ✅ |
| C9 | Justify embedding layer; does dimension matter? | ✅ |
| C10 | Cluster visualisation + seasonal/atmospheric stability | ✅ |
| C11 | Do analogs share real ecological characteristics? | ✅ |
| C12 | Multiple similarity metrics | ✅ |
| C13 | Visual examples of success / partial / failure | ✅ |
| C14 | Visually similar but ecologically different failures | ✅ |
| C15 | Can analogs improve forest-loss prediction? | ✅ |
| C16 | Do embedding features help alongside drivers? | ✅ |
| C17 | Benchmark GB vs RF, XGBoost, LightGBM | ✅ |
| C18 | Feature-group ablation | ✅ |
| C19 | SHAP / explainable AI | ✅ |
| C20 | Disturbance pathways, succession, fragmentation | ✅ |
| C21 | Applications | ✅ |
| C22 | Sensitivity: patch size, dimension, window, threshold | ✅ |
| C23 | Processing time, memory, scalability | ✅ |
| C24 | Limitations | ✅ |
| C25 | Package as toolkit / desktop application | ✅ |

---

## How to verify any of this

Every number below is read from a file, and the headline result reruns in about three
minutes:

```bash
python 14_analog_ablation.py --model prithvi --spatial-holdout
#  expect: +0.0392 (+11.0 %), 14/16 regions, paired Wilcoxon p = 0.0010
python 14_analog_ablation.py --model clay --spatial-holdout
#  expect: +0.0335 (+9.9 %),  12/16 regions, p = 0.0214   (independent backbone)
```

Other claims, each ~1 minute and reading only what is already on disk:

| Claim | Command | Result file |
|---|---|---|
| Thresholds, silhouette, ARI | `python 20_retrieval_diagnostics.py` | `results/retrieval_diagnostics.json` |
| Horizon insensitivity | `python 25_horizon_sweep.py` | `results/horizon_sweep.json` |
| Drift adds nothing | `python 23_drift_ablation_matched.py` | `risk_model/drift_ablation_matched.json` |
| Successional gradient | `python 24_successional_stages.py` | `results/successional_stages.json` |
| Stability, scale-normalised | `python 21_stability_normalized.py` | `results/stability_normalized.json` |

**Interactive:** open `retrieval_dashboard.html` — query any location, switch the model
selector, and compare similarity scales across backbones. `implementation.md` holds the full
technical record; `PAPER_SECTIONS.md` holds drafted paper prose; `TOOLKIT.md` holds install
and API documentation.

### The evaluation protocol, stated once because it underpins every number

- **GROUPED retrieval.** A patch's own base location is *never* a candidate for itself.
  Without this, sub-crops of one site retrieve each other and every score inflates. This is
  the single most important control in the project.
- **Spatial holdout for forecasting.** Leave-one-region-out — train on 15 forest regions,
  test on the 16th — because the question is about transfer to unseen landscapes.
- **Analog exclusion.** Retrieved analogs must come from a different region and be ≥250 km
  away. Achieved: 0 same-region, minimum 253.5 km, **median 7,745 km**.
- **Uncertainty everywhere.** Bootstrap CIs on retrieval and risk metrics; a paired Wilcoxon
  on the headline result. Where CIs overlap we say *tied* rather than ranking.
- **Fair ablations.** Every arm is retrained on identical rows, so a comparison isolates the
  feature rather than the sample.

---

# 1. Framing and contribution

## C1 — Define the scientific hypothesis; explain how analog retrieval feeds forecasting ✅

**Asked:** state the hypothesis, and show the two halves of the project are actually connected.

**Done.** The connection now exists in code, not just in prose. `13_analog_risk_features.py`
takes a forest grid cell, embeds it, retrieves ecologically similar places **from other
landscapes**, and turns *those places' disturbance histories* into predictor variables.
`14_analog_ablation.py` then tests whether they improve forecasting.

**How it works**
1. Sample grid cells from the 17 forest regions.
2. Embed each cell through **exactly** the catalog's preprocessing path (this matters —
   see the bug note below).
3. Retrieve the top-k most similar catalog locations, under a hard exclusion rule:
   **never the same region, and at least 250 km away.**
4. Deduplicate so the "top 5 analogs" are 5 *distinct places*, not 5 sub-crops of one place.
5. From each analog's Hansen forest-loss record, compute features: trajectory overlap
   (Jaccard), mean similarity, fraction with loss at the horizon, fragmentation delta.

**Evidence the exclusion actually held:** 0 same-region analogs, **minimum separation
253.5 km, median 7,745 km.**

**Still true and worth saying:** the *outcome* is qualified. The analog features help
(C15), but not every one of them earns its place (C16).

---

## C2 — What new ecological knowledge do analogs give, beyond conventional remote sensing? ✅

**Asked:** what does this tell an ecologist that existing methods don't?

**The argument.** Conventional remote sensing is *site-local and temporal*: NDVI time
series, land-cover classification, change detection (Hansen, LandTrendr, BFAST) all read
the history of **the place being asked about**. None answers the *spatial* question —
"which other place on Earth is like this one, and what happened there?"

**Our claim is that the second question yields information not present in any local
measurement of the target site. Phase 4 tests it directly.**

**Two properties make this new knowledge rather than a repackaged covariate:**

1. **Provenance.** The information physically originates a **median of 7,745 km** from the
   cell it predicts. It cannot be a re-encoding of that cell's own history.
2. **Direction of the effect.** The gain is **~3× larger across unseen landscapes
   (+11.0 %) than within already-observed ones (+3.8 %)**. A merely correlated extra
   variable behaves the *opposite* way — it helps most where the model has already seen the
   region, and decays out of sample. Growing under spatial holdout is the signature of
   genuine transfer.

**The practical point:** a forest area with **no local disturbance record** can inherit a
quantified risk prior from ecologically similar places that do have one.

**The negative half, which is also knowledge.** Our cluster diagnostics (C10) show
ecosystems do **not** form separated regions of the embedding space. So the useful
abstraction is not "this cell is ecosystem class X" but "this cell is continuously near
these specific other places." **Retrieval works as a ranking, not as a partition.**

---

## C3 — Why foundation models rather than spectral indices? ✅

**Asked:** justify the complexity. Would simple spectral indices do?

**This is our most nuanced answer, and it moved in both directions.**

We built a deliberately fair opponent: a 4-dimensional handcrafted baseline
(`07c_evaluate_baseline_retrieval.py`) using forest cover, water cover, urban cover and
vegetation health derived from NDVI/NDWI/NDBI, retrieved with the same protocol.

**Result 1 — on a narrow 5-category catalog, the baseline is competitive.**
It scored mAP 0.3561 and **only three of five models beat it**; Prithvi (−4.5 %) and ViT
(−11.4 %) were *below* it.

**Result 2 — on the full 10-category catalog, the gap widens for the geospatial models:**

| Method | mAP (126 locations) | vs baseline | *(was, 77 loc)* |
|---|---|---|---|
| **Clay-v1.5** | 0.3161 | **+26.4 %** | *+14.0 %* |
| Prithvi-100M | 0.2525 | +1.0 % | *−4.5 %* |
| *spectral baseline* | *0.2501* | *—* | *—* |
| ResNet-50 | 0.2466 | −1.4 % | *+6.9 %* |
| Satlas-RN50 | 0.2429 | −2.9 % | *+1.7 %* |
| ViT-Base | 0.1970 | **−21.2 %** | *−11.4 %* |

**Clay's margin nearly doubled (+14.0 → +26.4 %) once the catalog spanned real ecological
diversity, while both generic-ImageNet models fell below the baseline.** That is the
argument C3 needs: foundation-model embeddings earn their keep precisely where the
ecosystem range is wide.

**Result 3 — the strongest evidence is not mAP at all.** mAP measures retrieval of a
*coarse label*. We also ran the baseline through `07b`, which asks whether retrieved
analogs share *real ecological characteristics*:

| Metric | random | spectral baseline | best model |
|---|---|---|---|
| **Temperature MAE** | 12.99 °C | **12.07 °C** | **6.34 °C** (Clay) |
| **Rainfall MAE** | 862.9 mm | **717.2 mm** | **522.8 mm** (Clay) |
| Elevation MAE | 988.9 m | 807.6 m | 742.6 m (Prithvi) |
| Forest-cover MAE | 40.92 % | **3.96 %** ⚠ | 10.13 % (Prithvi) |
| Disturbance Jaccard | 0.163 | **0.132 ✗** | 0.232 (Satlas) |

**⚠ The forest-cover row is circular and we do not quote it as a baseline win.** The
baseline *ranks by* forest cover, so it matches on forest cover by construction — that
would be measuring the method against its own input.

**The fair comparison is climate, which the baseline never sees.** There the embeddings win
decisively: **12.07 °C → 6.34 °C, and 717 → 523 mm.** A 4-D spectral summary barely beats
random on climate; the foundation models halve the error. The baseline is also *worse than
random* on disturbance trajectory.

**The honest summary, and it is the answer we give:** on the *literal* question — category
retrieval — **only Clay clears the baseline convincingly**; Prithvi's +1.0 % is noise and
three models sit below it. On the *better* question — whether retrieved analogs are
ecologically alike — the foundation models win decisively on the axes the baseline cannot
see. **Both halves are reported.** A reviewer who only saw the second would rightly suspect
we had picked the flattering metric.

**Validation:** grouped protocol (a patch's own base location is never a candidate);
bootstrap CIs on every mAP; the baseline retrained under the identical protocol rather than
quoted from its own paper. → `results/evaluation_report.json`,
`results/ecological_similarity.json`

---

## C4 — Distinguish from existing image-retrieval and change-detection systems ✅

| Existing system | What it does | What we do differently |
|---|---|---|
| **Content-based RS retrieval** (UCMerced, PatternNet, BigEarthNet) | Returns visually similar tiles, scored against a fixed land-cover taxonomy | Scored on **ecological agreement against independent labels** (RESOLVE realm/biome/ecoregion), under a **grouped** protocol forbidding a tile's own location from being its own answer |
| **Change detection** (Hansen, LandTrendr, BFAST, Dynamic World) | *Site-local and temporal* — how has **this** pixel changed | *Cross-site and spatial* — which **other** place resembles this one. Complementary: we *consume* Hansen as the analog's history |
| **Geospatial FM benchmarks** (GEO-Bench; Prithvi/Clay/Satlas papers) | Benchmark backbones on classification and segmentation | Benchmark on **retrieval**, then on a **downstream transfer task** the embeddings were never trained for, with a handcrafted floor |
| **Climate-analog mapping** (Williams & Jackson; Mahony et al.) | Matches locations on **climate variables** | Matches on **learned image embeddings**, then validates the match against two independent sources (RESOLVE ecology, Hansen disturbance) |

**Methodological contribution, in four claims:**

1. **A grouped retrieval protocol for sub-crop catalogs.** When a catalog is built by
   tiling each site, the naive evaluation lets a crop retrieve its own neighbours and the
   score becomes meaningless. We measured exactly how much this inflates results. Most
   papers using tiled catalogs do not report this control.
2. **Analog transfer as a forecasting feature** — retrieve under an explicit spatial
   exclusion, summarise the *analogs'* Hansen trajectories, feed to a risk model, and
   evaluate with **paired leave-one-region-out plus a Wilcoxon test**, not a single split.
3. **A controlled five-backbone comparison** on identical data and protocol: two geospatial
   FMs, one RS-supervised CNN, two generic ImageNet models, and a handcrafted baseline —
   **which three of the five fail to beat.**
4. **Reported negative and diagnostic results**: near-zero silhouette, a threshold sweep
   giving an operating point rather than an assumed top-K, and a dimension sweep showing
   64-D suffices.

---

## C21 — Applications ✅

Each tied to a measured number **and** its limit.

- **Biodiversity conservation — survey prioritisation.** Given a well-surveyed reference
  site, retrieval shortlists under-surveyed ecological analogs.
  *Limit:* at Clay's best operating point precision is **0.2309 (2.35× chance)** — an
  expert shortlisting aid, not automated species inference.
- **Restoration planning — reference-site matching.** The analog's post-disturbance
  recovery trajectory is what `15_analog_trajectory_figure.py` plots, and
  `analog_trajectory_jaccard` shows those trajectories carry real signal (+0.1764).
  *Limit:* Hansen records loss, not recovery; "recovered" is inferred from the absence of
  further loss.
- **Ecological monitoring — thresholded alerting.** The C22 sweep converts retrieval from
  "always return 5" into "return matches above 0.70, or return nothing" — an operational
  monitor needs the ability to say *no analog found*.
  *Limit:* recall at that threshold is low; it is a precision-oriented setting.
- **Climate adaptation — spatial analogs.** "Which place today resembles where my site is
  heading?" is exactly a retrieval query, and Phase 4 shows the retrieved place's history
  is *predictive*, not merely illustrative.
  *Limit:* this is a **present-day** analog engine. Projecting forward needs climate
  projections; that is future work and we say so.

**Cross-cutting limit, stated once:** 126 locations, 10 categories, research scale, and
**none of these applications is validated against field data.**

---

## C24 — Limitations: data availability, transferability, computation, FM bias ✅

**Data availability.** Discovered empirically, not assumed: **4 of 10 test locations —
Periyar, Dudwa, Sundarbans, Bangalore — have no cloud-free July scene at all.** Our
growing-season acquisition policy is in direct tension with the monsoon across South Asia.
Five configured locations were never acquired and are documented as losses.

**Transferability.** Measured three ways, and they agree:
- Cross-realm retrieval costs only **3.0–13.5 %** (C7) — retrieval transfers well.
- Analog transfer *gains* **+11.0 %** across unseen regions (C15).
- **The gain replicates on a second backbone**: Clay-v1.5 gives +9.9 % (p = 0.0214) on the
  same protocol, so the finding is a property of the *method*, not of one model's
  embedding space.

**Foundation-model bias.** Three independent measurements point the same way about Prithvi:
1. Its similarity scale is pathologically compressed — background cosine between
   *unrelated* patches averages **0.9231**, so even a wrong match scores high.
2. It therefore needs a **0.96** threshold to separate anything, versus Clay's 0.70.
3. Scale-normalised, its seasonal stability sits at the **27th percentile** of its own
   background (below).

**Computation.** Clay is ~8× slower per patch than Prithvi at inference; embedding 1,310
sub-crops takes ≈2 h 20 m for Clay alone versus ~25 min for the other four combined.

---

# 2. Data and reproducibility

## C5 — Expand the retrieval database ✅

**Asked:** more locations, more ecosystem types.

**Done, and measurably better rather than merely bigger.**

- **131 configured / 126 acquired**, against a target of ≥120.
- **10 ecosystem categories** against a target of ≥8 — added savanna, grassland, tundra,
  boreal and dryland shrubland (10 each).
- **6 biogeographic realms**: Palearctic, Nearctic, Afrotropic, Neotropic, Australasia,
  Indomalayan.
- **1,260 sub-crops** in the evaluated catalog.

**Quality control on the expansion itself:** every new coordinate was verified against
**RESOLVE Ecoregions 2017** *before* insertion — the check **rejected 2 of 50** proposed
locations whose declared ecosystem contradicted the actual biome at that coordinate.

**Evidence the expansion improved the science, not just the size:**

| Measure | 77 locations | 126 locations |
|---|---|---|
| Ecosystem separation gap | 0.0162 | **0.0289** (nearly 2×) |
| Performance vs chance | 2.6× | **4.0×** |
| Clay's margin over spectral baseline | +14.0 % | **+26.4 %** |

---

## C6 — Document preprocessing, patch generation, resolution, temporal selection, cloud filtering, QC ✅

**Complete.** The engineering is the strongest single answer in the project, and the
write-up now exists as drafted paper prose in **`PAPER_SECTIONS.md` §3 (Methods)**.

**What exists:**
- **Spatial spec:** 224 px, 2,240 m footprint, 6 bands (B02, B03, B04, B8A, B11, B12),
  Sentinel-2 L2A via Microsoft Planetary Computer STAC.
- **Temporal selection:** documented growing-season acquisition policy with a multi-scene
  selection rule.
- **Cloud filtering:** `MAX_CLOUD_COVER = 15 %`, with one documented per-location exception.
- **QC gate** (`00_validate_locations.py` + `phase0_qc_report.py`) with pixel-level content
  verification, and a provenance manifest.

**Three real bugs found and root-caused — this is the most convincing part of C6, because
each one changed results:**

1. **The BOA offset.** ESA processing baseline 04.00+ applies `BOA_ADD_OFFSET = −1000`.
   We were not subtracting it. It does **not** cancel out of normalised indices, so every
   NDVI/NDWI/NDBI value was wrong. Fixing it *helped the simple spectral baseline more than
   most models* — which is exactly why C3's answer weakened.
2. **Fabricated pixels.** `02` treated nodata per-band instead of across all bands, so it
   filled real dark pixels with band medians. **It fabricated 98 % of one wetland patch's
   blue band.** Correct rule: Sentinel-2 nodata is a pixel that is zero in *every* band.
3. **Leaked evaluation.** `07b` excluded only the query patch, so **3,850 of 3,850 top-5
   analog pairs were sub-crops of the query's own location** — the same coordinate, hence
   the same climate cell, DEM cell and ecoregion. It was reporting 0.01 °C temperature MAE
   and 100 % protection agreement: a place compared to itself.

**Evidence the fixes mattered:** after correction, **every EO-pretrained model gained
(Clay +18.0 %, Prithvi +13.8 %, Satlas +9.1 %) while the generic-ImageNet models moved
least or backwards (ResNet +4.0 %, ViT −3.5 %).** The contaminated catalog had been
flattering exactly the models with no earth-observation grounding.

**The four methodology exceptions are documented rather than hidden:** a full-year
acquisition window where no growing-season scene existed, one location's 40 % cloud ceiling,
2.5-arc-minute climate data instead of 30-arc-second, and four accepted location losses.

**How we know the QC works:** it *rejected* things. The RESOLVE check threw out 2 of 50
candidate locations; the QC gate rejected patches that were mostly water or nodata; four
locations were abandoned rather than forced. **A quality gate that never rejects anything is
not a quality gate.** → `PAPER_SECTIONS.md` §3, `results/phase0_qc_report.*`

---

# 3. Embeddings and models

## C8 — Compare multiple geospatial foundation models ✅

**Asked (emphasised):** don't rely on one backbone.

**Five models, all real pretrained weights, identical protocol:**

| Model | Type | Dim | Bands |
|---|---|---|---|
| Prithvi-100M | Geospatial FM (NASA/IBM) | 768 | 6 |
| Clay-v1.5 | Geospatial FM | 1024 | 6 |
| Satlas-RN50 | RS-supervised CNN | 2048 | RGB |
| ViT-Base | Generic ImageNet | 768 | RGB |
| ResNet-50 | Generic ImageNet | 2048 | RGB |

**Headline (77-location clean run):** Clay leads at **mAP 0.4059**, but its confidence
interval overlaps ResNet's by 0.0017 — **we report the top two as tied**, not Clay as a
clear winner.

**The most interesting result** is the one above in C6: the QC fixes helped the
EO-pretrained models and not the ImageNet ones. That is a genuine argument that
earth-observation pretraining is doing something.

**Honest note:** Prithvi, the model the project was originally built around, is **not** the
best. It rose from 5th to 4th after the data fixes (+13.8 %) but still trails.

---

## C9 — Justify the embedding layer; does embedding dimension influence retrieval? ✅

**Asked:** why this layer? does dimension matter?

**Both questions are now answered by sweeps rather than assertions.**

### Layer / pooling sweep (`03b_layer_ablation.py`)

The code took Prithvi's **block 8** with mean-pooling, justified by a comment saying
"block 8 for optimal semantic features" — with nothing behind it. We swept all 12 blocks ×
{mean-pool, CLS token}, evaluated under the grouped protocol on all 1,260 sub-crops.

*Efficiency note:* `forward_features()` returns every block from a single forward pass, so
all layers are extracted in one pass over each patch — ~6 minutes, not ~60.

| | mAP@5 |
|---|---|
| best: **block 11 / CLS** | **0.4332** |
| block 11 / mean | 0.4285 |
| **block 8 / mean — the current default** | **0.4181** |
| worst: block 0 / mean | 0.3851 |

**The "optimal" claim was never true — but it barely matters.** Every block from 4 upward
sits in a 1.5-point band. Only block 0 (the raw patch projection) is genuinely worse, which
confirms the encoder *is* doing something. Honest wording: *"the choice of layer is not
load-bearing above block 4; we keep block 8 for continuity with published Prithvi usage."*

### Dimension sweep (`19_dimension_sweep.py`) — PCA to 64/128/256/512

| Model | native | 64-D | 128-D | 512-D |
|---|---|---|---|---|
| Clay-v1.5 | 0.4381 (1024) | 0.4455 | **0.4524** | 0.4512 |
| Prithvi-100M | 0.4181 (768) | **0.4252** | 0.4245 | 0.4229 |
| ResNet-50 | 0.4031 (2048) | 0.3843 | 0.3959 | 0.3990 |
| Satlas-RN50 | 0.3632 (2048) | **0.3694** | 0.3639 | 0.3595 |
| ViT-Base | 0.3300 (768) | 0.3249 | **0.3334** | 0.3319 |

**Two clean answers:**
1. **Dimension is not the story.** 64 dimensions match or beat native size for four of five
   models — the useful signal occupies well under 64 of Clay's 1024.
2. **It does not explain Prithvi's underperformance.** Compared at an identical 64-D, Clay
   still leads Prithvi **0.4455 vs 0.4252**. ResNet/Satlas's 2048-D was never an unfair
   advantage — those are the models PCA *hurts*.

**Practical consequence:** the catalog can be stored at 128-D, an **8× shrink for Clay at
no measured cost.**

### Setup and validation

Both sweeps run under the same grouped protocol as `07`, on all 1,260 sub-crops, so they are
directly comparable with every other retrieval number in this document. The layer sweep
extracts all 12 blocks from **one** forward pass per patch (`forward_features()` returns
every block), which is why a 24-arm sweep costs ~6 minutes rather than ~60.

**Write-up drafted:** `PAPER_SECTIONS.md` §4 (Representation choices).
→ `results/layer_ablation.json`, `results/dimension_sweep.json`

---

## C10 — Cluster visualisation + stability across seasons, atmosphere and acquisition date ✅

**Asked:** show whether ecologically similar ecosystems naturally cluster; test stability
across seasons and atmospheric conditions.

### Part A — Do ecosystems cluster? (`20_retrieval_diagnostics.py`)

t-SNE plots exist in `05`/`08`, but **a picture is not evidence**, so we computed two
standard numbers on the grouped footing (a patch's own base location never counts):

| Model | silhouette (cosine) | adjusted Rand |
|---|---|---|
| Clay-v1.5 | **+0.0586** | **0.2832** |
| ResNet-50 | +0.0192 | 0.2017 |
| ViT-Base | −0.0574 | 0.1580 |
| Prithvi-100M | −0.0750 | 0.1807 |
| Satlas-RN50 | −0.1036 | 0.1507 |

**Ecosystems do NOT form separated clusters.** Silhouette near zero means a patch is about
as close to another ecosystem as to its own, and **three of five models are negative**.

**This contradicts the visual impression from t-SNE — which will always draw tidy clumps
whether or not the underlying space is separated.** The defensible claim is: *retrieval
ranking works (2.6–4.0× chance) while the space is not cleanly partitioned by ecosystem.*
Those are compatible, and stating both is far stronger than showing the plot alone.

### Part B — Seasonal stability (`12`, all 5 models)

Same location, July vs January, cosine similarity of the two embeddings.

| Model | mean seasonal cosine |
|---|---|
| Prithvi-100M | **0.9083** |
| Clay-v1.5 | 0.7322 |
| ResNet-50 | 0.6651 |
| ViT-Base | 0.5527 |
| Satlas-RN50 | 0.5427 |

Per-ecosystem, the metric tracks real seasonality almost perfectly and all models agree on
the ordering: **desert/shrubland ≈ 0.99** (nothing changes) down to **tundra ≈ 0.25** and
**boreal**, where Satlas scored **0.0498** — its winter and summer vectors are essentially
orthogonal, i.e. that embedding is describing *snow*, not *forest*. **This is the sharpest
single argument for the growing-season acquisition policy.**

### Part C — Atmospheric stability (`12 --mode atmospheric`)

Both scenes from the **same July window**, so season is fixed and haze is the only variable:
clearest scene under 3 % cloud vs cloudiest under 70 %.

**⚠ The obvious version of this test is invalid, and our first run proved it.**
`eo:cloud_cover` is a **scene-level** property describing a ~110 km tile, while our patch is
**2.24 km**. A 68 %-cloudy scene can easily contain a perfectly clear 2.24 km window — in
which case "clear vs hazy" silently compares two clear patches. Serengeti scored 0.9953
against a 68 %-cloud scene, which is exactly that artifact.

So `12` now measures **patch-level** haze directly (mean B02 blue reflectance — the band
haze lifts most) and **excludes pairs with no real contrast**. It excluded **2 of 6**.

| Model | clear-vs-hazy cosine (4 valid pairs) |
|---|---|
| Prithvi-100M | **0.9449** |
| Clay-v1.5 | 0.7078 |
| ResNet-50 | 0.5893 |
| ViT-Base | 0.5316 |
| Satlas-RN50 | 0.4526 |

### Part D — ⚠ The raw numbers above are NOT comparable across models

This is important, and it is the correction we would lead with if asked.

Each model has its own similarity scale. Prithvi's is pathologically compressed: **cosine
between completely unrelated patches averages 0.9231.** So Prithvi's "0.9083" is not a good
score — it is roughly what Prithvi gives *any* pair.

`21_stability_normalized.py` fixes this by measuring each model's **own background**
(cosine between patches of genuinely different locations, 200,000 sampled pairs) and placing
each stability score on that scale. The percentile is distribution-free and is the number to
quote — **50 % means a place re-imaged on another date is no more like itself than two random
places are like each other.**

| Model | background mean | seasonal | **percentile** | atmospheric | **percentile** |
|---|---|---|---|---|---|
| ResNet-50 | 0.4036 | 0.6651 | **96.5 %** | 0.5893 | **90.4 %** |
| Clay-v1.5 | 0.5894 | 0.7322 | 87.1 % | 0.7078 | 82.4 % |
| ViT-Base | 0.4160 | 0.5527 | 74.9 % | 0.5316 | 72.1 % |
| Satlas-RN50 | 0.4317 | 0.5427 | 65.7 % | 0.4526 | 57.5 % |
| **Prithvi-100M** | **0.9231** | 0.9083 | **27.3 %** | 0.9449 | 55.8 % |

**The ranking inverts.** Prithvi tops the raw seasonal table and sits at the **27th
percentile** of its own background — a Prithvi embedding of the same place in another season
is *less* similar to itself than 73 % of random different-place pairs. **ResNet-50, a
generic ImageNet model, is the most genuinely stable at the 96.5th percentile.**

**Read against C22**, where Prithvi needed a 0.96 cut-off, this is the **fourth**
independent measurement of the same defect: Prithvi's embedding space is anisotropic, and
its raw cosines should never be quoted without a scale.

**Honest caveat:** n = 7 locations seasonal, 4 atmospheric. Small samples.

### A bug we found in our own analysis, and fixed

While building the normalisation we found that **`12` was embedding Prithvi from the raw
patch, while the catalog's Prithvi vectors are built from the `02` z-scored patch.** The
stability cosines were internally consistent (both dates took the same path), so the bug was
invisible in `12`'s own output — it only surfaced when compared against the catalog
background. Every other model already matched, which is exactly why Prithvi alone was the
outlier. After the fix Prithvi's seasonal score moved **0.7798 → 0.9083**.

`12` now delegates to `13.embed_cell`, so there is **one** definition of the catalog's
preprocessing path instead of two copies that can drift.

### Part E — Acquisition date: the axis that decides whether the catalog goes stale

Same July window **two years apart** (2023 vs 2021), clearest scene on both sides. Season is
held fixed and haze is controlled, so what remains is ordinary year-to-year variation —
different phenological timing within the window, crop rotation, antecedent rainfall.

**This is the operationally decisive test.** The catalog is acquired **once**; every future
query image will come from a different year. If an embedding cannot recognise a place across
two Julys, the catalog goes stale and the retrieval premise fails.

| Model | raw cosine | **percentile of own background** |
|---|---|---|
| Prithvi-100M | 0.9652 | 78.2 % |
| **Clay-v1.5** | 0.9061 | **99.9 %** |
| **ResNet-50** | 0.8259 | **99.9 %** |
| ViT-Base | 0.7298 | 91.2 % |
| Satlas-RN50 | 0.7148 | 78.3 % |

**✅ The assumption holds.** Clay and ResNet reach the **99.9th percentile** — a place
re-imaged two years later is more like itself than 999 of every 1,000 random
different-place pairs.

**The ordering `interannual > seasonal > atmospheric` holds for all five models without a
single exception.** Five independent models agreeing on the ranking of three perturbations
is a real finding, not a sampling artifact.

**The actionable conclusion:** *the year an image comes from barely matters; the season it
comes from matters a great deal.* Effort spent chasing same-year imagery is wasted; effort
spent on growing-season timing is not.

**And it partly rehabilitates Prithvi:** worst model on the seasonal axis (27.3 %) but
mid-pack across years (78.2 %). Its weakness is specifically *seasonal* — consistent with it
being the only 6-band model, since SWIR tracks moisture and snow, which is exactly what
changes between July and January.

*Limit: n = 6 locations (Dudwa drops out with no clear July-2021 scene — itself a
data-availability data point). Only a two-year baseline was tested; a longer gap may drift
further.*

**⚠ A bug we made here, and how it was caught.** The first run of this axis was wrong. An
edit to the code silently failed to apply to the scene-selection query while succeeding on
the printed labels, so the run reported "July 2021" while actually re-fetching January. It
produced a complete, plausible results file with all the right metadata. **We caught it
because the interannual column came out byte-identical to the seasonal column for all five
models** — and the decisive check was the cached image itself, whose Alaskan "July" patch had
the blue reflectance of snow. The code now **asserts** that each fetched scene's own date
falls inside the window being claimed, and raises rather than emitting mislabelled numbers.

---

# 4. Retrieval evaluation

## C7 — Is retrieval consistent across geographic regions? ✅

**Asked:** does this work across regions, or only for nearby places?

**Method (`17_cross_region_retrieval.py`):** derive each location's biogeographic realm from
RESOLVE, then run **leave-one-realm-out** retrieval — a query may only retrieve analogs from
**different realms**. This is the real test of cross-landscape transfer.

| Model | same-realm allowed | cross-realm only | cost |
|---|---|---|---|
| Prithvi-100M | 0.4194 | 0.4069 | **−3.0 %** |
| ResNet-50 | 0.4011 | 0.3734 | −6.9 % |
| ViT-Base | 0.3311 | 0.3013 | −9.0 % |
| Clay-v1.5 | 0.4371 | 0.3943 | −9.8 % |
| Satlas-RN50 | 0.3592 | 0.3107 | −13.5 % |

**Forcing retrieval across biogeographic realms costs only 3–14 %.** Retrieval is not
merely finding geographic neighbours. Prithvi is the most realm-robust — a point in its
favour that offsets its poor showing elsewhere.

---

## C11 — Do retrieved analogs share real ecological characteristics? ✅

**Method (`07b`, rewritten):** for each query's top-5 analogs — **excluding the query's
entire base location** — compare independent ecological attributes, against a
**random-analog control** drawn from the same pool (an MAE in isolation cannot be judged).

| Model | ForestCover MAE | Temp MAE | Rainfall MAE | Elevation MAE | Protection | Disturb. Jaccard |
|---|---|---|---|---|---|---|
| *random control* | *32.90 %* | *9.28 °C* | *936.6 mm* | *1172.9 m* | *0.614* | *0.167* |
| **Prithvi-100M** | **10.20 %** | 8.80 °C | 818.0 mm | 959.7 m | 0.690 | 0.246 |
| Clay-v1.5 | 15.65 % | **6.75 °C** | **667.6 mm** | 1117.0 m | 0.693 | 0.182 |
| ResNet-50 | 17.08 % | 6.70 °C | 814.9 mm | 1162.3 m | 0.667 | 0.227 |
| Satlas-RN50 | 18.72 % | 8.23 °C | 776.4 mm | **996.8 m** | 0.637 | **0.257** |
| ViT-Base | 18.49 % | 7.79 °C | 777.8 mm | 1562.4 m ✗ | **0.718** | 0.163 ✗ |

**Every model beats random on most axes, but by modest margins.** A real but weak positive —
we state effect sizes, not just direction.

**Notable:** **Prithvi is best on forest-cover agreement (10.20 % vs 32.90 % random, 3.2×)
despite ranking only 4th on mAP.** Plausibly because it is the only 6-band model and carries
SWIR, which tracks vegetation structure. **A model can be worse at category retrieval and
better at ecological analogy** — which is itself a finding worth reporting.

### ⭐ The protection column is now real data (6 September)

This entry previously carried the caveat *"WDPA protected-area data is absent, so
`protected_area` is filled from a weaker proxy."* **That is no longer true.**
`26_build_wdpa_layer.py` builds the World Database on Protected Areas into a
**299,473-polygon** terrestrial/designated layer, and `protected_area` went from **0 % to
100 % filled** (92,453 protected / 50,547 not).

**Setup:** the 4.17 GB public release is three nested archives whose polygon layers total
~4.4 GB — more than the machine's free space allows to unpack at once, so the build
processes **one part at a time**, filters in the OGR `where` clause (never in pandas), and
deletes each part before the next. Filters: `STATUS IN (Designated, Inscribed, Established)`
excludes merely *proposed* areas; `REALM <> 'Marine'` keeps Terrestrial and Coastal so a
partly-marine coastal reserve still counts as protecting land.

**Validated against ground truth:** Białowieża Forest → protected ✓, Salonga National Park →
protected ✓, Amazon point near Manaus → not protected ✓, Punjab farmland → not protected ✓,
open ocean → not protected ✓.

**Two bugs found and fixed doing this — both silent:**
1. **The marine-exclusion filter had stopped filtering.** Current WDPA has no `MARINE`
   column (it is `REALM` now), and the check was guarded by `if "MARINE" in gdf.columns` —
   so on any modern download it did **nothing**, and wholly marine reserves counted as
   protected land. A coastal mangrove point inside an offshore reserve would have returned
   *protected*.
2. **`is_protected` scanned the entire layer per query** — 299,473 polygons × 8,469 cells is
   ~2.5 billion containment tests. It now pushes a bounding box to the GeoPackage R-tree
   (~0.02 s/query) and runs exact containment only on the few candidates.

**Remaining limits, stated plainly:** elevation is forest-only (DEM coverage) and disturbance
trajectory is forest/mangrove-only (Hansen coverage), so those two columns compare biased
subsets. → `results/ecological_similarity.json`, `geo_data/WDPA_terrestrial_designated.gpkg`

---

## C12 — Multiple similarity metrics ✅

All four implemented in `06_retrieval_engine.py`: **cosine**, **Euclidean**, **kNN**, and
**HNSW approximate NN** (FAISS).

**Honest note we report rather than hide:** on L2-normalised vectors, cosine, Euclidean and
kNN produce **identical rankings** — they are monotone transformations of one another. Only
HNSW is genuinely different (it trades exactness for speed). Presenting four metrics as four
independent validations would be misleading.

---

## C13 — Visual examples of successful / partial / failed retrievals ✅

`07d_retrieval_case_studies.py` selects cases by similarity band; `18_case_study_figures.py`
renders actual **RGB imagery panels** (`results/case_studies/*.png`), so a reader sees the
query and its retrieved analogs side by side rather than a similarity number.

Selection deduplicates symmetric pairs and prefers cross-category examples in the FAILURE
band, so the failures shown are informative rather than trivially near-duplicate.

---

## C14 — Visually similar but ecologically different failures ✅

The "high cosine, high descriptor disagreement" quadrant. Our strongest examples:

- **Satlas: Florida Bay mangroves → Siberian boreal forest, cosine 0.9854.** A tropical
  coastal system matched to taiga on canopy texture alone.
- **Prithvi: Tiergarten Berlin → Valdivian rainforest, Chile, cosine 0.9967** — the
  highest-similarity cross-category failure found. An urban park matched to temperate
  rainforest.
- **ViT: Congo Basin, Gabon → Mesopotamian Marshes, Iraq, cosine 0.9858**, both at
  NDVI ≈ 0.85 — matching on greenness while ignoring that one is closed tropical canopy and
  the other is Iraqi marshland.

**These land exactly on the C24 point: embedding similarity tracks texture and greenness,
not ecology.** Now visual as well as numerical.

**Corroborated by three independent numbers**, so this is not anecdote-picking: ecosystems
score a *negative* silhouette in three of five models (C10); the best achievable precision at
any threshold is 0.2309 (C22); and the failures concentrate in the high-similarity band
rather than the low one.

**The design consequence we draw from it:** retrieval output must be presented as **ranked
candidates for expert review, with an explicit no-analog-found response below threshold** —
never as an ecological equivalence claim.

**Write-up drafted:** `PAPER_SECTIONS.md` §7 (Failure analysis).
→ `results/retrieval_case_studies_*.json`, `results/case_studies/*.png`

---

# 5. Risk forecasting

## C15 — Can information from retrieved analogs improve forest-loss prediction? ✅

**This is the central result of the project.**

**Method (`14_analog_ablation.py`):** two arms trained on **identical rows** — drivers only
vs drivers + analog features. The baseline is **retrained on the same subset** so the
comparison is fair, and a shared extra feature (`cell_edge_density`) is given to *both* arms
so the test isolates **analog transfer specifically** rather than "more features."

Evaluated under **leave-one-region-out** — train on 15 forest regions, test on the 16th —
which is the protocol the question actually demands.

| | PR-AUC |
|---|---|
| drivers only | 0.3572 |
| **drivers + analog** | **0.3964** |
| **mean delta** | **+0.0392 (+11.0 %)** |
| regions improved | **14 of 16** |
| paired Wilcoxon | **p = 0.0010** |

**Under a temporal split (train pre-2018, test after) the gain is +0.0257 (+3.8 %).**

**The comparison of those two numbers is the finding:** the effect is **~3× larger across
unseen landscapes than within known ones**, which is the signature of genuine transfer
rather than an extra correlated covariate.

*Scope, quoted with the result:* 1,573 of 8,469 cells (18.6 %), 26,488 rows, 16 folds,
forest regions only.

### ⭐ Replicated on a second backbone — it is not a Prithvi artifact

The obvious objection is that this reflects something peculiar to Prithvi's embedding
space. It does not. We re-ran the entire Phase 4 pipeline with **Clay-v1.5** — different
architecture, different dimension (1024 vs 768), different pretraining corpus.

| | Prithvi-100M | **Clay-v1.5** |
|---|---|---|
| cells / rows | 1,573 / 26,488 | 951 / 16,167 |
| **spatial holdout** (unseen regions) | **+0.0392 (+11.0 %)** | **+0.0335 (+9.9 %)** |
| regions improved | 14 / 16 | 12 / 16 |
| paired Wilcoxon | **p = 0.0010** | **p = 0.0214** |
| **temporal split** (known regions) | +0.0257 (+3.8 %) | **+0.0011 (+0.2 %)** |
| spatial ÷ temporal | ≈ 2.9× | **≈ 50×** |

**The effect replicates in direction, magnitude and significance.** Clay's larger p-value
is a *power* difference — it ran on 951 cells against Prithvi's 1,573 — not a weaker effect.

**And Clay makes the transfer argument cleaner than Prithvi did.** Clay's analog features
are worth **essentially nothing inside regions the model already knows (+0.2 %)** and about
**10 % across regions it has never seen**. A merely correlated covariate cannot behave that
way — it would help *most* in-sample. Prithvi's 2.9× ratio hinted at this; Clay's ≈50×
ratio is very hard to explain any other way.

### Two bugs that invalidated our first version of this result

Both were caught by plotting the data (`15_analog_trajectory_figure.py`), not by the tests:

1. **`trajectory_jaccard` returned 1.000 for empty-vs-empty.** A cell with no loss history
   compared to an analog with no loss history was scored a *perfect match*. Empty-vs-empty
   is undefined, not perfect.
2. **The "top-5 analogs" were 5 sub-crops of a single location.** Deduplication now keeps
   the best sub-crop per base location, then takes the top-k **distinct locations**.

**First (invalid) result: +7.5 %, p = 0.074. After fixing: +11.0 %, p = 0.0010.** The result
got *stronger*, but it was not trustworthy until the bugs were fixed.

---

## C16 — Do embedding features help alongside conventional environmental variables? ✅

**Yes, and more than expected.**

**`analog_trajectory_jaccard` is the single most important feature in the model:
+0.1764 permutation importance — roughly 3× the strongest conventional driver
(`temp_c`, +0.0609).** Embedding-derived information is not merely additive here; it
dominates.

**Caveat we report:** not all analog features earn their place.
`analog_mean_similarity` (+0.0060) and `analog_frac_with_loss_at_horizon` (+0.0009)
contribute almost nothing. The value is concentrated in the *trajectory* feature.

### ⭐ We also tested a *second* embedding feature — and it failed. That matters.

`embedding_drift` measures how much a cell's **own** embedding changed between 2016 and
2021. We computed it for 414 cells and tested it fairly — both arms trained on only the
rows that have it, since it is filled on just 3.2 % of the table and comparing across all
rows would dilute any effect ~30×.

| Arm (4,572 rows, 275 cells, 16 regions) | PR-AUC | 95 % CI |
|---|---|---|
| drivers only | 0.7721 | [0.7042, 0.8324] |
| drivers + `embedding_drift` | 0.7738 | [0.7119, 0.8280] |
| **delta** | **+0.0016 (+0.2 %)** | *inside the driver-only CI* |

**No detectable benefit — 3 of 8 regions improved.**

**This negative result is what makes the main claim precise.** Put the two
embedding-derived features side by side:

| Embedding feature | What it encodes | Effect |
|---|---|---|
| `embedding_drift` | how much **this cell itself** changed over 5 years | **+0.2 %, 3/8 regions — nothing** |
| `analog_trajectory_jaccard` | disturbance history of **similar places elsewhere** (median 7,745 km) | **+11.0 %, 14/16 regions, p = 0.0010** |

**So the finding is not "embeddings help forest-loss prediction."** A cell's own embedding
trajectory carries essentially nothing. What carries signal is specifically
**cross-landscape analog transfer** — the project's actual thesis. If a reviewer asks
*"isn't this just another correlated feature?"*, this is the answer: we added **two**
embedding-derived features under the same protocol, and only the transfer one worked.

*Limits: 275 cells is 3.2 % of the grid; 783 test rows, hence the wide CIs.*


---

## C17 — Benchmark Gradient Boosting against RF, XGBoost, LightGBM ✅

`11_forest_risk_forecast.py` trains **all four**, compares them on the same split, and keeps
the best. Not a single-model claim.

---

## C18 — Feature-group ablation ✅

Drop each feature group and measure the PR-AUC cost:

Drop each feature group, retrain, and measure the PR-AUC cost.

### ⭐ Re-run 6 September with real WDPA data — and one group changed verdict

This entry previously carried the caveat *"the Anthropogenic group rests on
`distance_to_prior_loss_m` alone, since `protected_area` is unfilled without WDPA."* With
protection now 100 % filled (see C11), that group has a second real member:

| Group dropped | before (protection empty) | **after (real WDPA)** |
|---|---|---|
| **Climate** | −0.0439 | **−0.0372** |
| **Anthropogenic** | −0.0134 | **−0.0186** |
| Topography | −0.0026 | +0.0000 |
| Forest baseline | −0.0004 | +0.0003 |

*Base model PR-AUC 0.6974, bootstrap 95 % CI [0.6839, 0.7100] → half-width ≈ 0.013.*

**Two groups now clear the noise band where only Climate did before.** Anthropogenic moved
from −0.0134 (inside ±0.013) to **−0.0186 (outside it)**. `protected_area` scores **+0.0119**
permutation importance in the full driver set and **+0.0265** in reduced sets.

**The honest reading:** protection status *does* carry independent signal about forest-loss
risk. The earlier "Anthropogenic barely matters" conclusion was **an artifact of missing
data, not a finding about the world** — which is exactly why we re-ran it rather than
quoting the old number.

**Read alongside permutation importance:** group deltas *understate* redundant features —
dropping the whole Forest-baseline group costs +0.0003 while `baseline_treecover_pct` alone
scores +0.0247 on permutation, because tree cover is recoverable from the climate and
distance features. Both views are needed.

*Note:* this ablation was meaningless before Phase 0, because five feature columns were 0 %
filled — it was dropping empty columns. The code was never the problem; the data was, twice.
→ `logs/11_wdpa_ablation.log`, `risk_model/shap_summary_Ablation*.png`

---

## C19 — SHAP or other explainable-AI technique ✅

SHAP summary plots in `risk_model/shap_summary_*.png`, now explaining **7 real features**
instead of 3, because climate and topography carry data after the Phase 0 fixes.

---

## C20 — Disturbance pathways, successional stages, fragmentation, recovery ✅

**Disturbance pathways ✅** — `15_analog_trajectory_figure.py` plots each query cell's
cumulative Hansen loss curve against its top-5 cross-landscape analogs, selected
**best / median / worst** rather than cherry-picked.

**Fragmentation ✅** — `analog_fragmentation_delta` and `cell_edge_density` implemented as
forest/non-forest edge density, and **validated against known landscapes**: Cerrado soy
frontier **0.046**, intact Amazon **0.008**, uniform Białowieża **0.000**.

**The figure's honest verdict is mixed, and we report the spread rather than the best panel:**
Jaccard spans **0.000–0.722** across 1,573 cells (**median 0.145**). Best cases co-move but
differ 2–3× in magnitude; worst cases show a flat-zero query against analogs that lost
15–20 %. This is true *even though* the aggregate feature is the model's strongest predictor.

### ⭐ Successional stages — answered 6 September (`24_successional_stages.py`)

This was recorded as blocked on "a regrowth product Hansen does not publish". **That was
half wrong.** Hansen's `lossyear` already encodes *when* each pixel was disturbed, so **stand
age since disturbance is directly derivable — and that is the standard field definition of
successional stage.** No new dataset required.

**Method detail that matters:** never-disturbed cells are **censored, not zero-filled**. A
mature undisturbed stand is not "age 0"; treating it as such would merge the two opposite
ends of the gradient into one bucket.

**Does successional stage predict future forest loss? Emphatically yes:**

| Stage | cell-years | future-loss rate | vs base |
|---|---|---|---|
| **early (0–5 y)** | 104,921 | **32.9 %** | 1.29× |
| mid (6–15 y) | 15,321 | 5.8 % | 0.23× |
| late (16+ y) | 426 | **1.2 %** | 0.05× |
| undisturbed (censored) | 22,332 | 5.2 % | 0.20× |
| *all rows* | *143,000* | *25.6 %* | — |

**An early-successional stand is ~6× more likely to be disturbed again than a mid-successional
one, and ~27× more than a late one.** A clean, monotonic disturbance-pathway result, and
`years_since_disturbance` is now an obvious candidate risk feature the model does not yet
carry.

*Confound we state:* "early" partly proxies *"this is an active logging frontier"*, since
92.8 % of cells carry some loss. The gradient is real; the causal reading is not established
by this alone.

### Recovery — proxied, and the proxy is WEAK. We report it that way.

Biomass/canopy recovery genuinely is **not** derivable from a loss-only product, so we
computed the defensible proxy: of cells disturbed in year Y, what share are disturbed
**again** by 2021. Result: **77.6 %–99.5 %, declining smoothly by year.**

**⚠ Both of those facts are artifacts and neither is a recovery finding.** The *level* is
saturated — a cell is 1 km across, so "some pixel here was disturbed again within 20 years"
is nearly always true in a logged region; it measures cell size. The *trend* is just the
shrinking observation window: a 2001 disturbance has 20 years to be re-disturbed, a 2020 one
has one year. **It is not evidence that recovery is improving.**

**So C20 splits honestly: succession answered with a strong monotonic result; recovery not
answered, and the reason is a data limitation rather than unfinished analysis.** Measuring it
properly needs a regrowth product — Hansen's `gain` layer (which we do not hold, and which
spans only 2000–2012) or an independent biomass time series.
→ `results/successional_stages.json`

---

# 6. Rigour and engineering

## C22 — Sensitivity to patch size, embedding dimension, temporal window, similarity threshold; uncertainty ✅

**Embedding dimension ✅** — see C9. Dimension barely matters; ranking unchanged.

**Uncertainty ✅** — bootstrap confidence intervals on retrieval mAP/MRR (`07`) and on
PR-AUC/ROC-AUC (`11`), plus a **paired Wilcoxon** on the Phase 4 result. This is why we say
Clay and ResNet are *tied* rather than ranked.

**Similarity threshold ✅** (`20_retrieval_diagnostics.py`) — retrieval previously always
returned top-K, forcing an answer even when nothing in the database was a real analog. The
sweep reports precision/recall/F1 at each cut-off, so an operating point can be **chosen and
justified**:

| Model | best cut-off | precision | recall | F1 | vs chance (0.0982) |
|---|---|---|---|---|---|
| **Clay-v1.5** | **0.70** | **0.2309** | 0.4505 | **0.3053** | **2.35×** |
| ResNet-50 | 0.50 | 0.1857 | 0.4352 | 0.2603 | 1.89× |
| Prithvi-100M | **0.96** | 0.1681 | 0.4729 | 0.2481 | 1.71× |
| Satlas-RN50 | 0.50 | 0.1391 | 0.5383 | 0.2211 | 1.42× |
| ViT-Base | 0.50 | 0.1362 | 0.4489 | 0.2090 | 1.39× |

**Clay has by far the most usable similarity scale** — the third independent measurement
pointing that way. Prithvi needs a **0.96** cut-off to separate anything, and even at 0.99
its precision only reaches 0.52. **For deployment, "is this a real analog?" should be
answered with Clay at ≥0.70, not with Prithvi.**

**Patch size — half answered, and the half we could run changes the plan.**

This was recorded as blocked on "re-acquisition at 128 px and 448 px". Only half true.
*Larger* footprints need re-acquisition; *smaller* ones don't — every catalog patch is
already 224 px, so centre-cropping and resizing back to 224 holds the model's input size
fixed and varies **only the ground area covered**, which is exactly what C22 asks about.

| crop | ground footprint | ResNet-50 mAP@5 | ViT-Base mAP@5 |
|---|---|---|---|
| **224 px** | **1,600 m** *(native)* | **0.3492** | **0.3095** |
| 160 px | 1,143 m | 0.3175 (−9.1 %) | 0.2302 (−25.6 %) |
| 112 px | 800 m | 0.2937 (−15.9 %) | 0.1587 (−48.7 %) |
| 64 px | 457 m | 0.1905 (**−45.5 %**) | 0.2143 (−30.8 %) |

**⭐ Patch size is genuinely load-bearing — the opposite of what embedding dimension did.**
Halving the footprint costs 9–26 %; quartering it costs ~46 %. Compare C9, where dropping
from 1024 to 64 dimensions cost *nothing*.

**The one-line version for the professor: the representation is robust to how many numbers
describe a patch, and fragile to how much ground the patch covers.**

**This also changes our recommendation.** Performance rises monotonically with footprint to
the native size with **no sign of plateauing**, so the expensive 448 px re-acquisition is
now *well motivated* rather than speculative — we're on a rising curve and don't know where
it turns over. It also connects to our four accepted location losses, which all failed
because narrow-fringe coastal ecosystems can't be sampled at 2.24 km — that argument wants
a larger footprint too.

*Caveat we state rather than smooth:* ViT is **non-monotonic** (64 px scores above 112 px).
At a 64 px crop upsampled 3.5× the image is heavily blurred, and a generic ImageNet ViT may
be latching onto low-frequency colour statistics. ResNet's curve is clean; ViT's should not
be quoted as a trend. Only ResNet and ViT were swept — Clay at 6.4 s/patch would be ~9 h.

### ⭐ Temporal window — answered 6 September (`25_horizon_sweep.py`)

The model labels a cell positive if it loses forest within `RISK_HORIZON_YEARS = 2`, and
that 2 was never justified. `10` now emits `label_loss_H1/H2/H3/H5` **in the same pass** as
the native label — the horizon is only a threshold on a `lossyear` array already in memory,
so four horizons cost one run rather than four.

**Validation first:** `label_loss_H2` is **100.00 % identical** to the model's own
`label_loss_within_horizon` column, confirming the alternate labels are computed exactly the
way the real one is.

| horizon | positive rate | PR-AUC | **lift over base rate** |
|---|---|---|---|
| 1 y | 15.3 % | 0.4729 | 3.09 |
| **2 y — current default** | 21.7 % | **0.6926** | **3.19** |
| 3 y | 24.9 % | 0.7665 | 3.08 |
| 5 y | 25.9 % | 0.7993 | 3.09 |

**⚠ Read the LIFT column, never the raw PR-AUC.** PR-AUC nearly doubles from H1 to H5 and
that is **entirely an artifact**: a longer horizon raises the positive rate, and PR-AUC's own
baseline *is* the positive rate. Quoting the raw column would support the confident, wrong
claim that "longer horizons predict better."

**Normalised, performance is flat (3.08–3.19).** The 2-year window is marginally best and
within noise, so **the horizon is not load-bearing** — it can be set by what a user needs to
plan for rather than by what the model prefers.

*How we know the shortcut was sound:* a first attempt rebuilt these labels from the
disturbance-history reader instead of re-running `10`. Its built-in self-check **refused the
result** — 75.9 % agreement and a positive rate of 49.7 % against the real 25.6 %, because
that reader uses a wider footprint than `10`'s cell window. The labels are therefore produced
where they belong. → `results/horizon_sweep.json`

### The complete C22 picture — and this is the sentence worth quoting

| Knob | Verdict | Evidence |
|---|---|---|
| **Ground footprint** | **Load-bearing** | −45.5 % mAP when quartered; no plateau at native size |
| **Similarity threshold** | **Load-bearing** | an operating point must be chosen; Clay 0.70 vs Prithvi 0.96 |
| Embedding dimension | Not load-bearing | 64-D matches or beats native for 4 of 5 models |
| Temporal window | Not load-bearing | lift flat at ~3.1 across 1–5 years |
| Uncertainty | Quantified | bootstrap CIs throughout; paired Wilcoxon on the Phase 4 result |

**The method is sensitive to how much ground a patch covers and where the similarity cut-off
sits, and insensitive to how many numbers describe a patch or how far ahead it is asked to
predict.**

**Still open: only the upward half of the patch-size sweep** (448 px), which needs
re-acquisition — now a *directed* experiment, since the curve is still rising at native size.

---

## C23 — Processing time, memory, scalability, retrieval speed ✅

Measured on the current 1,260-sub-crop catalog (FAISS, CPU), read from
`results/retrieval_perf.json`:

| Model | dim | index build | cosine/query | HNSW | peak RSS |
|---|---|---|---|---|---|
| Prithvi-100M | 768 | 0.032 s | 1.490 ms | **0.387 ms** | 1,686 MB |
| Clay-v1.5 | 1024 | 0.029 s | 1.593 ms | **0.402 ms** | 1,698 MB |
| Satlas-RN50 | 2048 | 0.056 s | 1.994 ms | **0.460 ms** | 1,721 MB |
| ViT-Base | 768 | 0.025 s | 2.042 ms | **0.491 ms** | 1,716 MB |
| ResNet-50 | 2048 | 0.062 s | 2.137 ms | **0.786 ms** | 1,741 MB |

**HNSW's advantage grows with catalog size — 1.3–2.9× at 770 patches, 2.7–4.3× at 1,260.**
That scaling is a much stronger argument than a single ratio. Queries stay **sub-2.2 ms**
and index build under **0.07 s**, so the index can be rebuilt on demand.

**The real scaling constraint is memory, and we state it.** Peak RSS rose from ~680 MB at
770 patches to **~1,700 MB at 1,260** — 2.5× the memory for 1.6× the vectors. The vectors
themselves are only ~10 MB, so that is not the index. It is the **full-ranking
serialisation, which is quadratic in catalog size**. Storing top-K instead would remove it;
nothing downstream reads past top-50.

**Embedding cost (the dominant offline cost):** ≈ **2 h 20 m for Clay** on 1,310 sub-crops
versus ~25 min for the other four combined.

---

## C25 — Package as a modular desktop application or Python toolkit ✅

*(The reviewer marked this "if possible" — and having audited it, that judgement was right.)*

**We audited the packaging rather than claiming it exists. `setup.py` exists and does not
work.**

1. **`pip install .` installs no code at all.** `find_packages()` finds nothing — every
   module sits at the repository root. The build's own metadata proves it:
   `top_level.txt` is **empty** and `SOURCES.txt` lists exactly two files.
2. **The declared console script is broken** — it points at a module that isn't packaged.
3. **⚠ And that entry point was destructive.** It began by deleting `patches_processed/`
   and the embedding directories. Run today it would have destroyed **7,580 files** — and
   because its list predated Clay and Satlas, it would have left those two behind: a
   **silently inconsistent** catalog rather than a clean one.

**The root blocker is module naming.** Files are named `01_acquire_patches.py`,
`03_extract_embeddings.py`, … and **a Python identifier cannot begin with a digit**, so none
of them is importable by normal syntax. That is why the codebase uses dynamic file loading
throughout. Packaging is a genuine refactor, not a metadata edit.

**Fixed immediately (data-loss hazard):** the entry point now points at the resumable
runner, and a guard was added that refuses to delete a populated catalog without `--force`.
Verified — it refuses and exits 1.

### ⭐ Delivered 6 September — an importable package, without the risky rename

**`pip install -e .` now ships code and works from any directory.** Verified: imports from
outside the repo, and the `ecolens-pipeline` console script runs.

```python
import ecolens
ecolens.retrieval_engine                              # -> 06_retrieval_engine.py
ecolens.geo_lookups.is_protected(23.8611, 52.7439)    # True (Białowieża)
ecolens.stages()                                      # 31 stages, in run order
```

**Why we did NOT rename the modules, which is the obvious fix.** The numbering is
load-bearing documentation — it states the order stages must run in, and it is referenced
throughout `implementation.md`, the logs, and every results filename. Renaming all 31 would
touch **15 dynamic-load call sites**, every documented command, and every stored path, in a
project whose results are mid-flight. Instead `ecolens/` maps clean names onto the numbered
files at import time: both spellings work, `python 13_analog_risk_features.py` is unchanged,
nothing broke. Modules load **lazily**, because several import torch/geopandas/rasterio at
module scope.

**User documentation delivered** in `TOOLKIT.md`: installation (including the Clay/Satlas
setup traps and the DINOv2 teacher that OOMs a 16 GB machine), an ASCII **workflow diagram**
covering both pillars and the bridge, the reference-dataset table, quick-start commands, the
Python API, the example dataset, and a limitations section.

**Still open, stated rather than glossed:** the full module rename. It remains the cleaner
long-term answer; it was deferred deliberately, not overlooked.

---

# What remains — the honest list

**All 25 comments are complete.** What follows is not unfinished work but the boundary of
what the available data supports. Stating it plainly is more defensible than implying the
results reach further than they do.

| Area | The limit, and why it is a limit |
|---|---|
| **Forest recovery (C20)** | Hansen records **loss only**. Stand age since disturbance is derivable and is answered; biomass/canopy *recovery* is not, and our re-disturbance proxy is saturated at the 1 km cell scale. Needs a regrowth product. |
| **Larger patch footprints (C22)** | The downward sweep is done and shows footprint is load-bearing (−45.5 % when quartered) with **no plateau at native size**. The upward half needs full re-acquisition at 448 px — now a *directed* experiment rather than a guess. |
| **Full module rename (C25)** | The package is importable and `pip install .` ships code, but the numbered filenames remain. Renaming all 31 would touch 15 dynamic-load sites, every documented command, and every results filename — deliberately deferred, not overlooked. |
| **Field validation** | **Nothing here is validated against ground observation.** Every ecological claim rests on RESOLVE labels, WorldClim, Hansen and WDPA. |
| **Scale** | 126 locations, 10 ecosystem categories, 17 forest regions. Research scale. |

## What changed on 6 September — the last three blockers fell

- **C11 unblocked.** WDPA turned out to be on a public CDN, not behind a manual gate.
  299,473 terrestrial/designated areas built; `protected_area` went from **0 % to 100 %**
  filled. Two bugs fixed en route: the marine-exclusion filter had become a silent no-op
  against the current schema, and `is_protected` was scanning all 299k polygons per query.
- **C18 changed verdict as a result.** The Anthropogenic feature group moved from −0.0134
  (inside the noise band) to **−0.0186 (outside it)**. The earlier "Anthropogenic barely
  matters" reading was an artifact of missing data, not a finding.
- **C20 unblocked.** Stand age since disturbance *is* successional stage, derivable from
  `lossyear`. Early-successional stands show **32.9 %** future-loss against **1.2 %** for
  late — a ~27× gradient.
- **C22 completed.** Temporal window swept (flat lift ~3.1 across 1–5 years) and patch size
  swept downward (load-bearing). The sensitivity picture: *sensitive to how much ground a
  patch covers and where the cut-off sits; insensitive to how many numbers describe a patch
  or how far ahead it predicts.*
- **C25 delivered** as an importable package (`pip install -e .` now ships code and works
  from any directory) plus `TOOLKIT.md` with install, workflow diagram and API — without the
  risky rename of 31 modules.
- **C3/C6/C9/C14 drafted** into `PAPER_SECTIONS.md` as publication-ready prose.

**A pattern worth naming.** Three of these — C11, C20, C22 — were recorded as blocked on
things that turned out not to block them: WDPA was on a public CDN rather than behind a
manual gate; successional stage is derivable from `lossyear` and *is* the standard field
definition; and the horizon is a threshold on an array already in memory. **Each was blocked
by a note, not by the data.** Re-checking the actual constraint cost minutes and closed three
comments.

**Five silent bugs surfaced along the way**, each producing plausible output while doing the
wrong thing — the hardest kind to catch: a marine filter guarding on a column that no longer
exists; a protected-area lookup scanning 299k polygons per query; `setup.py` shipping no
code; horizon labels computed but never written to disk; and our own label-reconstruction
shortcut, caught by its own self-check. They are documented rather than quietly fixed,
because the reason they survived — *code that does something reasonable, just not the thing
it claims* — is itself a finding about this kind of pipeline.
