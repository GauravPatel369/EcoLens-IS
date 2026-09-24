# EcoLens — Draft Paper Sections

Draft prose for the four review comments whose evidence is complete but whose **write-up**
was the outstanding item: **C6** (methods), **C9** (representation choices), **C3**
(motivation vs. a spectral baseline), **C14** (failure analysis).

Every number here is on disk. File pointers are given so each claim can be checked.
Written to be pasted into the paper and edited for house style, not to be read as a report.

---

## §3 Methods — Data acquisition, preprocessing and quality control  *(C6)*

### 3.1 Imagery and patch definition

All imagery is Sentinel-2 Level-2A surface reflectance, accessed through the Microsoft
Planetary Computer STAC API. Each catalog entry is a **224 × 224 pixel** patch at Sentinel-2's
native 10 m ground sampling distance, covering **2,240 m** on a side, in six bands:
B02 (blue), B03 (green), B04 (red), B8A (narrow NIR), B11 and B12 (SWIR). The six-band
selection matches the input specification of the geospatial foundation models under
comparison; the RGB-only baselines consume the first three.

Each acquired patch is then centre-cropped to 160 px and resampled to 224 px, yielding
catalog entries with a **1,600 m** effective footprint. Sub-crops of a common acquisition
share a base-location identifier, which the evaluation protocol (§4) uses to prevent a patch
from retrieving its own neighbours.

### 3.2 Scene selection and cloud filtering

Scenes are filtered to **< 15 % scene-level cloud cover** and, within the qualifying set,
the least-cloudy scene is selected. Acquisition is constrained to each site's **growing
season**, determined per location rather than by a single global window; sites in the
tropics carry `growing_season: true` year-round, since they have no dormant period.

Three documented exceptions were required and are reported rather than smoothed:
a full-year acquisition window for sites with no qualifying growing-season scene; a relaxed
40 % cloud ceiling for one mangrove site; 2.5 arc-minute rather than 30 arc-second climate
data; and four accepted location losses (§3.5).

### 3.3 Radiometric correction

ESA processing baseline 04.00 and later applies a **`BOA_ADD_OFFSET` of −1000** to Level-2A
reflectance. This offset does **not** cancel in normalised band ratios: for indices of the
form (A − B)/(A + B), an additive offset alters the denominator and therefore the index
value. Uncorrected reflectance systematically distorted every NDVI, NDWI and NDBI value in
early runs. The offset is subtracted at read time.

The correction's effect is measurable and, notably, **asymmetric across model families**.
Re-running the full evaluation on corrected data improved every earth-observation-pretrained
model — Clay +18.0 %, Prithvi +13.8 %, Satlas +9.1 % — while the two generic ImageNet models
moved least or backwards (ResNet +4.0 %, ViT −3.5 %). The contaminated data had been
flattering precisely the models with no earth-observation grounding.

### 3.4 No-data handling

Sentinel-2 marks absent data as zero. A per-band zero test is unsafe, because genuinely dark
surfaces — deep water, shadow, burn scars — are legitimately near-zero in individual bands.
An earlier per-band implementation of this test replaced such pixels with band medians and,
in the worst case, **fabricated 98 % of one wetland patch's blue band**. The corrected rule
treats a pixel as no-data only when it is zero in **every** band, and fills from the median
of valid pixels within the same patch.

### 3.5 Quality control and provenance

A QC gate runs before any embedding is computed, performing pixel-level content
verification: patches that are predominantly water, predominantly no-data, or spectrally
implausible for their declared ecosystem are rejected. Every location's declared ecosystem
is additionally verified against the **RESOLVE Ecoregions 2017** biome at its coordinate
before insertion into the catalog; this check rejected **2 of 50** candidate locations
during the catalog expansion.

Of 131 configured locations, **126 were acquired**. The five losses are documented
individually. Four share a single structural cause: **narrow-fringe coastal ecosystems
cannot be sampled at a 2,240 m footprint.** Dutch salt marshes sit in strips a few hundred
metres wide behind a dike; the Florida Keys and sparse Western Australian mangroves are
similarly sub-footprint. Six of the ten original acquisition failures, and three of the
final four, were mangrove sites — a systematic bias, not chance, and one that leaves the
surviving mangrove set skewed toward large, landward-fringed systems.

*Artifacts:* `metadata/catalog.json`, `results/phase0_qc_report.*`, `metadata/norm_stats.json`.

---

## §4 Representation choices — layer, pooling and dimensionality  *(C9)*

Two choices in the embedding pipeline were initially inherited rather than justified: which
transformer block to read, and what embedding dimensionality to retain. Both were swept.

### 4.1 Layer and pooling

The implementation read Prithvi's **block 8** with mean-pooling over patch tokens, annotated
in the source as "block 8 for optimal semantic features" — an assertion with no measurement
behind it. We swept all twelve blocks under both mean-pooling and CLS-token extraction,
evaluated with the grouped retrieval protocol on all 1,260 catalog sub-crops. Because
`forward_features()` returns every block from a single forward pass, the entire sweep costs
one pass per patch rather than twelve.

| Configuration | mAP@5 |
|---|---|
| Block 11, CLS | **0.4332** |
| Block 11, mean | 0.4285 |
| Block 4, CLS | 0.4285 |
| **Block 8, mean (inherited default)** | 0.4181 |
| Block 0, mean | 0.3851 |

The inherited default is **not optimal**, but the choice is not load-bearing: every block
from 4 upward falls within a 1.5-point band. Only block 0 — effectively the patch
projection before transformer processing — is clearly worse, which confirms the encoder
contributes real structure. We therefore retain block 8 for continuity with published
Prithvi usage, and report the sweep rather than the assertion.

### 4.2 Dimensionality

Embeddings were PCA-reduced to 64, 128, 256 and 512 dimensions and re-evaluated.

| Model | native | 64-D | 128-D | 512-D |
|---|---|---|---|---|
| Clay-v1.5 | 0.4381 (1024) | 0.4455 | **0.4524** | 0.4512 |
| Prithvi-100M | 0.4181 (768) | **0.4252** | 0.4245 | 0.4229 |
| ResNet-50 | 0.4031 (2048) | 0.3843 | 0.3959 | 0.3990 |
| Satlas-RN50 | 0.3632 (2048) | **0.3694** | 0.3639 | 0.3595 |
| ViT-Base | 0.3300 (768) | 0.3249 | **0.3334** | 0.3319 |

Two conclusions follow. First, **dimensionality is not a meaningful constraint**: 64
dimensions match or exceed native performance for four of five models, so the retrieval-relevant
signal occupies a small subspace. Second, dimensionality **does not explain the ranking**.
Compared at an identical 64 dimensions, Clay still leads Prithvi 0.4455 to 0.4252, so the
2048-dimensional CNNs were never enjoying an unfair advantage — indeed they are the models
PCA harms. Practically, the catalog can be stored at 128 dimensions, an eight-fold
reduction for Clay at no measured cost.

### 4.3 Footprint

By contrast with dimensionality, the **ground footprint is load-bearing**. Holding model
input size fixed at 224 px and varying only the area of ground covered, retrieval degrades
monotonically: ResNet-50 falls from 0.3492 at 1,600 m to 0.1905 at 457 m, a **45.5 % loss**.
The representation is thus robust to how many numbers describe a patch and fragile to how
much ground the patch covers. Performance had not plateaued at the largest footprint
available from existing data, so the upper half of this curve remains open.

*Artifacts:* `results/layer_ablation.json`, `results/dimension_sweep.json`,
`results/patch_size_sweep.json`.

---

## §5 Motivation — why learned embeddings rather than spectral indices  *(C3)*

A reasonable objection to foundation-model retrieval is that handcrafted spectral indices
might suffice. We therefore constructed a deliberately fair opponent: a four-dimensional
descriptor of forest cover, water cover, urban cover and vegetation health derived from
NDVI, NDWI and NDBI, retrieved and evaluated under an identical protocol.

**The answer depends on what is asked, and we report both halves.**

On **category retrieval** over the ten-ecosystem catalog, only one model clears the baseline
convincingly:

| Method | mAP@5 | vs. baseline |
|---|---|---|
| **Clay-v1.5** | 0.3161 | **+26.4 %** |
| Prithvi-100M | 0.2525 | +1.0 % |
| *spectral baseline* | *0.2501* | — |
| ResNet-50 | 0.2466 | −1.4 % |
| Satlas-RN50 | 0.2429 | −2.9 % |
| ViT-Base | 0.1970 | −21.2 % |

Two of five models fall below a four-dimensional handcrafted descriptor. This is reported,
not buried. It is, however, catalog-dependent in an informative direction: on a narrower
five-category catalog Clay's margin was +14.0 %, and it **nearly doubled to +26.4 %** once
the catalog spanned genuine ecological diversity. Learned embeddings earn their keep
precisely where the ecosystem range is wide.

On **ecological agreement** — whether retrieved analogs share real environmental
characteristics, rather than merely a shared label — the comparison is decisive:

| Attribute | random | spectral baseline | best model |
|---|---|---|---|
| Temperature MAE | 12.99 °C | 12.07 °C | **6.34 °C** (Clay) |
| Rainfall MAE | 862.9 mm | 717.2 mm | **522.8 mm** (Clay) |
| Disturbance trajectory (Jaccard) | 0.163 | 0.132 | **0.232** (Satlas) |

The spectral baseline barely improves on random for climate (12.07 °C against 12.99 °C),
while the foundation models roughly halve the error. It is **worse than random** on
disturbance trajectory.

We exclude the forest-cover row from this comparison on principle: the baseline *ranks by*
forest cover, so agreement on forest cover measures the method against its own input.

**The defensible claim is therefore narrow and specific:** learned embeddings encode
climatic and biogeographic structure that spectral indices do not, even where indices remain
competitive at coarse category retrieval.

*Artifacts:* `results/evaluation_report.json`, `results/ecological_similarity.json`.

---

## §7 Failure analysis — visually similar, ecologically different  *(C14)*

Aggregate metrics conceal the failure mode that matters most for an analog-retrieval system:
pairs with high embedding similarity and high ecological disagreement. We selected cases
from the high-similarity band and inspected them against independent descriptors.

**Satlas-RN50 — Florida Bay mangroves → Siberian boreal forest (cosine 0.9854).**
A tropical coastal wetland matched to taiga. The retrieval justification cites pristine
surface condition, strong photosynthetic activity and seven shared Hansen loss years. The
match is defensible on canopy texture and greenness and indefensible ecologically.

**Prithvi-100M — Tiergarten, Berlin → Valdivian temperate rainforest, Chile (cosine 0.9967).**
The highest-similarity cross-category failure observed: an urban park matched to temperate
rainforest.

**ViT-Base — Congo Basin, Gabon → Mesopotamian Marshes, Iraq (cosine 0.9858).**
Both sit near NDVI 0.85. The model matches greenness while disregarding that one is closed
tropical canopy and the other is Iraqi marshland.

These cases share a mechanism: **embedding similarity tracks canopy texture and greenness,
not ecological function.** Three quantitative results corroborate it. Ecosystems do not form
separated clusters in the embedding space — three of five models score a *negative*
silhouette. Similarity is usable as a ranking but not as a partition. And a threshold sweep
shows that even at the best operating point, precision reaches only 0.2309 (2.35 × chance)
for the strongest model.

The practical consequence is a design constraint rather than a defect: **retrieval output
should be presented as ranked candidates for expert review, with an explicit no-analog-found
response below threshold, and never as an ecological equivalence claim.**

*Artifacts:* `results/retrieval_case_studies_*.json`, `results/case_studies/*.png`,
`results/retrieval_diagnostics.json`.
