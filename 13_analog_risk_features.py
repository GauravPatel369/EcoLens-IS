"""
EcoLens Step 13 -- ANALOG-TRANSFERRED RISK FEATURES  (Phase 4; C1, C15, C16, C20)

This is the script that makes the re-framed research question answerable:

    Can geospatial foundation models identify ecological analogs whose historical
    disturbance trajectories improve forest-loss risk forecasting ACROSS
    GEOGRAPHICALLY DISTINCT LANDSCAPES?

Pillar A (retrieval) and Pillar B (forecasting) have never been connected. The
only embedding-derived column the risk model has ever seen is `embedding_drift`,
which is a cell's similarity to ITSELF at an earlier date -- self-similarity over
time, not analog transfer. This script produces the missing link: for a grid cell,
find ecologically similar places ELSEWHERE, and summarise what happened to them.

WHY THE SPATIAL-HOLDOUT NUMBER IS THE ONE THAT MATTERS
-----------------------------------------------------
implementation.md 3.4d: the driver-only model scores PR-AUC 0.6923 on a temporal
split but collapses to a mean 0.3172 under leave-one-location-out, with 4 of 17
regions BELOW random. It predicts future years of regions it has seen and does not
transfer. That gap is the headroom this feature set has to fill, and a gain on the
temporal split would prove almost nothing.

TWO CORRECTNESS REQUIREMENTS, BOTH EASY TO GET WRONG
----------------------------------------------------
1. NORMALIZATION AND GEOMETRY MUST MATCH THE CATALOG.
   Catalog embeddings come from patches_processed/: a 160px centre-ish crop of the
   224px acquisition, resized to 224, z-scored with per-band stats computed over
   all acquired patches. 10_grid_tiling_labels.compute_cell_embedding() does NOT
   do this -- it feeds raw DN straight into the model. That is fine for
   embedding_drift (a cell compared to itself, both raw) but it puts cell vectors
   in a DIFFERENT SPACE from catalog vectors, so every cosine between them would
   be meaningless. This script therefore reimplements the 02 path exactly and
   loads the stats from metadata/norm_stats.json rather than guessing them.

2. LEAKAGE. An analog must not be the cell's own neighbourhood. Two guards:
     * same-region exclusion  -- drop every catalog patch belonging to the region
       being scored (region_id -> the config location it was tiled around)
     * geographic exclusion   -- drop any candidate within --exclusion-km
   Without these the feature is a laundered copy of the cell's own history, and
   the whole result is worthless. This is the grid-cell equivalent of 07's
   GROUPED protocol.

Outputs results/analog_cell_features.csv, keyed by (cell_lon, cell_lat, obs_year),
ready to merge into risk_model/cell_year_features.csv for 11's four-arm ablation.

RESUMABLE: per-cell embeddings are cached to results/analog_cell_embeddings.npz and
the CSV is written incrementally, so an interrupted run continues where it stopped.

Run:
    python 13_analog_risk_features.py --cells-per-region 20
    python 13_analog_risk_features.py --cells-per-region 20 --model clay
"""

import argparse
import json
import math
import os
import sys

import numpy as np

from config import (
    METADATA_CATALOG_PATH, METADATA_DIR, RESULTS_DIR, SUPPORTED_MODELS,
    RISK_FEATURES_PATH, RISK_HORIZON_YEARS, PATCH_SIZE_M, PATCH_SIZE_PX,
    PRITHVI_BANDS, PATCH_LOCATIONS,
)

# Cache and output are PER MODEL. They were not, and that was a live bug: the cache is
# keyed only by coordinate, so `--model clay` would have loaded Prithvi's 1,573 cached
# vectors, reported "resuming: 1573 cell embeddings already cached", skipped all the work
# and then retrieved Clay analogs using PRITHVI embeddings. The comparison would have been
# silently meaningless -- the same class of failure as 03's skip-if-exists (§3.2).
def cache_path(model_key):
    return f"{RESULTS_DIR}/analog_cell_embeddings_{model_key}.npz"


def out_path(model_key):
    # prithvi keeps the unsuffixed name so 14/15 and existing results keep working
    return (f"{RESULTS_DIR}/analog_cell_features.csv" if model_key == "prithvi"
            else f"{RESULTS_DIR}/analog_cell_features_{model_key}.csv")


LEGACY_CACHE = f"{RESULTS_DIR}/analog_cell_embeddings.npz"   # prithvi, pre-fix
OUT_PATH = f"{RESULTS_DIR}/analog_cell_features.csv"
NORM_STATS_PATH = f"{METADATA_DIR}/norm_stats.json"

CROP_SIZE = 160
RESIZE_TO = 224


# ---------------------------------------------------------------
# lazy module loading (the numbered scripts are not importable by name)
# ---------------------------------------------------------------
_mods = {}


def _mod(name, path):
    if name not in _mods:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        _mods[name] = m
    return _mods[name]


def acq():
    return _mod("_acq", "01_acquire_patches.py")


def pre():
    return _mod("_pre", "02_preprocess_patches.py")


def emb():
    return _mod("_emb", "03_extract_embeddings.py")


def tiling():
    return _mod("_tiling", "10_grid_tiling_labels.py")


def explain():
    return _mod("_explain", "09_explainability_engine.py")


# ---------------------------------------------------------------
# geometry
# ---------------------------------------------------------------

def find_scene_resilient(tl, stac, lon, lat, year, attempts=4):
    """_find_scene_for_year with backoff.

    A single transient DNS failure killed a whole run on 3 Sep:
      APIError: ... Failed to resolve 'planetarycomputer.microsoft.com'
    01_acquire_patches.py already learned this lesson (its STAC search retries;
    see implementation.md 3.1) but that retry lives inside 01, not in 10's
    _find_scene_for_year, which this script uses. An unattended run must survive a
    few seconds of flaky network, so wrap it here.

    Returns None only after every attempt fails -- never a fabricated scene.
    """
    import time
    delay = 3.0
    for i in range(attempts):
        try:
            return tl._find_scene_for_year(stac, lon, lat, year)
        except Exception as e:
            if i == attempts - 1:
                print(f"  STAC search gave up after {attempts} attempts "
                      f"at ({lon:.4f}, {lat:.4f}): {type(e).__name__}")
                return None
            time.sleep(delay)
            delay *= 2
    return None


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_norm_stats():
    if not os.path.exists(NORM_STATS_PATH):
        raise FileNotFoundError(
            f"{NORM_STATS_PATH} not found. Run 02_preprocess_patches.py (it writes the "
            f"stats now). Embedding a cell without the catalog's exact normalization "
            f"puts it in a different space -- see this module's docstring."
        )
    with open(NORM_STATS_PATH, encoding="utf-8") as f:
        s = json.load(f)
    return (np.array(s["means"], dtype=np.float32),
            np.array(s["stds"], dtype=np.float32), s)


def crop_like_catalog(raw_patch):
    """Centre 160 crop -> 224 resize, matching 02's sub-crop geometry. No z-score."""
    p = pre()
    c = (PATCH_SIZE_PX - CROP_SIZE) // 2
    crop = raw_patch[:, c:c + CROP_SIZE, c:c + CROP_SIZE]
    return p.resize_patch_torch(crop, target_size=RESIZE_TO)


def embed_cell(model, model_key, raw_patch, lon, lat, scene_date, means, stds):
    """Embed a cell EXACTLY the way 03 built that model's catalog vectors.

    THE MODELS DO NOT SHARE A PREPROCESSING PATH, and getting this wrong silently
    puts cell and catalog vectors in different spaces (see this module's docstring
    for what that costs -- an Amazon cell retrieving Egyptian farmland):
      * prithvi           -> 03.run_prithvi reads entry["processed_path"], i.e. the
                             02 Z-SCORED patch. So z-score here too.
      * vit/resnet/satlas -> 03.run_timm_model / run_satlas read entry["patch_path"],
                             i.e. the RAW patch. No z-score.
      * clay              -> 03.run_clay reads the RAW patch and lets
                             prepare_clay_datacube apply Clay's own per-band
                             normalization from metadata.yaml. Z-scoring first would
                             normalize twice.
    """
    import torch
    ext = emb()
    resized = crop_like_catalog(raw_patch)

    if model_key == "prithvi":
        p = pre()
        proc = p.normalize(p.handle_nodata(resized), means, stds)
        return ext.extract_embedding(model, torch.from_numpy(proc).float())

    if model_key == "clay":
        dc = ext.prepare_clay_datacube(resized, lon, lat, scene_date)
        return ext.extract_clay_embedding(model, dc)

    if model_key == "satlas":
        t = ext.prepare_satlas_rgb_tensor(resized)
        t = t.unsqueeze(0) if t.dim() == 3 else t
        with torch.no_grad():
            feats = model(t.to(tiling().DEVICE if hasattr(tiling(), "DEVICE") else "cpu"))
        return feats[-1].mean(dim=[2, 3]).squeeze(0).cpu().numpy()

    # vit / resnet -- raw patch, no z-score
    return ext.extract_timm_embedding(model, ext.prepare_rgb_tensor(resized))


def preprocess_like_catalog(raw_patch, means, stds):
    """Prithvi-only: the full 02 path (centre 160 crop -> 224 resize -> nodata fill
    -> z-score). Kept for the space-match test; embed_cell() is the general path."""
    p = pre()
    c = (PATCH_SIZE_PX - CROP_SIZE) // 2
    crop = raw_patch[:, c:c + CROP_SIZE, c:c + CROP_SIZE]
    resized = p.resize_patch_torch(crop, target_size=RESIZE_TO)
    clean = p.handle_nodata(resized)
    return p.normalize(clean, means, stds)


# ---------------------------------------------------------------
# analog pool
# ---------------------------------------------------------------

def load_analog_pool(model_key):
    """Catalog embeddings = the analog pool. Returns (ids, base_ids, lons, lats,
    ecosystems, matrix)."""
    with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    key = f"{model_key}_embedding" if model_key != "prithvi" else "prithvi_embedding"
    ids, bases, lons, lats, ecos, vecs = [], [], [], [], [], []
    for e in catalog:
        p = e.get(key)
        if not p or not os.path.exists(p):
            continue
        v = np.load(p).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        ids.append(e["id"])
        bases.append(e.get("base_id", e["id"].rsplit("_p", 1)[0]))
        lons.append(e["lon"]); lats.append(e["lat"])
        ecos.append(e["ecosystem"])
        vecs.append(v)
    if not vecs:
        raise RuntimeError(f"No embeddings found for model '{model_key}'. Run 03 first.")
    return (np.array(ids), np.array(bases), np.array(lons), np.array(lats),
            np.array(ecos), np.stack(vecs))


def region_base_ids(region_id):
    """Catalog base_ids that belong to the same config location a region was tiled
    around. region_id IS the location id (10 tiles around each forest location)."""
    return {region_id}


# ---------------------------------------------------------------
# analog-derived features
# ---------------------------------------------------------------

def summarise_analog_history(histories, obs_year, horizon):
    """Turn the analogs' Hansen loss-year vectors into transferable features.

    histories: list of {year: pct_of_patch_lost} for each analog (may be empty).
    All percentages are 'percent of that analog's patch that lost tree cover in
    that year', straight from 09.get_disturbance_history.
    """
    prior, at_horizon, had_any = [], [], []
    for h in histories:
        if h is None:
            continue
        prior.append(sum(p for y, p in h.items() if y <= obs_year))
        fut = sum(p for y, p in h.items() if obs_year < y <= obs_year + horizon)
        at_horizon.append(fut)
        had_any.append(1.0 if fut > 0 else 0.0)
    if not prior:
        return None
    return {
        # how disturbed were places like this, historically
        "analog_prior_loss_rate": float(np.mean(prior)),
        # "what happened to places like you" -- the transfer feature
        "analog_loss_rate_at_horizon": float(np.mean(at_horizon)),
        "analog_frac_with_loss_at_horizon": float(np.mean(had_any)),
        "analog_n": len(prior),
    }


def edge_density(lon, lat):
    """Forest/non-forest EDGE DENSITY over the patch footprint -- a fragmentation
    proxy (C20's fragmentation half).

    Hansen treecover2000 is thresholded at HANSEN_TREECOVER_THRESHOLD into a binary
    forest mask, and edge density is the fraction of adjacent pixel pairs that
    straddle the boundary. Intact canopy and bare ground both score near 0; a
    fragmented, perforated or edge-heavy landscape scores high. It is deliberately
    scale-free (a fraction, not a count) so patches of different valid extents stay
    comparable.

    Returns None when the tile is unavailable -- never a fabricated 0.0, which would
    read as "perfectly intact" and be the worst possible default (§9).
    """
    from config import HANSEN_TREECOVER_THRESHOLD
    tl = tiling()
    half_km = (PATCH_SIZE_M / 1000.0) / 2.0
    d_lat, d_lon = tl.km_to_deg(half_km, lat)
    path = tl.ensure_hansen_tile("treecover2000", lat, lon)
    if path is None:
        return None
    try:
        arr, _, _ = tl.read_window(path, lon - d_lon, lat - d_lat,
                                   lon + d_lon, lat + d_lat)
    except Exception:
        # A truncated tile (an interrupted download) raises RasterioIOError here and
        # would otherwise kill the whole run. One unreadable tile must cost one
        # feature value, not the job -- return None so the cell is skipped, never a
        # fabricated 0.0 (which would read as "perfectly intact", §9).
        return None
    if arr is None or arr.size == 0 or min(arr.shape) < 2:
        return None
    forest = (arr >= HANSEN_TREECOVER_THRESHOLD)
    h_edges = forest[:, 1:] != forest[:, :-1]
    v_edges = forest[1:, :] != forest[:-1, :]
    total = h_edges.size + v_edges.size
    if total == 0:
        return None
    return float((h_edges.sum() + v_edges.sum()) / total)


def trajectory_jaccard(cell_hist, analog_hists):
    """Mean Jaccard over loss-year sets.

    EMPTY-vs-EMPTY IS UNDEFINED, NOT A PERFECT MATCH. An earlier version returned
    1.0 when the union was empty, so "neither place has any recorded tree-cover
    loss" scored a flawless 1.000. The C20 trajectory figure exposed it
    immediately: the highest-scoring cells were flat zero lines agreeing with flat
    zero lines. That is not shared disturbance history, it is shared absence of
    data, and it was inflating the feature that ranked 4th in the ablation's
    permutation importance.

    Pairs where BOTH sides are empty are now skipped. If every pair is empty the
    feature is None -- unavailable, never fabricated (§9).
    """
    if cell_hist is None:
        return None
    cy = set(cell_hist.keys())
    vals = []
    for h in analog_hists:
        if h is None:
            continue
        ay = set(h.keys())
        union = cy | ay
        if not union:
            continue                      # <-- was `vals.append(1.0)`
        vals.append(len(cy & ay) / len(union))
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="EcoLens 13: analog-transferred risk features")
    ap.add_argument("--model", default="prithvi", choices=list(SUPPORTED_MODELS.keys()),
                    help="embedding model for the analog search. Default prithvi: it has "
                         "the best ecological-agreement scores in 07b (forest-cover MAE "
                         "10.13%% vs 40.92%% random) even though it ranks 2nd on mAP -- "
                         "analog quality is what matters here, not category retrieval.")
    ap.add_argument("--cells-per-region", type=int, default=20,
                    help="cells sampled per region. 8,469 cells x (STAC fetch + embed) is "
                         "many hours; this bounds it to a real, honest subset.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--exclusion-km", type=float, default=250.0,
                    help="drop analog candidates within this radius of the cell. Without "
                         "it the 'analog' can be the cell's own neighbourhood and the "
                         "feature is leakage.")
    ap.add_argument("--embed-year", type=int, default=2021,
                    help="imagery year used to embed each cell")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    os.makedirs(RESULTS_DIR, exist_ok=True)
    means, stds, stats_meta = load_norm_stats()
    print(f"norm stats loaded ({stats_meta['n_base_patches']} base patches) "
          f"-- cell patches will be preprocessed identically to the catalog")

    ids, bases, plons, plats, pecos, P = load_analog_pool(args.model)
    print(f"analog pool: {len(ids)} catalog embeddings from {len(set(bases))} locations "
          f"({SUPPORTED_MODELS[args.model]['label']})")

    df = pd.read_csv(RISK_FEATURES_PATH)
    cells = df[["region_id", "cell_lon", "cell_lat"]].drop_duplicates()
    # Explicit loop rather than groupby().apply(): with group_keys=False pandas moved
    # region_id into the index, so itertuples() rows had no .region_id and the
    # same-region leakage guard blew up with AttributeError. Concatenating the
    # per-group samples keeps every column a column.
    parts = []
    for _rid, g in cells.groupby("region_id"):
        parts.append(g.sample(min(len(g), args.cells_per_region), random_state=args.seed))
    picked = pd.concat(parts, ignore_index=True)
    print(f"cells sampled: {len(picked)} "
          f"({args.cells_per_region}/region x {cells.region_id.nunique()} regions)")

    # ---- resumable embedding cache ----
    CACHE = cache_path(args.model)
    OUT = out_path(args.model)
    # one-time migration: the pre-fix unsuffixed cache holds PRITHVI vectors only
    if args.model == "prithvi" and not os.path.exists(CACHE) and os.path.exists(LEGACY_CACHE):
        import shutil
        shutil.copyfile(LEGACY_CACHE, CACHE)
        print(f"migrated legacy prithvi cache -> {CACHE}")
    cache = {}
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        cache = {k: z[k] for k in z.files}
        print(f"resuming: {len(cache)} cell embeddings already cached")

    def ckey(lon, lat):
        return f"{lon:.5f}_{lat:.5f}"

    tl = tiling()
    todo = [r for r in picked.itertuples() if ckey(r.cell_lon, r.cell_lat) not in cache]
    print(f"cells needing embedding: {len(todo)}")

    if todo:
        # 10._get_drift_model only knows prithvi + timm models; for clay/satlas it
        # passes timm_name=None and crashes. Load those two directly from 03.
        ex = emb()
        if args.model == "clay":
            model = ex.load_clay_model()
        elif args.model == "satlas":
            model = ex.load_satlas_model()
        else:
            model = tl._get_drift_model(args.model)
        stac = tl._get_drift_stac_catalog()
        from tqdm import tqdm
        n_ok = n_fail = 0
        for i, r in enumerate(tqdm(todo, desc="embedding cells")):
            item = find_scene_resilient(tl, stac, r.cell_lon, r.cell_lat, args.embed_year)
            if item is None:
                n_fail += 1
                continue
            scene_date = None
            try:
                scene_date = str(item.properties.get("datetime", ""))[:10] or None
            except Exception:
                pass
            try:
                raw = acq().extract_patch(item, r.cell_lon, r.cell_lat,
                                          PATCH_SIZE_M, PATCH_SIZE_PX, PRITHVI_BANDS)
                v = embed_cell(model, args.model, raw, r.cell_lon, r.cell_lat,
                               scene_date, means, stds)
            except Exception:
                n_fail += 1
                continue
            v = v.astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            cache[ckey(r.cell_lon, r.cell_lat)] = v
            n_ok += 1
            if n_ok % 25 == 0:
                np.savez_compressed(CACHE, **cache)
        np.savez_compressed(CACHE, **cache)
        print(f"embedded {n_ok}, failed {n_fail} (no cloud-free scene / read error)")

    # ---- retrieve analogs + build features ----
    hist_cache = {}

    def hist(lon, lat):
        k = (round(lon, 4), round(lat, 4))
        if k not in hist_cache:
            h = explain().get_disturbance_history(lon, lat)
            hist_cache[k] = h.get("loss_years") if h else None
        return hist_cache[k]

    edge_cache = {}

    def edge(lon, lat):
        k = (round(lon, 4), round(lat, 4))
        if k not in edge_cache:
            edge_cache[k] = edge_density(lon, lat)
        return edge_cache[k]

    from tqdm import tqdm
    rows = []
    years = sorted(df.obs_year.unique())
    for r in tqdm(list(picked.itertuples()), desc="analog features"):
        v = cache.get(ckey(r.cell_lon, r.cell_lat))
        if v is None:
            continue

        # --- leakage guards ---
        same_region = np.isin(bases, list(region_base_ids(r.region_id)))
        dists = np.array([haversine_km(r.cell_lon, r.cell_lat, lo, la)
                          for lo, la in zip(plons, plats)])
        too_close = dists < args.exclusion_km
        allowed = ~(same_region | too_close)
        if not allowed.any():
            continue

        sims = P[allowed] @ v
        allowed_idx = np.where(allowed)[0]

        # ONE ANALOG PER LOCATION. The pool holds 10 overlapping sub-crops per
        # location, so a naive top-k returns k sub-crops of the SAME place --
        # the C20 figure showed "analog 1..5" all reading tundra_007. That makes
        # top-5 a disguised top-1: the features average five near-identical
        # vectors, giving false confidence and no diversity (it is also why
        # analog_frac_with_loss_at_horizon only ever took 6 values). Keep the
        # best-matching sub-crop per base location, then take the top-k
        # DISTINCT locations.
        best_per_base = {}
        for rank in np.argsort(-sims):
            b = bases[allowed_idx[rank]]
            if b not in best_per_base:
                best_per_base[b] = rank
            if len(best_per_base) >= args.top_k:
                break
        order = np.array(sorted(best_per_base.values(), key=lambda k: -sims[k]))
        a_idx = allowed_idx[order]
        a_sims = sims[order]

        a_hists = [hist(plons[j], plats[j]) for j in a_idx]
        cell_hist = hist(r.cell_lon, r.cell_lat)
        jac = trajectory_jaccard(cell_hist, a_hists)

        # fragmentation: how much more/less fragmented is this cell than its analogs
        cell_ed = edge(r.cell_lon, r.cell_lat)
        a_eds = [e for e in (edge(plons[j], plats[j]) for j in a_idx) if e is not None]
        frag_delta = (float(cell_ed - np.mean(a_eds))
                      if cell_ed is not None and a_eds else None)

        for y in years:
            s = summarise_analog_history(a_hists, y, RISK_HORIZON_YEARS)
            if s is None:
                continue
            rows.append({
                "region_id": r.region_id,
                "cell_lon": round(r.cell_lon, 5),
                "cell_lat": round(r.cell_lat, 5),
                "obs_year": y,
                **s,
                "analog_trajectory_jaccard": jac,
                "analog_fragmentation_delta": frag_delta,
                "cell_edge_density": cell_ed,
                "analog_mean_similarity": float(np.mean(a_sims)),
                "analog_similarity_spread": float(np.std(a_sims)),
                "analog_top_ecosystems": "|".join(pecos[j] for j in a_idx),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out)} cell-year rows "
          f"for {out[['cell_lon','cell_lat']].drop_duplicates().shape[0]} cells")
    if len(out):
        print("\nfeature fill rates:")
        for c in ["analog_prior_loss_rate", "analog_loss_rate_at_horizon",
                  "analog_frac_with_loss_at_horizon", "analog_trajectory_jaccard",
                  "analog_fragmentation_delta", "cell_edge_density",
                  "analog_mean_similarity"]:
            print(f"  {c:<38} {100*out[c].notna().mean():5.1f}%  "
                  f"mean={out[c].mean():.4f}")
        print("\nanalog ecosystems retrieved (top-1 across cells):")
        top1 = out.drop_duplicates(["cell_lon", "cell_lat"])["analog_top_ecosystems"].str.split("|").str[0]
        print(top1.value_counts().to_string())


if __name__ == "__main__":
    main()
