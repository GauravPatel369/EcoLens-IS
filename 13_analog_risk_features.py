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

# ---- NETWORK TIMEOUTS (added 27 Sep) ----------------------------------------------------
# GDAL applies NO timeout to HTTP range reads by default, and pystac_client's session has
# none either. One stalled connection to Planetary Computer therefore blocks forever: this
# script hung for 6 h 19 m at cell 193 of 318, process alive, not one byte written to its
# log. The stall happened inside the ThreadPoolExecutor, and because results are collected
# with pool.map the whole chunk waited on that one dead thread -- so the entire job froze
# instead of losing a single cell.
#
# Set before rasterio/GDAL is imported anywhere so the driver picks them up at init.
# Generous but finite: a real six-band read takes ~20 s, so 120 s means "this connection is
# dead", not "this connection is slow".
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

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


def _clip_hist(h, obs_year):
    """Drop loss years after obs_year.

    Any feature derived from a disturbance history has to be built only from what was
    observable at prediction time. Returns None unchanged so callers keep their
    "no history available" path.
    """
    if h is None:
        return None
    return {y: p for y, p in h.items() if y <= obs_year}


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
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt cells previously recorded as unembeddable. Off by "
                         "default so a resumed run does not spend its first hour failing "
                         "the same fetches again.")
    ap.add_argument("--random-analogs", action="store_true",
                    help="CONTROL ARM: pick analogs at random from the allowed pool instead "
                         "of by similarity. Everything else -- leakage guards, top-k, the 8 "
                         "feature columns, the horizon windows -- is identical, so comparing "
                         "this against the real run isolates whether SIMILARITY matters or "
                         "whether any 8 extra columns would have helped. Writes to a "
                         "_randctrl suffixed file so it never overwrites the real features.")
    ap.add_argument("--workers", type=int, default=8,
                    help="threads used for the Sentinel-2 windowed reads. The fetch is "
                         "~99%% network wait (measured: 21-33 s per cell, of which the "
                         "forward pass is 0.28 s), so this scales nearly linearly. Only "
                         "the reads are threaded; STAC search and the model stay serial.")
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
    # The control arm must never overwrite the real features -- 14 reads these files by name,
    # and a clobbered analog_cell_features.csv would silently turn the headline result into
    # the control result with no visible sign.
    _rand_ctrl = np.random.default_rng(args.seed + 991)
    if args.random_analogs:
        OUT = OUT.replace(".csv", "_randctrl.csv")
        print("*** RANDOM-ANALOG CONTROL ARM: analogs drawn at random, not by similarity ***")
        print(f"*** writing to {OUT} ***")
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
    # Cells already known to be unembeddable are skipped: they have no cloud-free scene or no
    # readable bands, and retrying them each run costs the same failures again. --retry-failed
    # forces another attempt (worth it after a long gap, since new imagery keeps arriving).
    unembeddable = set()
    _up = f"{RESULTS_DIR}/analog_cells_unembeddable_{args.model}.json"
    if os.path.exists(_up) and not args.retry_failed:
        try:
            unembeddable = set(json.load(open(_up, encoding="utf-8")))
            print(f"skipping {len(unembeddable)} cell(s) previously found unembeddable "
                  f"(--retry-failed to try again)")
        except Exception:
            unembeddable = set()
    todo = [r for r in picked.itertuples()
            if ckey(r.cell_lon, r.cell_lat) not in cache
            and ckey(r.cell_lon, r.cell_lat) not in unembeddable]
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
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor
        import threading

        # ---- THREADED FETCH; THE PER-CELL SCENE SEARCH IS PRESERVED ----------------
        # Measured cost per cell: 21-33 s, of which the model forward pass is 0.28 s. The
        # rest is one STAC search plus six windowed HTTPS reads, so this loop is ~99%
        # network wait and threads recover most of it.
        #
        # An earlier version of this block ALSO resolved one scene per region rather than
        # per cell, which is ~18% faster again. It was reverted on purpose: the region
        # scene is not always the scene a per-cell search would choose, so cells would be
        # embedded from different imagery and the resulting vectors would no longer match
        # the ones already in the cache. The analog ablation rests on those vectors, so the
        # speedup would have cost a re-run of the project's central result to save ~25
        # minutes. Keeping the per-cell search means embeddings are unchanged.
        #
        # pystac_client wraps a requests.Session that is not documented as thread-safe, and
        # 10._get_drift_stac_catalog() memoises ONE client in a module global -- so calling
        # it from every worker would share that single session. Each thread builds its own.
        _local = threading.local()

        def thread_stac():
            if not hasattr(_local, "stac"):
                import planetary_computer
                import pystac_client
                from config import PC_STAC_URL
                _local.stac = pystac_client.Client.open(
                    PC_STAC_URL, modifier=planetary_computer.sign_inplace)
            return _local.stac

        acq_mod = acq()          # pre-warm the lazy module import before threads touch it

        def fetch(r):
            """Scene search + windowed read for one cell. Runs in a worker thread.

            Returns (raw_patch, scene_date), or (None, None) when there is no cloud-free
            scene or a band read fails -- the caller counts those as failures rather than
            fabricating a vector.
            """
            item = find_scene_resilient(tl, thread_stac(), r.cell_lon, r.cell_lat,
                                        args.embed_year)
            if item is None:
                return None, None
            try:
                raw = acq_mod.extract_patch(item, r.cell_lon, r.cell_lat,
                                            PATCH_SIZE_M, PATCH_SIZE_PX, PRITHVI_BANDS)
            except Exception:
                return None, None
            try:
                date = str(item.properties.get("datetime", ""))[:10] or None
            except Exception:
                date = None
            return raw, date

        # Fetch in bounded chunks: pool.map over all of `todo` would queue every raw
        # patch in memory (602 KB each -- 8 GB for a 13,300-cell run). A chunk also
        # marks a natural save point, so an interrupted run resumes from the last one.
        n_ok = n_fail = n_timeout = 0
        CHUNK = max(1, args.workers * 4)
        # Wall-clock budget for a whole chunk. The GDAL timeouts above should catch a dead
        # socket, but they do not cover every way a read can stall (DNS, TLS handshake, a
        # server holding the connection open). submit()+as_completed with a timeout is the
        # backstop: whatever has not arrived is abandoned and the run moves on. The worker
        # thread may leak, which is acceptable -- a leaked thread costs memory, a hung
        # pool.map costs the entire job, as it did for 6 h 19 m.
        CHUNK_TIMEOUT_S = 90 * CHUNK / max(1, args.workers)
        with tqdm(total=len(todo), desc="embedding cells") as bar:
            for start in range(0, len(todo), CHUNK):
                batch = todo[start:start + CHUNK]
                fetched = [(None, None)] * len(batch)
                pool = ThreadPoolExecutor(max_workers=args.workers)
                try:
                    futs = {pool.submit(fetch, r): i for i, r in enumerate(batch)}
                    from concurrent.futures import as_completed, TimeoutError as FTimeout
                    try:
                        for fu in as_completed(futs, timeout=CHUNK_TIMEOUT_S):
                            try:
                                fetched[futs[fu]] = fu.result()
                            except Exception:
                                pass
                    except FTimeout:
                        stalled = sum(1 for fu in futs if not fu.done())
                        n_timeout += stalled
                        print(f"\n  [Warn] {stalled} cell(s) exceeded "
                              f"{CHUNK_TIMEOUT_S:.0f}s and were abandoned; continuing.")
                finally:
                    # do not wait on stalled threads -- that is the hang we are fixing
                    pool.shutdown(wait=False, cancel_futures=True)
                # embed serially: one torch model, shared, and the forward pass is 1% of
                # the cost -- threading it would risk the model for no measurable gain
                for r, (raw, scene_date) in zip(batch, fetched):
                    if raw is None:
                        n_fail += 1
                        bar.update(1)
                        continue
                    try:
                        v = embed_cell(model, args.model, raw, r.cell_lon, r.cell_lat,
                                       scene_date, means, stds)
                    except Exception:
                        n_fail += 1
                        bar.update(1)
                        continue
                    v = v.astype(np.float32)
                    v /= np.linalg.norm(v) + 1e-8
                    cache[ckey(r.cell_lon, r.cell_lat)] = v
                    n_ok += 1
                    bar.update(1)
                np.savez_compressed(CACHE, **cache)
        print(f"embedded {n_ok}, failed {n_fail} (no cloud-free scene / read error), "
              f"timed out {n_timeout}")
        # Record cells that could not be embedded so later runs skip them instead of
        # re-attempting the same hopeless fetches every time. They failed for real reasons --
        # persistent cloud, a scene with no usable bands -- and retrying them on every run is
        # what turned a resumable job into one that spent its first hour going nowhere.
        if n_fail or n_timeout:
            failed_keys = [ckey(r.cell_lon, r.cell_lat) for r in todo
                           if ckey(r.cell_lon, r.cell_lat) not in cache]
            fpath = f"{RESULTS_DIR}/analog_cells_unembeddable_{args.model}.json"
            prev = []
            if os.path.exists(fpath):
                try:
                    prev = json.load(open(fpath, encoding="utf-8"))
                except Exception:
                    prev = []
            merged = sorted(set(prev) | set(failed_keys))
            json.dump(merged, open(fpath, "w", encoding="utf-8"), indent=1)
            print(f"  {len(failed_keys)} unembeddable cell(s) recorded in "
                  f"{os.path.basename(fpath)} ({len(merged)} total); future runs skip them")

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

        if args.random_analogs:
            # RANDOM-ANALOG CONTROL. Identical pipeline, identical leakage guards, identical
            # number of analogs and identical downstream features -- the ONLY difference is
            # that similarity is replaced by noise, so the "analogs" are drawn at random from
            # the allowed pool instead of being the most similar places.
            #
            # This is the control the whole project rests on. Without it, "drivers + 8 analog
            # columns beats drivers alone" has an obvious rival explanation: eight extra
            # columns of any kind give a gradient-boosted model more capacity to fit. If real
            # analogs beat random ones, similarity is doing the work and the claim holds. If
            # they tie, the +11.5% is an artifact of feature count, not of retrieval.
            sims = _rand_ctrl.random(int(allowed.sum()))
        else:
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

        # fragmentation: how much more/less fragmented is this cell than its analogs
        cell_ed = edge(r.cell_lon, r.cell_lat)
        a_eds = [e for e in (edge(plons[j], plats[j]) for j in a_idx) if e is not None]
        frag_delta = (float(cell_ed - np.mean(a_eds))
                      if cell_ed is not None and a_eds else None)

        for y in years:
            s = summarise_analog_history(a_hists, y, RISK_HORIZON_YEARS)
            if s is None:
                continue
            # LEAK FIX (26 Sep). analog_trajectory_jaccard used to be computed ONCE here,
            # outside this loop, from the FULL 2001-2023 loss-year sets, and then written
            # identically for all 17 observation years. So the row for obs_year 2010 carried
            # a feature that already knew about loss up to 2023 -- the model could read the
            # future through it. summarise_analog_history was always correctly windowed
            # (y <= obs_year for prior, obs_year < y <= obs_year+horizon for the target);
            # this one column was not. Clip BOTH sides to <= y and recompute per year.
            jac_y = trajectory_jaccard(_clip_hist(cell_hist, y),
                                       [_clip_hist(h, y) for h in a_hists])
            rows.append({
                "region_id": r.region_id,
                "cell_lon": round(r.cell_lon, 5),
                "cell_lat": round(r.cell_lat, 5),
                "obs_year": y,
                **s,
                "analog_trajectory_jaccard": jac_y,
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
