"""
EcoLens Step 18 -- VISUAL CASE-STUDY PANELS  (Phase 3; C13, C14)

C13 asks for "visual examples of successful / partial / failed retrievals".
07d_retrieval_case_studies.py identifies the cases but emits JSON with **no imagery
at all** (implementation.md C13), so the half the reviewer actually asked for has
never existed. This renders them.

Each panel: the query patch's RGB alongside its retrieved analog, annotated with
cosine similarity, whether the categories match, and the descriptor disagreement
that 09 measured. Three rows -- SUCCESS, PARTIAL, FAILURE -- in one figure per model,
so the reader sees the failure modes beside the wins rather than a curated highlight
reel (implementation.md 9).

RGB rendering matches what the models actually saw: the acquired 6-band patch,
BOA-corrected, bands red/green/blue = indices 2/1/0 of PRITHVI_BANDS, contrast
stretched per-image with a percentile clip. The stretch is cosmetic and is stated on
the figure -- it changes how the patch LOOKS, never what was embedded.

Outputs results/case_studies/<model>_cases.png

Run:
    python 18_case_study_figures.py
    python 18_case_study_figures.py --model clay
"""

import argparse
import json
import os

import numpy as np

from config import RESULTS_DIR, SUPPORTED_MODELS, METADATA_CATALOG_PATH

OUT_DIR = f"{RESULTS_DIR}/case_studies"


def load_catalog_index():
    with open(METADATA_CATALOG_PATH, encoding="utf-8") as f:
        cat = json.load(f)
    return {e["id"]: e for e in cat}


def rgb_from_patch(entry, idx):
    """Render a sub-crop as RGB, reproducing 02's crop geometry so the picture shows
    what was embedded rather than the whole acquisition."""
    p = entry.get("patch_path")
    if not p or not os.path.exists(p):
        return None
    arr = np.load(p).astype(np.float32)
    dy, dx = entry.get("crop_offset", [0, 0])
    cs = entry.get("crop_size", 160)
    r, c = 32 + dy, 32 + dx
    if r + cs <= arr.shape[1] and c + cs <= arr.shape[2]:
        arr = arr[:, r:r + cs, c:c + cs]
    rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1)   # red, green, blue
    lo, hi = np.percentile(rgb, [2, 98])
    if hi <= lo:
        return None
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def select_cases(cases, band, n):
    """Pick n cases worth looking at.

    Taking 07d's list in order gives a poor panel: it contains SYMMETRIC DUPLICATES
    (A->B and B->A are separate entries, and both render as the same pair of images)
    and, for Clay, 14 of 15 cases were urban_green, so a naive head-3 produced three
    near-identical city parks in every band.

    Rules, in order:
      * drop symmetric duplicates and repeated base-location pairs
      * in the FAILURE band prefer same_category == False -- the "visually similar but
        ecologically different" quadrant is what C14 is actually about
      * otherwise spread across query ecosystems
    """
    seen_pairs, seen_ecos, picked = set(), set(), []

    def base(pid):
        return pid.rsplit("_p", 1)[0]

    ordered = cases
    if band == "failure":
        ordered = ([c for c in cases if not c["same_category"]] +
                   [c for c in cases if c["same_category"]])

    for pass_no in (0, 1):          # pass 0 enforces ecosystem diversity, pass 1 fills up
        for c in ordered:
            if len(picked) >= n:
                break
            key = frozenset((base(c["query_id"]), base(c["analog_id"])))
            if key in seen_pairs:
                continue
            if pass_no == 0 and c["query_ecosystem"] in seen_ecos:
                continue
            seen_pairs.add(key)
            seen_ecos.add(c["query_ecosystem"])
            picked.append(c)
    return picked


def main():
    ap = argparse.ArgumentParser(description="EcoLens 18: case-study figures (C13/C14)")
    ap.add_argument("--model", default=None, help="default: every model with a case file")
    ap.add_argument("--per-band", type=int, default=3, help="cases per band (success/partial/failure)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    index = load_catalog_index()
    models = [args.model] if args.model else list(SUPPORTED_MODELS)

    for mk in models:
        path = f"{RESULTS_DIR}/retrieval_case_studies_{mk}.json"
        if not os.path.exists(path):
            print(f"[{mk}] no case file -- run 07d_retrieval_case_studies.py --model {mk}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f).get(mk, {})

        bands = [(b, select_cases(data.get(b, []), b, args.per_band))
                 for b in ("success", "partial", "failure")]
        rows = sum(len(v) for _, v in bands)
        if rows == 0:
            print(f"[{mk}] no cases in file")
            continue

        fig, axes = plt.subplots(rows, 2, figsize=(7.2, 3.5 * rows), squeeze=False)
        ri = 0
        drawn = 0
        for band, cases in bands:
            for case in cases:
                q = index.get(case["query_id"])
                a = index.get(case["analog_id"])
                for col, (ent, role, cid) in enumerate(
                        ((q, "QUERY", case["query_id"]), (a, "ANALOG", case["analog_id"]))):
                    ax = axes[ri, col]
                    img = rgb_from_patch(ent, cid) if ent else None
                    if img is None:
                        ax.text(0.5, 0.5, "patch unavailable", ha="center", va="center",
                                fontsize=8, transform=ax.transAxes)
                    else:
                        ax.imshow(img)
                        drawn += 1
                    ax.set_xticks([]); ax.set_yticks([])
                    name = (case["query_name"] if role == "QUERY" else case["analog_name"])
                    eco = (case["query_ecosystem"] if role == "QUERY" else case["analog_ecosystem"])
                    name = name.split(" (Patch")[0]      # drop the sub-crop suffix
                    ax.set_title(f"{role}: {name[:30]}\n[{eco}]", fontsize=7.5)

                colour = {"success": "#15803d", "partial": "#b45309", "failure": "#b91c1c"}[band]
                same = "same category" if case["same_category"] else "DIFFERENT category"
                axes[ri, 0].set_ylabel(band.upper(), fontsize=10, color=colour,
                                       fontweight="bold")
                axes[ri, 1].text(1.02, 0.5,
                                 f"cosine {case['cosine_score']:.4f}\n{same}\n"
                                 f"descriptor\ndisagreement\n{case['descriptor_disagreement']:.3f}",
                                 transform=axes[ri, 1].transAxes, fontsize=7.5,
                                 va="center", ha="left", color=colour)
                ri += 1

        fig.suptitle(f"{SUPPORTED_MODELS[mk]['label']} — retrieval case studies\n"
                     f"green = success · amber = partial · red = failure "
                     f"(RGB contrast-stretched for display only)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 0.88, 0.96])
        out = f"{OUT_DIR}/{mk}_cases.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[{mk}] wrote {out}  ({rows} cases, {drawn}/{rows*2} images rendered)")

    print(f"\nAll panels in {OUT_DIR}/")


if __name__ == "__main__":
    main()
