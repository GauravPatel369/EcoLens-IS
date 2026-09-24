"""
EcoLens Step 15 -- ANALOG DISTURBANCE-TRAJECTORY FIGURES  (Phase 4 / C20)

implementation.md Phase 4: "for a handful of query cells, plot the cell's Hansen
loss-year curve against its analogs' curves -- the visual evidence that analogs
share disturbance PATHWAYS, not merely appearance."

That distinction is the whole claim. Two places can look alike in a single
snapshot and have completely unrelated histories; if the analogs' loss curves
track the query's, the embedding is finding something temporal that it was never
shown -- the models see one date per patch and no Hansen data at all.

Selection is deliberately NOT cherry-picked: cells are ranked by
analog_trajectory_jaccard and the figure shows the best, median and worst
together, so a reader sees where the mechanism fails as well as where it works
(implementation.md 9 -- report negative results).

Outputs results/trajectory_figures/*.png plus a summary panel.

Run:
    python 15_analog_trajectory_figure.py
    python 15_analog_trajectory_figure.py --n 6
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

from config import RESULTS_DIR

ANALOG_CSV = f"{RESULTS_DIR}/analog_cell_features.csv"
OUT_DIR = f"{RESULTS_DIR}/trajectory_figures"
YEARS = list(range(2001, 2023))


def _mod(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


def a13():
    return _mod("_a13", "13_analog_risk_features.py")


def explain():
    return _mod("_explain", "09_explainability_engine.py")


def cumulative(loss_years):
    """Cumulative % of the patch lost, by year."""
    if not loss_years:
        return [0.0] * len(YEARS)
    out, run = [], 0.0
    for y in YEARS:
        run += float(loss_years.get(y, loss_years.get(str(y), 0.0)) or 0.0)
        out.append(run)
    return out


def main():
    ap = argparse.ArgumentParser(description="EcoLens 15: analog trajectory figures (C20)")
    ap.add_argument("--n", type=int, default=6, help="cells to plot")
    ap.add_argument("--model", default="prithvi")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--exclusion-km", type=float, default=250.0)
    args = ap.parse_args()

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(ANALOG_CSV):
        raise SystemExit(f"{ANALOG_CSV} not found -- run 13_analog_risk_features.py first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    m = a13()
    ids, bases, plons, plats, pecos, P = m.load_analog_pool(args.model)
    z = np.load(f"{RESULTS_DIR}/analog_cell_embeddings.npz", allow_pickle=True)

    df = pd.read_csv(ANALOG_CSV)
    cells = (df.drop_duplicates(["cell_lon", "cell_lat"])
               [["region_id", "cell_lon", "cell_lat", "analog_trajectory_jaccard"]]
               .dropna(subset=["analog_trajectory_jaccard"])
               .sort_values("analog_trajectory_jaccard", ascending=False)
               .reset_index(drop=True))
    if cells.empty:
        raise SystemExit("no cells with a trajectory jaccard")

    # best / median / worst -- NOT cherry-picked
    n = min(args.n, len(cells))
    picks = []
    picks += list(range(min(2, n)))                                    # best
    mid = len(cells) // 2
    picks += [mid, mid + 1][: max(0, min(2, n - len(picks)))]          # median
    picks += [len(cells) - 1, len(cells) - 2][: max(0, n - len(picks))]  # worst
    picks = [p for p in dict.fromkeys(picks) if 0 <= p < len(cells)][:n]

    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 3.0 * len(picks)), squeeze=False)
    summary = []

    for ax, pi in zip(axes[:, 0], picks):
        row = cells.iloc[pi]
        lon, lat = float(row.cell_lon), float(row.cell_lat)
        key = f"{lon:.5f}_{lat:.5f}"
        if key not in z.files:
            continue
        v = z[key]

        same = bases == row.region_id
        d = np.array([m.haversine_km(lon, lat, lo, la) for lo, la in zip(plons, plats)])
        allowed = ~(same | (d < args.exclusion_km))
        sims = P[allowed] @ v
        allowed_idx = np.where(allowed)[0]
        # one analog per LOCATION -- mirrors 13; a naive top-k returns k sub-crops
        # of the same place (this figure is what revealed that).
        best_per_base = {}
        for rank in np.argsort(-sims):
            b = bases[allowed_idx[rank]]
            if b not in best_per_base:
                best_per_base[b] = rank
            if len(best_per_base) >= args.top_k:
                break
        order = np.array(sorted(best_per_base.values(), key=lambda k: -sims[k]))
        idx = allowed_idx[order]

        cell_curve = cumulative(explain().get_disturbance_history(lon, lat).get("loss_years"))
        ax.plot(YEARS, cell_curve, color="#111", lw=2.6, marker="o", ms=3,
                label=f"QUERY CELL ({row.region_id})", zorder=5)

        for rank, j in enumerate(idx):
            h = explain().get_disturbance_history(float(plons[j]), float(plats[j]))
            ax.plot(YEARS, cumulative(h.get("loss_years")), lw=1.4, alpha=0.85,
                    label=f"analog {rank+1}: {bases[j]} ({pecos[j]}, "
                          f"{d[j]:,.0f} km, cos {sims[order[rank]]:.3f})")

        band = ("BEST" if pi < 2 else "WORST" if pi >= len(cells) - 2 else "MEDIAN")
        ax.set_title(f"[{band}] {row.region_id} @ ({lon:.3f}, {lat:.3f})  "
                     f"trajectory Jaccard = {row.analog_trajectory_jaccard:.3f}",
                     fontsize=10, loc="left")
        ax.set_ylabel("cumulative % lost")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
        summary.append({"region": row.region_id, "lon": lon, "lat": lat, "band": band,
                        "jaccard": float(row.analog_trajectory_jaccard),
                        "analogs": [str(bases[j]) for j in idx],
                        "analog_km": [float(d[j]) for j in idx]})

    axes[-1, 0].set_xlabel("year")
    fig.suptitle("Do ecological analogs share DISTURBANCE PATHWAYS?\n"
                 "Hansen cumulative tree-cover loss: query cell (black) vs its "
                 "top-5 analogs from other landscapes", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{OUT_DIR}/analog_trajectories.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    with open(f"{OUT_DIR}/analog_trajectories.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT_DIR}/analog_trajectories.json")
    print(f"\njaccard distribution across {len(cells)} cells: "
          f"min {cells.analog_trajectory_jaccard.min():.3f} / "
          f"median {cells.analog_trajectory_jaccard.median():.3f} / "
          f"max {cells.analog_trajectory_jaccard.max():.3f}")


if __name__ == "__main__":
    main()
