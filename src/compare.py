"""Compare ROI recovered by PyMC-Marketing and Meridian against the ground truth.

Output: results/roi_comparison.csv and results/roi_comparison.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RUNS = {
    "PyMC (Hill)": "roi_pymc.csv",
    "PyMC (Hill) + lift test": "roi_pymc_calibrated.csv",
    "PyMC (Logistic)": "roi_pymc_logistic.csv",
    "Meridian": "roi_meridian.csv",
    "Meridian + ROI prior": "roi_meridian_calibrated.csv",
}
COLORS = ["#4C78A8", "#9ECAE9", "#B6C9DD", "#F58518", "#FFBF79"]


def main() -> None:
    truth = pd.read_csv(ROOT / "data" / "ground_truth.csv")[["channel", "true_roi"]]
    table = truth.copy()
    # align every run to the ground-truth channel order
    runs = {k: pd.read_csv(RES / v).set_index("channel").loc[truth.channel].reset_index()
            for k, v in RUNS.items() if (RES / v).exists()}
    for name, r in runs.items():
        table[name] = [f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
                       for m, lo, hi in zip(r.roi_mean, r.roi_lo, r.roi_hi)]
    table["true_roi"] = table["true_roi"].round(2)
    table.to_csv(RES / "roi_comparison.csv", index=False)
    print(table.to_string(index=False))

    # Mean absolute error vs truth, per run
    for name, r in runs.items():
        mae = np.mean(np.abs(r.roi_mean.values - truth.true_roi.values))
        covered = np.sum((truth.true_roi.values >= r.roi_lo) & (truth.true_roi.values <= r.roi_hi))
        print(f"{name:28s} MAE={mae:.2f}  truth inside 90% interval: {covered}/{len(truth)} channels")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(truth))
    w = 0.8 / max(len(runs), 1)
    for i, (name, r) in enumerate(runs.items()):
        pos = x - 0.4 + w * (i + 0.5)
        err = [r.roi_mean - r.roi_lo, r.roi_hi - r.roi_mean]
        ax.bar(pos, r.roi_mean, w, yerr=err, capsize=3, label=name, color=COLORS[i])
    ax.scatter(x, truth.true_roi, marker="D", s=70, color="black", zorder=5, label="True ROI")
    ax.set_xticks(x, truth.channel)
    ax.set_ylabel("ROI (revenue per $1 spend)")
    ax.set_title("Recovered channel ROI vs ground truth (90% intervals)")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / "roi_comparison.png", dpi=150)


if __name__ == "__main__":
    main()
