"""Create report-ready figures from the saved raw results."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("results")
COLORS = {"FCFS": "#a7b3c4", "SCAN": "#5581ac", "C-SCAN": "#7a67a8",
          "SSTF": "#3d8b7a", "selector": "#e68640", "oracle": "#263947"}


def rows(name):
    with (ROOT / f"{name}_batches.csv").open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def polish(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e3e8ee", linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ROOT.mkdir(exist_ok=True)
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    timeline = sorted(rows("timeline"), key=lambda row: int(row["batch"]))
    test = rows("test")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "white", "axes.titlesize": 14})

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    x = np.arange(1, 61)
    for name, key in (("FCFS", "fcfs_seek"), ("SSTF", "sstf_seek"),
                      ("selector", "selector_seek"), ("oracle", "oracle_seek")):
        values = np.array([int(row[key]) for row in timeline])
        smooth = np.convolve(values, np.ones(5) / 5, mode="valid")
        ax.plot(x[4:], smooth, label=name, color=COLORS[name],
                linewidth=2.4 if name == "selector" else 1.8)
    ax.axvline(30.5, color="#ba4b57", linestyle="--", linewidth=1.6)
    ax.text(31.2, ax.get_ylim()[1] * 0.78, "pattern shift", color="#ba4b57", va="top")
    ax.set(title="Head movement changes when the request pattern shifts",
           xlabel="Batch number (5-batch moving average)", ylabel="Seek distance / batch (cylinders)")
    ax.legend(ncol=4, frameon=False, loc="upper left")
    polish(ax)
    fig.savefig(ROOT / "shift_timeline.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True, sharex=True,
                             layout="constrained")
    names = ["FCFS", "SCAN", "C-SCAN", "SSTF", "selector", "oracle"]
    for ax, phase in zip(axes, ("before", "after")):
        totals = summary["timeline"]["phases"][phase]["seek_totals"]
        vals = [totals[name] / 30 for name in names]
        ax.barh(names[::-1], vals[::-1], color=[COLORS[name] for name in names[::-1]])
        ax.set_title("Sequential" if phase == "before" else "Bursty")
        ax.set_xlim(0, 560)
        polish(ax)
    fig.suptitle("Same schedulers, different request patterns", fontsize=14)
    fig.supxlabel("Mean seek distance / batch (cylinders)")
    fig.savefig(ROOT / "phase_comparison.png", dpi=180)
    plt.close(fig)

    conf = np.array([float(row["confidence"]) for row in test])
    correct = np.array([int(row["correct"]) for row in test])
    edges = np.linspace(0, 1, 6)
    centers, observed, counts = [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (conf >= left) & (conf < right if right < 1 else conf <= right)
        if in_bin.any():
            centers.append(float(conf[in_bin].mean()))
            observed.append(float(correct[in_bin].mean()))
            counts.append(int(in_bin.sum()))
    fig, ax = plt.subplots(figsize=(5.2, 4.7), layout="constrained")
    ax.plot([0, 1], [0, 1], color="#a7b3c4", linestyle="--", label="Perfect calibration")
    ax.scatter(centers, observed, s=np.array(counts) * 2 + 25, color=COLORS["selector"],
               edgecolor="white", linewidth=1.2, zorder=3, label="Held-out batches")
    for c, o, n in zip(centers, observed, counts):
        ax.annotate(str(n), (c, o), xytext=(7, 4), textcoords="offset points", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence",
           ylabel="Observed selection accuracy", title="Is selector confidence reliable?")
    ax.legend(frameon=False, loc="upper left")
    polish(ax)
    fig.savefig(ROOT / "confidence_calibration.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
