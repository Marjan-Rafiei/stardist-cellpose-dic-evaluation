"""Render MI05 per-frame count and mitotic-ratio agreement diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts_figures.provenance import write_figure_manifest


SYSTEM_ORDER = ("stardist_saved", "cellpose_auto", "cellpose_fixed")
DISPLAY_NAMES = {
    "stardist_saved": "StarDist",
    "cellpose_auto": "Cellpose auto",
    "cellpose_fixed": "Cellpose fixed",
}
COLORS = {
    "stardist_saved": "#2A6F97",
    "cellpose_auto": "#C76D2A",
    "cellpose_fixed": "#7A6F2B",
}
MARKERS = {"stardist_saved": "o", "cellpose_auto": "s", "cellpose_fixed": "^"}
LINESTYLES = {"stardist_saved": "-", "cellpose_auto": "--", "cellpose_fixed": ":"}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--by-frame", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    by_frame_path = Path(args.by_frame).resolve()
    summary_path = Path(args.summary).resolve()
    manifest_path = Path(args.analysis_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(by_frame_path)
    summary = pd.read_csv(summary_path).set_index("system")
    if set(frame["system"]) != set(SYSTEM_ORDER) or set(summary.index) != set(SYSTEM_ORDER):
        raise ValueError("Inputs do not contain exactly the three approved systems")

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.5))
    count_scatter, ratio_scatter = axes[0]
    count_time, ratio_time = axes[1]

    for system in SYSTEM_ORDER:
        subset = frame[frame["system"] == system].sort_values("frame_index")
        count_scatter.scatter(
            subset["gt_total"],
            subset["pred_total"],
            s=24,
            facecolors="none",
            edgecolors=COLORS[system],
            marker=MARKERS[system],
            linewidths=0.9,
            alpha=0.78,
            label=DISPLAY_NAMES[system],
        )
        ratio_scatter.scatter(
            100 * subset["gt_mitotic_ratio"],
            100 * subset["pred_mitotic_ratio"],
            s=24,
            facecolors="none",
            edgecolors=COLORS[system],
            marker=MARKERS[system],
            linewidths=0.9,
            alpha=0.78,
            label=DISPLAY_NAMES[system],
        )
        count_time.plot(
            subset["frame_index"],
            subset["count_error"],
            color=COLORS[system],
            linestyle=LINESTYLES[system],
            linewidth=1.2,
            marker=MARKERS[system],
            markersize=2.8,
            markevery=4,
            label=DISPLAY_NAMES[system],
        )
        ratio_time.plot(
            subset["frame_index"],
            100 * subset["ratio_signed_error"],
            color=COLORS[system],
            linestyle=LINESTYLES[system],
            linewidth=1.2,
            marker=MARKERS[system],
            markersize=2.8,
            markevery=4,
            label=DISPLAY_NAMES[system],
        )

    count_min = min(frame["gt_total"].min(), frame["pred_total"].min())
    count_max = max(frame["gt_total"].max(), frame["pred_total"].max())
    count_scatter.plot([count_min, count_max], [count_min, count_max], color="#303030", linewidth=1, label="Identity")
    count_scatter.set_xlim(count_min - 0.5, count_max + 0.5)
    count_scatter.set_ylim(count_min - 0.5, count_max + 0.5)
    count_scatter.set_aspect("equal", adjustable="box")
    count_scatter.set_xlabel("Ground-truth count per frame")
    count_scatter.set_ylabel("Predicted count per frame")
    count_scatter.set_title("A. Count agreement")

    ratio_max = 100 * max(frame["gt_mitotic_ratio"].max(), frame["pred_mitotic_ratio"].max())
    ratio_limit = max(50.0, float(np.ceil(ratio_max / 10.0) * 10.0))
    ratio_scatter.plot([0, ratio_limit], [0, ratio_limit], color="#303030", linewidth=1, label="Identity")
    ratio_scatter.set_xlim(-2, ratio_limit + 2)
    ratio_scatter.set_ylim(-2, ratio_limit + 2)
    ratio_scatter.set_aspect("equal", adjustable="box")
    ratio_scatter.set_xlabel("Ground-truth mitotic annotation ratio (%)")
    ratio_scatter.set_ylabel("Predicted mitotic annotation ratio (%)")
    ratio_scatter.set_title("B. Ratio agreement")

    count_time.axhline(0, color="#303030", linewidth=0.9)
    count_time.set_xlabel("MI05 frame index")
    count_time.set_ylabel("Predicted minus ground-truth count")
    count_time.set_title("C. Count error over the sequence")

    ratio_time.axhspan(-5, 5, color="#D9D9D9", alpha=0.55, zorder=0)
    ratio_time.axhline(0, color="#303030", linewidth=0.9)
    ratio_time.set_xlabel("MI05 frame index")
    ratio_time.set_ylabel("Predicted minus ground-truth ratio (percentage points)")
    ratio_time.set_title("D. Ratio error over the sequence")

    for axis in (count_time, ratio_time):
        for boundary in (23.5, 46.5):
            axis.axvline(boundary, color="#9B9B9B", linewidth=0.7, linestyle="--")
        axis.set_xlim(1, 69)
        axis.set_xticks([1, 12, 23, 35, 46, 58, 69])
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    ratio_time.text(2, 6.5, "5 pp closeness band", color="#666666", fontsize=8)

    for axis in axes.flat:
        axis.grid(color="#E2E2E2", linewidth=0.6)
        axis.set_axisbelow(True)
    handles, labels = count_scatter.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("MI05 per-frame cell-count and mitotic-ratio validation", fontsize=14, y=0.995)
    fig.text(
        0.5,
        0.018,
        "Dashed vertical lines separate descriptive temporal thirds. Correlation is supplementary; identity and error plots assess agreement.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.89, hspace=0.33, wspace=0.28)

    png_path = output_dir / "mi05_count_ratio_agreement.png"
    pdf_path = output_dir / "mi05_count_ratio_agreement.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    write_figure_manifest(
        output_dir,
        Path(__file__),
        (by_frame_path, summary_path, manifest_path),
        (png_path, pdf_path),
        "MI05 descriptive per-frame agreement. The five-percentage-point band is an operational case-finding tolerance, not a biological accuracy standard.",
    )


if __name__ == "__main__":
    main()
