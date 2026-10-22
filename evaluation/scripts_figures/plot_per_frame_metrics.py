"""Plot per-frame MI05 metrics and counts from verified result files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from scripts_figures.provenance import write_figure_manifest


COLORS = {"stardist_saved": "#0072B2", "cellpose_auto": "#D55E00", "cellpose_fixed": "#009E73"}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.results_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    input_paths = []
    for system in sorted(COLORS):
        path = root / system / "metrics_by_frame.csv"
        if path.is_file():
            input_paths.append(path)
            item = pd.read_csv(path)
            item.insert(0, "system", system)
            item["frame_number"] = item["frame_id"].str.rsplit("_", n=1).str[-1].astype(int)
            frames.append(item)
    if not frames:
        raise ValueError("No verified per-frame metrics found")
    data = pd.concat(frames, ignore_index=True).sort_values(["system", "frame_number"], kind="stable")
    data.to_csv(output / "per_frame_plot_data.csv", index=False, lineterminator="\n", float_format="%.10g")

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for system, group in data.groupby("system", sort=True):
        color = COLORS.get(system, None)
        axes[0].plot(group["frame_number"], group["f1"], label=system, color=color)
        axes[1].plot(group["frame_number"], group["pq"], label=system, color=color)
        axes[2].plot(group["frame_number"], group["pred_total"], label=f"{system} predicted", color=color)
        axes[3].plot(group["frame_number"], group["pred_mitotic_ratio"], label=system, color=color)
    reference = data.drop_duplicates("frame_number")
    axes[2].plot(reference["frame_number"], reference["gt_total"], label="ground truth", color="#000000", linestyle="--")
    axes[3].plot(reference["frame_number"], reference["gt_mitotic_ratio"], label="ground truth", color="#000000", linestyle="--")
    for axis, ylabel in zip(axes, ("Instance F1", "Panoptic quality", "Instance count", "Mitotic annotation ratio")):
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("MI05 frame")
    fig.suptitle("Per-frame evaluation on MI05")
    fig.tight_layout()
    fig.savefig(output / "per_frame_metrics.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "per_frame_metrics.pdf", bbox_inches="tight")
    plt.close(fig)
    write_figure_manifest(
        output,
        Path(__file__),
        input_paths,
        (output / "per_frame_plot_data.csv", output / "per_frame_metrics.png", output / "per_frame_metrics.pdf"),
        "Per-frame plots generated only from verified metrics_by_frame.csv files.",
    )


if __name__ == "__main__":
    main()
