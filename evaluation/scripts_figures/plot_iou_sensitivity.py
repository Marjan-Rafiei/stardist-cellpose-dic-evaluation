"""Plot MI05 matching-threshold sensitivity from verified metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from code.unified_evaluation import sha256_file
from scripts_figures.provenance import write_figure_manifest


SYSTEM_ORDER = ("stardist_saved", "cellpose_auto", "cellpose_fixed")
DISPLAY_NAMES = {
    "stardist_saved": "StarDist",
    "cellpose_auto": "Cellpose automatic",
    "cellpose_fixed": "Cellpose fixed",
}
COLORS = {
    "stardist_saved": "#0072B2",
    "cellpose_auto": "#D55E00",
    "cellpose_fixed": "#009E73",
}
LINE_STYLES = {
    "stardist_saved": "-",
    "cellpose_auto": "--",
    "cellpose_fixed": "-.",
}
MARKERS = {
    "stardist_saved": "o",
    "cellpose_auto": "s",
    "cellpose_fixed": "^",
}


def verify_metrics(metrics_path: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = {Path(item["path"]).resolve(): item["sha256"] for item in manifest["outputs"]}
    expected = outputs.get(metrics_path.resolve())
    if expected is None:
        raise ValueError(f"Metrics file is not declared in {manifest_path}")
    actual = sha256_file(metrics_path)
    if actual != expected:
        raise ValueError(f"Metrics hash mismatch: {actual} != {expected}")


def save_figure(fig: mpl.figure.Figure, base: Path) -> list[Path]:
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    fig.savefig(
        outputs[0],
        bbox_inches="tight",
        metadata={"Creator": "scripts_figures/plot_iou_sensitivity.py", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(outputs[1], dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    metrics_path = Path(args.metrics).resolve()
    manifest_path = Path(args.analysis_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_metrics(metrics_path, manifest_path)

    frame = pd.read_csv(metrics_path)
    if list(frame["system"].drop_duplicates()) != list(SYSTEM_ORDER):
        raise ValueError("Unexpected method order in sensitivity metrics")
    thresholds = sorted(frame["iou_threshold"].unique())

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), sharex=True, constrained_layout=True)
    panels = (("f1", "Detection F1"), ("pq", "Panoptic quality"))
    for axis, (column, label) in zip(axes, panels, strict=True):
        for system in SYSTEM_ORDER:
            subset = frame.loc[frame["system"] == system].sort_values("iou_threshold")
            axis.plot(
                subset["iou_threshold"],
                subset[column],
                color=COLORS[system],
                linestyle=LINE_STYLES[system],
                marker=MARKERS[system],
                markersize=5,
                linewidth=1.8,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=DISPLAY_NAMES[system],
            )
        axis.axvline(0.5, color="#666666", linewidth=1.0, linestyle=":", zorder=0)
        axis.set_title(label)
        axis.set_xlabel("Matching IoU threshold")
        axis.set_ylabel("Score")
        axis.set_xticks(thresholds)
        axis.set_xlim(min(thresholds) - 0.015, max(thresholds) + 0.015)
        axis.set_ylim(0, 0.9)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("MI05 matching-threshold sensitivity", y=1.13, fontsize=12)
    fig.text(0.5, -0.035, "Dotted vertical line: primary IoU threshold 0.50", ha="center", fontsize=8, color="#555555")

    outputs = save_figure(fig, output_dir / "mi05_iou_sensitivity")
    write_figure_manifest(
        output_dir=output_dir,
        script_path=Path(__file__).resolve(),
        inputs=(metrics_path, manifest_path),
        outputs=outputs,
        notes="Frozen MI05 predictions; threshold sensitivity only. Colors and redundant line styles follow the thesis method identity.",
    )


if __name__ == "__main__":
    main()
