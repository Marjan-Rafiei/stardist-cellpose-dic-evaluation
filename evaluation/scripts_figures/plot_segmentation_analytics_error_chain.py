"""Plot the verified MI05 segmentation-to-analytics accounting chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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


def verify_input(path: Path, manifest: dict[str, object]) -> None:
    outputs = {
        Path(item["path"]).resolve(): item["sha256"]
        for item in manifest["outputs"]
    }
    expected = outputs.get(path.resolve())
    if expected is None:
        raise ValueError(f"Input is not declared by the analysis manifest: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Input hash mismatch for {path}: {actual} != {expected}")


def save_figure(fig: mpl.figure.Figure, base: Path) -> list[Path]:
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    fig.savefig(
        outputs[0],
        bbox_inches="tight",
        metadata={
            "Creator": "scripts_figures/plot_segmentation_analytics_error_chain.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(outputs[1], dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--ratio-bridge", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    summary_path = Path(args.summary).resolve()
    bridge_path = Path(args.ratio_bridge).resolve()
    manifest_path = Path(args.analysis_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in (summary_path, bridge_path):
        verify_input(path, manifest)

    summary = pd.read_csv(summary_path)
    bridge = pd.read_csv(bridge_path)
    if summary["system"].tolist() != list(SYSTEM_ORDER):
        raise ValueError("Unexpected system order")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8.3,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    count_axis, ratio_axis = axes
    x = np.arange(len(SYSTEM_ORDER))
    width = 0.28
    count_axis.bar(
        x - width / 2,
        -summary["false_negatives"],
        width=width,
        color="#4477AA",
        label="False negatives",
    )
    count_axis.bar(
        x + width / 2,
        summary["false_positives"],
        width=width,
        color="#EE8844",
        label="False positives",
    )
    count_axis.scatter(
        x,
        summary["total_count_error"],
        marker="D",
        color="#222222",
        s=34,
        zorder=3,
        label="Net count error",
    )
    count_axis.axhline(0, color="#555555", linewidth=0.9)
    count_axis.set_xticks(x, [DISPLAY_NAMES[system] for system in SYSTEM_ORDER])
    count_axis.set_ylabel("Instance contribution")
    count_axis.set_title("Detection errors propagate to total count")
    count_axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    count_axis.legend(frameon=False, loc="lower left")

    stage_labels = (
        "Ground\ntruth",
        "Matched\nGT only",
        "Add false\npositives",
        "GT-shape\nclassifier",
        "Pred-shape\nclassifier",
    )
    for system in SYSTEM_ORDER:
        subset = bridge.loc[bridge["system"] == system].sort_values("stage_order")
        ratio_axis.plot(
            subset["stage_order"],
            100 * subset["mitotic_ratio"],
            color=COLORS[system],
            linestyle=LINE_STYLES[system],
            marker=MARKERS[system],
            markersize=5,
            linewidth=1.8,
            label=DISPLAY_NAMES[system],
        )
    ratio_axis.set_xticks(range(len(stage_labels)), stage_labels)
    ratio_axis.set_ylabel("Mitotic annotation ratio (%)")
    ratio_axis.set_title("Sequential ratio accounting bridge")
    ratio_axis.set_ylim(0, 86)
    ratio_axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ratio_axis.legend(frameon=False, loc="upper right")

    fig.suptitle("MI05 segmentation-to-analytics error chain", y=1.04, fontsize=12)
    fig.text(
        0.5,
        -0.055,
        "False negatives and positives are unmatched instances at IoU 0.50. The ratio bridge is path-dependent accounting, not a causal model.",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = save_figure(fig, output_dir / "mi05_segmentation_analytics_error_chain")
    write_figure_manifest(
        output_dir=output_dir,
        script_path=Path(__file__).resolve(),
        inputs=(summary_path, bridge_path, manifest_path),
        outputs=outputs,
        notes="Verified MI05 accounting chain. Ratio steps are sequential and path-dependent; no biological causality is inferred.",
    )


if __name__ == "__main__":
    main()
