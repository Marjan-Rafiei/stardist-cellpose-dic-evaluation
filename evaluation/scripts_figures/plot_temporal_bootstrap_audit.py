"""Plot MI05 temporal dependence and bootstrap interval sensitivity."""

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
            "Creator": "scripts_figures/plot_temporal_bootstrap_audit.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(outputs[1], dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autocorrelation", required=True)
    parser.add_argument("--intervals", required=True)
    parser.add_argument("--analysis-summary", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    acf_path = Path(args.autocorrelation).resolve()
    interval_path = Path(args.intervals).resolve()
    summary_path = Path(args.analysis_summary).resolve()
    manifest_path = Path(args.analysis_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in (acf_path, interval_path, summary_path):
        verify_input(path, manifest)

    acf = pd.read_csv(acf_path)
    intervals = pd.read_csv(interval_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    chosen = int(summary["bootstrap"]["chosen_block_length"])

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

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    acf_axis, interval_axis = axes
    for system in SYSTEM_ORDER:
        subset = acf.loc[(acf["system"] == system) & (acf["quantity"] == "f1")]
        acf_axis.plot(
            subset["lag_frames"],
            subset["autocorrelation"],
            color=COLORS[system],
            linestyle=LINE_STYLES[system],
            marker=MARKERS[system],
            markevery=2,
            markersize=4,
            linewidth=1.7,
            label=DISPLAY_NAMES[system],
        )
    acf_axis.axhline(0, color="#555555", linewidth=0.9)
    acf_axis.axhline(0.2, color="#888888", linewidth=0.9, linestyle=":")
    acf_axis.set_title("Per-frame detection F1 autocorrelation")
    acf_axis.set_xlabel("Lag (frames)")
    acf_axis.set_ylabel("Autocorrelation")
    acf_axis.set_xticks(range(0, int(acf["lag_frames"].max()) + 1, 3))
    acf_axis.set_ylim(-0.35, 1.05)
    acf_axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)

    f1 = intervals.loc[intervals["metric"] == "f1"]
    blocks = sorted(f1["block_length"].unique())
    offsets = dict(zip(SYSTEM_ORDER, (-0.20, 0.0, 0.20), strict=True))
    for system in SYSTEM_ORDER:
        subset = f1.loc[f1["system"] == system].sort_values("block_length")
        x = subset["block_length"].to_numpy(dtype=float) + offsets[system]
        y = subset["point_estimate"].to_numpy(dtype=float)
        lower = y - subset["percentile_2_5"].to_numpy(dtype=float)
        upper = subset["percentile_97_5"].to_numpy(dtype=float) - y
        interval_axis.errorbar(
            x,
            y,
            yerr=np.vstack((lower, upper)),
            color=COLORS[system],
            linestyle="none",
            marker=MARKERS[system],
            markersize=5,
            capsize=3,
            elinewidth=1.5,
            label=DISPLAY_NAMES[system],
        )
    interval_axis.axvline(chosen, color="#555555", linewidth=1.0, linestyle=":")
    interval_axis.set_title("Pooled F1 percentile intervals")
    interval_axis.set_xlabel("Moving-block length (frames)")
    interval_axis.set_ylabel("Detection F1")
    interval_axis.set_xticks(blocks)
    interval_axis.set_ylim(0.2, 0.9)
    interval_axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)

    handles, labels = acf_axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("MI05 temporal dependence and bootstrap sensitivity", y=1.12, fontsize=12)
    fig.text(
        0.5,
        -0.035,
        f"Dotted vertical line: selected {chosen}-frame block. Intervals describe within-MI05 temporal variability only.",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = save_figure(fig, output_dir / "mi05_temporal_bootstrap_sensitivity")
    write_figure_manifest(
        output_dir=output_dir,
        script_path=Path(__file__).resolve(),
        inputs=(acf_path, interval_path, summary_path, manifest_path),
        outputs=outputs,
        notes="Descriptive MI05 temporal diagnostics. Intervals are not uncertainty across independent experiments or cell lines.",
    )


if __name__ == "__main__":
    main()
