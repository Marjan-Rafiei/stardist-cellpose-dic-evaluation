"""Render correlation and permutation-importance audit figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code.unified_evaluation import FEATURE_NAMES
from scripts_figures.provenance import write_figure_manifest


DISPLAY_NAMES = {
    "area": "Area",
    "perimeter": "Perimeter",
    "eccentricity": "Eccentricity",
    "solidity": "Solidity",
    "extent": "Extent",
    "major_axis_length": "Major axis",
    "minor_axis_length": "Minor axis",
    "aspect_ratio": "Aspect ratio",
    "elongation": "Elongation",
    "size": "Size",
    "compactness": "Compactness",
}


def annotate_bars(axis: plt.Axes, values: np.ndarray) -> None:
    offset = max(float(np.max(values)) * 0.02, 0.002)
    for index, value in enumerate(values):
        axis.text(value + offset, index, f"{value:.3f}", va="center", fontsize=8)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correlations", required=True)
    parser.add_argument("--individual", required=True)
    parser.add_argument("--grouped", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    correlations_path = Path(args.correlations).resolve()
    individual_path = Path(args.individual).resolve()
    grouped_path = Path(args.grouped).resolve()
    analysis_manifest_path = Path(args.analysis_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    correlations = pd.read_csv(correlations_path)
    individual = pd.read_csv(individual_path).sort_values(
        ["importance_mean", "feature"], ascending=[True, True], kind="stable"
    )
    grouped = pd.read_csv(grouped_path).sort_values(
        ["importance_mean", "group"], ascending=[True, True], kind="stable"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=True, sharey=True)
    image = None
    mask = np.triu(np.ones((len(FEATURE_NAMES), len(FEATURE_NAMES)), dtype=bool), k=1)
    for axis, split, title in zip(
        axes,
        ("training", "validation"),
        ("MI01 to MI03 training", "MI04 validation"),
    ):
        subset = correlations[correlations["split"] == split]
        matrix = subset.pivot(
            index="feature_a", columns="feature_b", values="spearman_correlation"
        ).loc[list(FEATURE_NAMES), list(FEATURE_NAMES)]
        values = matrix.to_numpy(dtype=float)
        shown = np.ma.array(values, mask=mask)
        image = axis.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1)
        for row in range(len(FEATURE_NAMES)):
            for column in range(row + 1):
                value = values[row, column]
                color = "white" if abs(value) >= 0.65 else "#202020"
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
        labels = [DISPLAY_NAMES[feature] for feature in FEATURE_NAMES]
        axis.set_xticks(range(len(labels)), labels=labels, rotation=50, ha="right")
        axis.set_yticks(range(len(labels)), labels=labels)
        axis.set_title(title)
        axis.set_xlim(-0.5, len(labels) - 0.5)
        axis.set_ylim(len(labels) - 0.5, -0.5)
    if image is None:
        raise ValueError("No correlation matrix was rendered")
    fig.subplots_adjust(left=0.12, right=0.85, bottom=0.25, top=0.85, wspace=0.16)
    colorbar_axis = fig.add_axes([0.88, 0.20, 0.018, 0.64])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Spearman correlation")
    fig.suptitle("Morphology-feature correlation structure")
    correlation_png = output_dir / "feature_correlation_heatmaps.png"
    correlation_pdf = output_dir / "feature_correlation_heatmaps.pdf"
    fig.savefig(correlation_png, dpi=200, bbox_inches="tight")
    fig.savefig(
        correlation_pdf,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    individual_values = individual["importance_mean"].to_numpy(dtype=float)
    axes[0].barh(
        [DISPLAY_NAMES[item] for item in individual["feature"]],
        individual_values,
        xerr=individual["importance_std"],
        color="#2A6F97",
        edgecolor="#173F5F",
        linewidth=0.6,
        error_kw={"ecolor": "#202020", "elinewidth": 0.8, "capsize": 2},
    )
    axes[0].axvline(0, color="#202020", linewidth=0.8)
    axes[0].set_title("Individual-feature permutation")
    axes[0].set_xlabel("Decrease in MI04 balanced accuracy")
    annotate_bars(axes[0], individual_values)

    grouped_values = grouped["importance_mean"].to_numpy(dtype=float)
    axes[1].barh(
        [DISPLAY_NAMES[item] for item in grouped["group"]],
        grouped_values,
        xerr=grouped["importance_std"],
        color="#C78D2A",
        edgecolor="#6F4E16",
        linewidth=0.6,
        error_kw={"ecolor": "#202020", "elinewidth": 0.8, "capsize": 2},
    )
    axes[1].axvline(0, color="#202020", linewidth=0.8)
    axes[1].set_title("Joint group permutation")
    axes[1].set_xlabel("Decrease in MI04 balanced accuracy")
    annotate_bars(axes[1], grouped_values)
    upper = max(individual_values.max() + individual["importance_std"].max(), grouped_values.max() + grouped["importance_std"].max())
    for axis in axes:
        axis.set_xlim(min(-0.01, axis.get_xlim()[0]), upper * 1.18)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
    fig.suptitle("MI04 permutation-importance stability audit")
    fig.text(
        0.5,
        0.01,
        "Bars are means across 100 shuffles from five seeded runs; error bars are shuffle SD, not confidence intervals.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.84, wspace=0.38)
    importance_png = output_dir / "permutation_importance_stability_and_groups.png"
    importance_pdf = output_dir / "permutation_importance_stability_and_groups.pdf"
    fig.savefig(importance_png, dpi=200, bbox_inches="tight")
    fig.savefig(
        importance_pdf,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    write_figure_manifest(
        output_dir,
        Path(__file__),
        (correlations_path, individual_path, grouped_path, analysis_manifest_path),
        (correlation_png, correlation_pdf, importance_png, importance_pdf),
        "MI04 feature-importance audit. Error bars show shuffle variability, not confidence intervals; importance is predictive and noncausal.",
    )


if __name__ == "__main__":
    main()
