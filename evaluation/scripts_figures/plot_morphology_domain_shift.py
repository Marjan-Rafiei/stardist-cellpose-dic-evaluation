"""Plot paired MI05 morphology-domain shift summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from code.unified_evaluation import FEATURE_NAMES, sha256_file
from scripts_figures.provenance import write_figure_manifest


SYSTEM_ORDER = ("stardist_saved", "cellpose_auto", "cellpose_fixed")
DISPLAY_NAMES = {
    "stardist_saved": "StarDist",
    "cellpose_auto": "Cellpose automatic",
    "cellpose_fixed": "Cellpose fixed",
}
FEATURE_LABELS = {
    "area": "Area",
    "perimeter": "Perimeter",
    "eccentricity": "Eccentricity",
    "solidity": "Solidity",
    "extent": "Extent",
    "major_axis_length": "Major-axis length",
    "minor_axis_length": "Minor-axis length",
    "aspect_ratio": "Aspect ratio",
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
            "Creator": "scripts_figures/plot_morphology_domain_shift.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(outputs[1], dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return outputs


def annotate_heatmap(axis: mpl.axes.Axes, matrix: np.ndarray, formatter: str) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                format(matrix[row, column], formatter),
                ha="center",
                va="center",
                color="#222222",
                fontsize=8.2,
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    summary_path = Path(args.summary).resolve()
    manifest_path = Path(args.analysis_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_input(summary_path, manifest)
    summary = pd.read_csv(summary_path)

    relative_matrix = np.zeros((len(FEATURE_NAMES), len(SYSTEM_ORDER)), dtype=float)
    positive_matrix = np.zeros_like(relative_matrix)
    for row_index, feature in enumerate(FEATURE_NAMES):
        for column_index, system in enumerate(SYSTEM_ORDER):
            selected = summary.loc[
                (summary["system"] == system) & (summary["feature"] == feature)
            ]
            if len(selected) != 1:
                raise ValueError(f"Expected one summary row for {system} and {feature}")
            relative_matrix[row_index, column_index] = selected.iloc[0][
                "median_relative_difference_percent"
            ]
            positive_matrix[row_index, column_index] = 100 * selected.iloc[0][
                "positive_fraction"
            ]

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9), constrained_layout=True)
    relative_axis, direction_axis = axes
    relative_norm = TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    relative_image = relative_axis.imshow(
        relative_matrix, cmap="PuOr_r", norm=relative_norm, aspect="auto"
    )
    direction_norm = TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
    direction_image = direction_axis.imshow(
        positive_matrix, cmap="PuOr_r", norm=direction_norm, aspect="auto"
    )

    feature_labels = [FEATURE_LABELS[feature] for feature in FEATURE_NAMES]
    system_labels = [DISPLAY_NAMES[system] for system in SYSTEM_ORDER]
    for axis in axes:
        axis.set_xticks(range(len(system_labels)), system_labels, rotation=20, ha="right")
        axis.set_yticks(range(len(feature_labels)), feature_labels)
        axis.set_xticks(np.arange(-0.5, len(system_labels), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(feature_labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.6)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)

    relative_axis.set_title("Median paired relative shift")
    direction_axis.set_title("Pairs with predicted value above GT")
    annotate_heatmap(relative_axis, relative_matrix, "+.1f")
    annotate_heatmap(direction_axis, positive_matrix, ".0f")
    relative_colorbar = fig.colorbar(relative_image, ax=relative_axis, shrink=0.88, pad=0.03)
    relative_colorbar.set_label("Median relative difference (%)")
    direction_colorbar = fig.colorbar(direction_image, ax=direction_axis, shrink=0.88, pad=0.03)
    direction_colorbar.set_label("Positive paired differences (%)")

    fig.suptitle("MI05 morphology domain shift in matched instances", y=1.03, fontsize=12)
    fig.text(
        0.5,
        -0.035,
        "GT denotes ground-truth-mask morphology. Values are descriptive within MI05; matched objects are not independent experiments.",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = save_figure(fig, output_dir / "mi05_morphology_domain_shift")
    write_figure_manifest(
        output_dir=output_dir,
        script_path=Path(__file__).resolve(),
        inputs=(summary_path, manifest_path),
        outputs=outputs,
        notes="Paired matched-instance morphology shifts. Relative eccentricity changes can be sensitive to small ground-truth values; raw signed and absolute differences are retained in verified outputs.",
    )


if __name__ == "__main__":
    main()
