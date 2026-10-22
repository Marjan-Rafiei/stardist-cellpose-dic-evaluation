"""Plot matched-instance classification correctness transition matrices."""

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
MATRIX_POSITIONS = {
    "correct_to_correct": (0, 0),
    "correct_to_incorrect": (0, 1),
    "incorrect_to_correct": (1, 0),
    "incorrect_to_incorrect": (1, 1),
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
            "Creator": "scripts_figures/plot_classification_transitions.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(outputs[1], dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    transitions_path = Path(args.transitions).resolve()
    manifest_path = Path(args.analysis_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_input(transitions_path, manifest)
    transitions = pd.read_csv(transitions_path)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.15), constrained_layout=True)
    image = None
    for axis, system in zip(axes, SYSTEM_ORDER, strict=True):
        subset = transitions.loc[transitions["system"] == system]
        matrix = np.zeros((2, 2), dtype=float)
        counts = np.zeros((2, 2), dtype=int)
        for row in subset.itertuples(index=False):
            position = MATRIX_POSITIONS[row.transition]
            matrix[position] = 100 * row.percent_of_matches
            counts[position] = row.count
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=65, aspect="equal")
        for row_index in range(2):
            for column_index in range(2):
                value = matrix[row_index, column_index]
                color = "white" if value >= 37 else "#222222"
                axis.text(
                    column_index,
                    row_index,
                    f"{counts[row_index, column_index]}\n{value:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                    fontweight="bold" if value >= 20 else "normal",
                )
        axis.set_title(DISPLAY_NAMES[system])
        axis.set_xticks((0, 1), ("Correct", "Incorrect"))
        axis.set_yticks((0, 1), ("Correct", "Incorrect"))
        axis.set_xlabel("Predicted-mask morphology")
        if axis is axes[0]:
            axis.set_ylabel("Ground-truth-mask morphology")
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.tick_params(which="minor", bottom=False, left=False)

    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("Percent of matched instances")
    fig.suptitle("MI05 classification correctness transitions", y=1.05, fontsize=12)
    fig.text(
        0.5,
        -0.045,
        "Incorrect-to-correct cells are accidental classifier transitions from altered geometry, not beneficial biological effects.",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = save_figure(fig, output_dir / "mi05_classification_transition_matrices")
    write_figure_manifest(
        output_dir=output_dir,
        script_path=Path(__file__).resolve(),
        inputs=(transitions_path, manifest_path),
        outputs=outputs,
        notes="Matched MI05 instances only. Common percentage scale across all three matrices. Incorrect-to-correct is not interpreted as a beneficial biological effect.",
    )


if __name__ == "__main__":
    main()
