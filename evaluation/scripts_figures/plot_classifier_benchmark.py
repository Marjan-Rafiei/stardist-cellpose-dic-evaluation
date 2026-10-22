"""Plot MI04 confusion matrices for the RF and linear benchmark."""

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


MODEL_ORDER = ("random_forest", "logistic_regression")
DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
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
            "Creator": "scripts_figures/plot_classifier_benchmark.py",
            "CreationDate": None,
            "ModDate": None,
        },
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_input(metrics_path, manifest)
    metrics = pd.read_csv(metrics_path).set_index("model")

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

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), constrained_layout=True)
    image = None
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        row = metrics.loc[model]
        counts = np.asarray(
            [
                [row.cm_interphase_as_interphase, row.cm_interphase_as_mitotic],
                [row.cm_mitotic_as_interphase, row.cm_mitotic_as_mitotic],
            ],
            dtype=int,
        )
        percentages = 100 * counts / counts.sum(axis=1, keepdims=True)
        image = axis.imshow(percentages, cmap="Blues", vmin=0, vmax=100, aspect="equal")
        for row_index in range(2):
            for column_index in range(2):
                value = percentages[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{counts[row_index, column_index]}\n{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value >= 55 else "#222222",
                    fontsize=9,
                    fontweight="bold" if value >= 55 else "normal",
                )
        axis.set_title(f"{DISPLAY_NAMES[model]}\nmacro F1 = {row.macro_f1:.3f}")
        axis.set_xticks((0, 1), ("Interphase", "Mitotic"))
        axis.set_yticks((0, 1), ("Interphase", "Mitotic"))
        axis.set_xlabel("Predicted annotation class")
        if axis is axes[0]:
            axis.set_ylabel("Ground-truth annotation class")
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 2, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.tick_params(which="minor", bottom=False, left=False)

    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.84, pad=0.03)
    colorbar.set_label("Row percentage (%)")
    fig.suptitle("MI04 classifier benchmark on ground-truth morphology", y=1.07, fontsize=12)
    fig.text(
        0.5,
        -0.045,
        "Same MI01 to MI03 training data, MI04 validation, raw features, median imputation, and balanced class weights.",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = save_figure(fig, output_dir / "mi04_classifier_benchmark_confusion_matrices")
    write_figure_manifest(
        output_dir=output_dir,
        script_path=Path(__file__).resolve(),
        inputs=(metrics_path, manifest_path),
        outputs=outputs,
        notes="MI04 validation only. Logistic Regression is one simple untuned benchmark, not part of a model leaderboard. No MI05 data were used.",
    )


if __name__ == "__main__":
    main()
