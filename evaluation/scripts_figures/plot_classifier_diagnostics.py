"""Render classifier confusion and permutation-importance figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts_figures.provenance import write_figure_manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--importance", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = json.loads(Path(args.diagnostics).read_text(encoding="utf-8"))
    importance = pd.read_csv(args.importance).sort_values("importance_mean", ascending=True)

    raw = np.asarray(diagnostics["confusion_matrix"])
    normalized = np.asarray(diagnostics["normalized_confusion_matrix"])
    labels = ["Interphase", "Mitotic"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for axis, matrix, title, fmt in (
        (axes[0], raw, "Raw counts", "d"),
        (axes[1], normalized, "Row-normalized", ".2f"),
    ):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() if matrix.max() else 1)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, format(matrix[row, column], fmt), ha="center", va="center")
        axis.set_xticks([0, 1], labels=labels)
        axis.set_yticks([0, 1], labels=labels)
        axis.set_xlabel("Predicted annotation class")
        axis.set_ylabel("Ground-truth annotation class")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle("Random Forest validation on MI04")
    fig.tight_layout()
    fig.savefig(output / "classifier_confusion_matrices.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "classifier_confusion_matrices.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.barh(importance["feature"], importance["importance_mean"], xerr=importance["importance_std"], color="#0072B2")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Decrease in balanced accuracy after permutation")
    axis.set_title("MI04 permutation importance")
    fig.tight_layout()
    fig.savefig(output / "classifier_permutation_importance.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "classifier_permutation_importance.pdf", bbox_inches="tight")
    plt.close(fig)
    write_figure_manifest(
        output,
        Path(__file__),
        (Path(args.diagnostics), Path(args.importance)),
        (
            output / "classifier_confusion_matrices.png",
            output / "classifier_confusion_matrices.pdf",
            output / "classifier_permutation_importance.png",
            output / "classifier_permutation_importance.pdf",
        ),
        "Classifier figures generated from locked MI04 diagnostic outputs.",
    )


if __name__ == "__main__":
    main()
