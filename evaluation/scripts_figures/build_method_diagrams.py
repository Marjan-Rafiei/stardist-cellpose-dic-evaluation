"""Render source-backed method and split diagrams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scripts_figures.provenance import write_figure_manifest


GROUND_TRUTH = "#333333"
STARDIST = "#0072B2"
CELLPOSE_AUTO = "#009E73"
CELLPOSE_FIXED = "#E69F00"
CLASSIFIER = "#A64D79"
INK = "#25313C"
MUTED = "#5F6B76"
PROCESS_EDGE = "#607282"
PROCESS_FILL = "#F4F7F9"


def box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str = PROCESS_FILL,
    edgecolor: str = PROCESS_EDGE,
    fontsize: float = 8.6,
    linewidth: float = 1.25,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    axis.add_patch(patch)
    label = axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        weight=weight,
        linespacing=1.12,
    )
    if not hasattr(axis, "_qc_box_labels"):
        axis._qc_box_labels = []
    axis._qc_box_labels.append((patch, label))


def arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = MUTED,
    linewidth: float = 1.2,
    linestyle: str = "-",
    connectionstyle: str = "arc3",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def path_arrow(
    axis,
    points: list[tuple[float, float]],
    color: str = MUTED,
    linewidth: float = 1.1,
    linestyle: str = "-",
) -> None:
    xs, ys = zip(*points)
    axis.plot(xs[:-1], ys[:-1], color=color, linewidth=linewidth, linestyle=linestyle)
    arrow(axis, points[-2], points[-1], color=color, linewidth=linewidth, linestyle=linestyle)


def panel(axis, x: float, y: float, width: float, height: float, title: str) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05,rounding_size=0.10",
            facecolor="#FBFCFD",
            edgecolor="#C7D0D9",
            linewidth=0.9,
            zorder=-10,
        )
    )
    axis.text(
        x + 0.25,
        y + height - 0.32,
        title,
        ha="left",
        va="center",
        fontsize=10.3,
        weight="bold",
        color=INK,
    )


def save_figure(fig, output: Path, name: str) -> tuple[Path, Path]:
    png = output / f"{name}.png"
    pdf = output / f"{name}.pdf"
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "scripts_figures/build_method_diagrams.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "Matplotlib"},
    )
    plt.close(fig)
    return png, pdf


def build_evaluation_pipeline(output: Path) -> tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(7.2, 7.6))
    axis.set_xlim(0, 16.6)
    axis.set_ylim(0, 15.35)
    axis.axis("off")

    panel(axis, 0.15, 11.35, 16.25, 3.45, "A. Ground-truth classifier development and lock")
    panel(axis, 0.15, 0.20, 16.25, 10.75, "B. Frozen MI05 evaluation")

    # Development path: MI01-MI03 fit, MI04 validation, then lock.
    box(axis, 0.55, 13.05, 2.75, 0.78, "MI01-MI03 masks\n+ ALFI classes", "#F1F2F3", GROUND_TRUTH)
    box(axis, 3.85, 13.05, 2.55, 0.78, "Eight morphology\nfeatures")
    box(axis, 6.95, 13.05, 2.35, 0.78, "Train\nRandom Forest", "#F7EAF3", CLASSIFIER)
    box(axis, 9.85, 13.05, 2.75, 0.78, "Fitted feature and\nclassifier pipeline", "#F7EAF3", CLASSIFIER)

    box(axis, 0.55, 11.80, 2.75, 0.78, "MI04 GT masks\n+ ALFI classes", "#F1F2F3", GROUND_TRUTH)
    box(axis, 3.85, 11.80, 2.55, 0.78, "Eight morphology\nfeatures")
    box(axis, 6.95, 11.80, 2.35, 0.78, "Validate on MI04", "#F7EAF3", CLASSIFIER)
    box(axis, 9.85, 11.80, 2.75, 0.78, "Locked RF\npipeline", "#F7EAF3", CLASSIFIER, fontsize=8.0, weight="bold")
    box(axis, 13.15, 11.80, 2.75, 0.92, "No MI05 in\nmodel development", "#FFF8E8", "#A66B00", fontsize=8.1)

    arrow(axis, (3.30, 13.44), (3.85, 13.44))
    arrow(axis, (6.40, 13.44), (6.95, 13.44))
    arrow(axis, (9.30, 13.44), (9.85, 13.44), color=CLASSIFIER)
    arrow(axis, (3.30, 12.19), (3.85, 12.19))
    arrow(axis, (6.40, 12.19), (6.95, 12.19))
    arrow(axis, (11.22, 13.05), (8.12, 12.58), color=CLASSIFIER, connectionstyle="arc3,rad=0.10")
    arrow(axis, (9.30, 12.19), (9.85, 12.19), color=CLASSIFIER)
    arrow(axis, (12.60, 12.19), (13.15, 12.19), color="#A66B00", linestyle="--")

    # Frozen MI05 segmentation sources and reference annotations.
    axis.text(0.55, 9.98, "Segmentation stage", fontsize=8.8, weight="bold", color=INK, ha="left")
    box(axis, 0.55, 8.15, 2.25, 0.98, "MI05 DIC", "#F1F2F3", GROUND_TRUTH, weight="bold")
    box(axis, 3.40, 9.05, 2.85, 0.72, "Saved StarDist", "#E5F2FA", STARDIST, linewidth=1.8, weight="bold")
    box(axis, 3.40, 8.15, 2.85, 0.72, "Cellpose\nautomatic", "#E6F5F0", CELLPOSE_AUTO, linewidth=1.8, weight="bold", fontsize=8.0)
    box(axis, 3.40, 7.25, 2.85, 0.72, "Cellpose fixed", "#FDF2D7", CELLPOSE_FIXED, linewidth=1.8, weight="bold")
    box(axis, 7.00, 8.15, 2.55, 0.98, "Predicted\ninstance masks", "#EEF3F6", PROCESS_EDGE, weight="bold")
    box(axis, 3.40, 6.00, 2.85, 0.98, "MI05 GT\n+ class labels", "#F1F2F3", GROUND_TRUTH)

    arrow(axis, (2.80, 8.64), (3.40, 9.41), color=STARDIST)
    arrow(axis, (2.80, 8.64), (3.40, 8.51), color=CELLPOSE_AUTO)
    arrow(axis, (2.80, 8.64), (3.40, 7.61), color=CELLPOSE_FIXED)
    arrow(axis, (6.25, 9.41), (7.00, 8.88), color=STARDIST)
    arrow(axis, (6.25, 8.51), (7.00, 8.64), color=CELLPOSE_AUTO)
    arrow(axis, (6.25, 7.61), (7.00, 8.35), color=CELLPOSE_FIXED)

    # Branch A: matching and segmentation quality.
    axis.text(10.20, 9.98, "Branch A: segmentation quality", fontsize=8.8, weight="bold", color=INK, ha="left")
    box(axis, 10.20, 8.15, 2.55, 0.98, "One-to-one\nIoU matching\n(threshold 0.5)", "#EEF3F6", PROCESS_EDGE, fontsize=8.0, weight="bold")
    box(
        axis,
        13.35,
        7.95,
        2.55,
        1.38,
        "TP / FP / FN\nP / R / F1\nPQ / matched IoU",
        "#EDF6FC",
        STARDIST,
        fontsize=8.1,
    )
    arrow(axis, (9.55, 8.64), (10.20, 8.64))
    arrow(axis, (6.25, 6.49), (10.45, 8.15), color=GROUND_TRUTH, linestyle="--")
    arrow(axis, (12.75, 8.64), (13.35, 8.64))

    # Branch B: morphology, shared classifier, and downstream outputs.
    axis.text(10.20, 5.98, "Branch B: class recovery and cell analytics", fontsize=8.8, weight="bold", color=INK, ha="left")
    box(axis, 7.00, 4.72, 2.55, 0.98, "Predicted-mask\nshape features")
    box(axis, 10.20, 4.72, 2.55, 0.98, "Apply locked RF\nfrom panel A", "#F7EAF3", CLASSIFIER, weight="bold")
    box(axis, 13.35, 4.72, 2.55, 0.98, "Predicted\nALFI class", "#F7EAF3", CLASSIFIER, weight="bold")
    arrow(axis, (8.27, 8.15), (8.27, 5.70))
    arrow(axis, (9.55, 5.21), (10.20, 5.21))
    arrow(axis, (12.75, 5.21), (13.35, 5.21), color=CLASSIFIER)

    box(axis, 7.00, 2.05, 2.55, 1.22, "Matched-only\nconfusion matrix\n+ macro F1", "#F5F7F9", PROCESS_EDGE, fontsize=7.9)
    box(axis, 10.20, 2.05, 2.55, 1.22, "End-to-end class\nprecision, recall,\nand F1", "#F5F7F9", PROCESS_EDGE, fontsize=7.9)
    box(axis, 13.35, 1.97, 2.55, 1.30, "All classified\npredictions:\ncell count + mitotic\nratio vs GT", "#F5F7F9", PROCESS_EDGE, fontsize=7.3)

    # Matching status feeds the conditional and end-to-end class endpoints.
    path_arrow(axis, [(10.72, 8.15), (9.92, 7.48), (9.92, 3.60), (8.27, 3.60), (8.27, 3.27)])
    path_arrow(axis, [(9.92, 3.60), (11.47, 3.60), (11.47, 3.27)])

    # Predicted class labels feed both class endpoints and all-prediction analytics.
    axis.plot([14.62, 14.62, 8.80], [4.72, 3.86, 3.86], color=CLASSIFIER, linewidth=1.1)
    arrow(axis, (8.80, 3.86), (8.80, 3.27), color=CLASSIFIER)
    arrow(axis, (11.47, 3.86), (11.47, 3.27), color=CLASSIFIER)
    arrow(axis, (14.62, 3.86), (14.62, 3.27), color=CLASSIFIER)

    axis.text(
        0.58,
        5.10,
        "Segmentation sources output\ninstance masks only.",
        ha="left",
        va="center",
        fontsize=8.8,
        weight="bold",
        color=INK,
        linespacing=1.3,
    )
    axis.text(
        0.58,
        3.82,
        "The Random Forest produces\nALFI annotation-class labels.",
        ha="left",
        va="center",
        fontsize=8.8,
        weight="bold",
        color=CLASSIFIER,
        linespacing=1.3,
    )
    axis.text(
        0.58,
        1.50,
        "Matched-only: accepted pairs.\n"
        "End-to-end: misses, extra predictions,\n"
        "and class swaps. Analytics: all predictions.",
        ha="left",
        va="center",
        fontsize=8.0,
        color=MUTED,
        linespacing=1.28,
    )

    fig.suptitle("From DIC images to verified cell analytics", fontsize=12.2, weight="bold", color=INK, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.01)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflowing_labels = []
    for patch, label in axis._qc_box_labels:
        boundary = patch.get_window_extent(renderer)
        text_bounds = label.get_window_extent(renderer)
        if (
            text_bounds.x0 < boundary.x0 + 1
            or text_bounds.x1 > boundary.x1 - 1
            or text_bounds.y0 < boundary.y0 + 1
            or text_bounds.y1 > boundary.y1 - 1
        ):
            overflowing_labels.append(label.get_text())
    if overflowing_labels:
        raise ValueError(f"Diagram labels exceed their boxes: {overflowing_labels!r}")
    return save_figure(fig, output, "evaluation_pipeline")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Regenerate only the evaluation pipeline while retaining existing split assets.",
    )
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    pipeline_outputs = build_evaluation_pipeline(output)

    split_outputs = (output / "dataset_split.png", output / "dataset_split.pdf")
    if not args.pipeline_only:
        counts = manifest["counts"]
        labels = ["MI01", "MI02", "MI03", "MI04", "MI05"]
        values = [counts[label]["aligned_core"] for label in labels]
        roles = ["Classifier training", "Classifier training", "Classifier training", "Validation and lock", "Retrospective final report"]
        colors = ["#56B4E9", "#56B4E9", "#56B4E9", "#E69F00", "#009E73"]
        fig, axis = plt.subplots(figsize=(9, 4.3))
        bars = axis.bar(labels, values, color=colors)
        axis.set_ylabel("Aligned frames")
        axis.set_ylim(0, max(values) + 18)
        axis.set_title("Sequence-level split with no cross-sequence frame mixing")
        for bar, value, role in zip(bars, values, roles):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value}\n{role}", ha="center", va="bottom", fontsize=8)
        axis.text(0.02, 0.02, "MI01 also has two annotation-only masks excluded from aligned analysis.", transform=axis.transAxes, fontsize=8)
        fig.tight_layout()
        split_outputs = save_figure(fig, output, "dataset_split")
    elif not all(path.exists() for path in split_outputs):
        raise FileNotFoundError("--pipeline-only requires existing dataset_split PNG and PDF assets")
    write_figure_manifest(
        output,
        Path(__file__),
        (Path(args.dataset_manifest),),
        (
            *pipeline_outputs,
            *split_outputs,
        ),
        "Method diagrams generated from the verified dataset manifest and the approved workflow specification encoded in this script.",
    )


if __name__ == "__main__":
    main()
