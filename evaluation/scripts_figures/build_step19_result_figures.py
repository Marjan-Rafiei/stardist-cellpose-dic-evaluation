"""Build the Step 19 quantitative result figures from verified machine-readable results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_figures.provenance import write_figure_manifest


SYSTEMS = ["stardist_saved", "cellpose_auto", "cellpose_fixed"]
NAMES = {
    "stardist_saved": "StarDist",
    "cellpose_auto": "Cellpose automatic",
    "cellpose_fixed": "Cellpose fixed",
}
COLORS = {
    "stardist_saved": "#0072B2",
    "cellpose_auto": "#009E73",
    "cellpose_fixed": "#E69F00",
}
MARKERS = {"stardist_saved": "o", "cellpose_auto": "s", "cellpose_fixed": "^"}
LINES = {"stardist_saved": "-", "cellpose_auto": "--", "cellpose_fixed": ":"}


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_output(path: Path, manifest_path: Path) -> None:
    manifest = _read_manifest(manifest_path)
    recorded = {Path(item["path"]).resolve(): item["sha256"] for item in manifest["outputs"]}
    expected = recorded.get(path.resolve())
    if expected is None or _sha256(path) != expected:
        raise ValueError(f"Unverified input: {path}")


def _verify_classifier_predictions(path: Path, manifest_path: Path) -> None:
    expected = _read_manifest(manifest_path)["validation_predictions_sha256"]
    if _sha256(path) != expected:
        raise ValueError(f"Changed classifier validation predictions: {path}")


def _verify_indexed(path: Path, results_root: Path) -> None:
    index = _read_manifest(results_root / "results_index.json")
    recorded = {
        (results_root / item["relative_path"]).resolve(): item["sha256"]
        for item in index["sources"]
    }
    expected = recorded.get(path.resolve())
    if expected is None or _sha256(path) != expected:
        raise ValueError(f"Unindexed or changed input: {path}")


def _save(fig: mpl.figure.Figure, base: Path) -> list[Path]:
    pdf, png = base.with_suffix(".pdf"), base.with_suffix(".png")
    metadata = {
        "Creator": "scripts_figures/build_step19_result_figures.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, bbox_inches="tight", metadata=metadata)
    fig.savefig(png, dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return [pdf, png]


def _confusion(predictions: pd.DataFrame, out: Path) -> list[Path]:
    matrix = pd.crosstab(predictions["target"], predictions["predicted"]).reindex(
        index=[1, 2], columns=[1, 2], fill_value=0
    ).to_numpy()
    proportions = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.7), constrained_layout=True)
    image = ax.imshow(proportions, vmin=0, vmax=1, cmap="Blues")
    for row in range(2):
        for column in range(2):
            color = "white" if proportions[row, column] > 0.52 else "#222222"
            ax.text(column, row, f"{matrix[row, column]}\n({proportions[row, column]:.0%})",
                    ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks([0, 1], ["Interphase", "Mitotic"])
    ax.set_yticks([0, 1], [f"Interphase (n={matrix[0].sum()})", f"Mitotic (n={matrix[1].sum()})"])
    ax.set_xlabel("Predicted annotation class")
    ax.set_ylabel("Ground-truth annotation class")
    ax.set_title("MI04 locked-classifier validation")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row proportion")
    return _save(fig, out / "classifier_confusion_matrices")


def _importance(individual: pd.DataFrame, grouped: pd.DataFrame, out: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.3), constrained_layout=True)
    panels = [
        (individual.sort_values("importance_mean"), "Individual features", "feature", "#4C78A8"),
        (grouped.sort_values("importance_mean"), "Prespecified correlated groups", "group", "#7A5195"),
    ]
    xmax = 1.18 * max(frame["importance_mean"].max() + frame["importance_std"].max() for frame, *_ in panels)
    for ax, (frame, title, label_column, color) in zip(axes, panels):
        labels = frame[label_column].str.replace("_", " ").str.title()
        y = np.arange(len(frame))
        ax.barh(y, frame["importance_mean"], xerr=frame["importance_std"], color=color,
                alpha=0.9, ecolor="#333333", capsize=2)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, xmax)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Decrease in MI04 balanced accuracy after permutation")
    fig.suptitle("Permutation importance of the locked morphology classifier", fontsize=12)
    return _save(fig, out / "permutation_importance_stability_and_groups")


def _detection(comparison: pd.DataFrame, out: Path) -> list[Path]:
    data = comparison.set_index("system").loc[SYSTEMS]
    metrics = [
        ("pooled_precision", "Precision"), ("pooled_recall", "Recall"),
        ("pooled_f1", "F1"), ("pooled_pq", "Panoptic quality"),
        ("pooled_matched_iou", "Matched IoU")]
    y = np.arange(len(metrics))[::-1]
    offsets = [0.20, 0.0, -0.20]
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    for system, offset in zip(SYSTEMS, offsets):
        values = np.array([data.loc[system, column] for column, _ in metrics])
        ax.scatter(values, y + offset, s=48, color=COLORS[system], marker=MARKERS[system],
                   edgecolor="white", linewidth=0.6, label=NAMES[system], zorder=3)
        f1_index = 2
        lower = data.loc[system, "block_bootstrap_f1_lower"]
        upper = data.loc[system, "block_bootstrap_f1_upper"]
        ax.errorbar(values[f1_index], y[f1_index] + offset,
                    xerr=[[values[f1_index] - lower], [upper - values[f1_index]]],
                    fmt="none", ecolor=COLORS[system], capsize=2, linewidth=1.2, zorder=2)
    ax.axhline(0.5, color="#999999", linewidth=0.8)
    ax.text(0.995, 0.57, "Conditional on accepted matches", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="#555555")
    ax.set_yticks(y, [label for _, label in metrics])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    fig.suptitle("MI05 instance segmentation at IoU 0.5", y=0.98, fontsize=12)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 0.935))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, out / "mi05_detection_comparison")


def _classification(comparison: pd.DataFrame, out: Path) -> list[Path]:
    data = comparison.set_index("system").loc[SYSTEMS]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1), constrained_layout=True)
    y = np.arange(len(SYSTEMS))[::-1]
    for row, system in zip(y, SYSTEMS):
        matched = data.loc[system, "classification_only_macro_f1"]
        end = data.loc[system, "end_to_end_macro_f1"]
        axes[0].plot([end, matched], [row, row], color=COLORS[system], linewidth=2)
        axes[0].scatter(matched, row, color=COLORS[system], marker="o", s=55, zorder=3)
        axes[0].scatter(end, row, facecolor="white", edgecolor=COLORS[system], marker="o", s=55, zorder=3)
        inter = data.loc[system, "end_to_end_interphase_f1"]
        mit = data.loc[system, "end_to_end_mitotic_f1"]
        axes[1].plot([inter, mit], [row, row], color=COLORS[system], linewidth=2)
        axes[1].scatter(inter, row, color=COLORS[system], marker="o", s=55, zorder=3)
        axes[1].scatter(mit, row, color=COLORS[system], marker="D", s=48, zorder=3)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_yticks(y, [NAMES[s] for s in SYSTEMS])
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("F1 score")
    axes[0].set_title("Matched-only versus end-to-end macro F1")
    axes[1].set_title("End-to-end annotation-class F1")
    axes[0].scatter([], [], color="#555555", marker="o", label="Matched only")
    axes[0].scatter([], [], facecolor="white", edgecolor="#555555", marker="o", label="End-to-end")
    axes[1].scatter([], [], color="#555555", marker="o", label="Interphase")
    axes[1].scatter([], [], color="#555555", marker="D", label="Mitotic")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="lower right")
    return _save(fig, out / "mi05_classification_comparison")


def _temporal(per_frame: pd.DataFrame, out: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.8), sharex=True, constrained_layout=True)
    for ax, (column, label) in zip(axes, [("f1", "Instance F1"), ("pq", "Panoptic quality")]):
        for system in SYSTEMS:
            frame = per_frame[per_frame["system"] == system].sort_values("frame_number")
            ax.plot(frame["frame_number"], frame[column], color=COLORS[system], linestyle=LINES[system],
                    linewidth=1.5, marker=MARKERS[system], markersize=3, markevery=6,
                    label=NAMES[system])
        ax.set_ylim(0, 1)
        ax.set_ylabel(label)
        ax.grid(color="#E0E0E0", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    axes[-1].set_xlabel("MI05 frame")
    fig.suptitle("Frame-level segmentation performance across MI05", fontsize=12)
    return _save(fig, out / "mi05_temporal_segmentation_metrics")


def _bubble_panel(axis: mpl.axes.Axes, frame: pd.DataFrame, xcol: str, ycol: str,
                  system: str, max_value: float, band: float | None = None) -> None:
    counts = Counter(zip(frame[xcol].round(8), frame[ycol].round(8)))
    x = np.array([pair[0] for pair in counts])
    y = np.array([pair[1] for pair in counts])
    n = np.array(list(counts.values()))
    if band is not None:
        grid = np.linspace(0, max_value, 200)
        axis.fill_between(grid, np.maximum(0, grid - band), np.minimum(max_value, grid + band),
                          color="#BBBBBB", alpha=0.22, label="±5 pp band")
    axis.plot([0, max_value], [0, max_value], color="#333333", linewidth=1, linestyle="--")
    axis.scatter(x, y, s=22 + 18 * n, color=COLORS[system], alpha=0.75,
                 edgecolor="white", linewidth=0.5)
    axis.set_xlim(0, max_value)
    axis.set_ylim(0, max_value)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#E5E5E5", linewidth=0.5)
    axis.set_axisbelow(True)


def _agreement(by_frame: pd.DataFrame, summary: pd.DataFrame, out: Path, ratio: bool) -> list[Path]:
    summary = summary.set_index("system")
    if ratio:
        xcol, ycol = "gt_mitotic_ratio", "pred_mitotic_ratio"
        maximum = np.ceil(100 * max(by_frame[xcol].max(), by_frame[ycol].max()) / 10) * 10
        xscale, band = 100.0, 5.0
        filename = "mi05_mitotic_ratio_validation"
        xlabel, ylabel = "Ground-truth mitotic ratio (%)", "Predicted mitotic ratio (%)"
    else:
        xcol, ycol = "gt_total", "pred_total"
        maximum = np.ceil(max(by_frame[xcol].max(), by_frame[ycol].max()) / 5) * 5
        xscale, band = 1.0, None
        filename = "mi05_cell_count_validation"
        xlabel, ylabel = "Ground-truth count", "Predicted count"
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.6), constrained_layout=True)
    for axis, system in zip(axes, SYSTEMS):
        frame = by_frame[by_frame["system"] == system].copy()
        frame[xcol] *= xscale
        frame[ycol] *= xscale
        _bubble_panel(axis, frame, xcol, ycol, system, maximum, band)
        if ratio:
            bias, mae = 100 * summary.loc[system, "ratio_bias"], 100 * summary.loc[system, "ratio_mae"]
            unit = " pp"
        else:
            bias, mae = summary.loc[system, "count_bias"], summary.loc[system, "count_mae"]
            unit = ""
        axis.set_title(f"{NAMES[system]}\nBias {bias:+.2f}{unit}; MAE {mae:.2f}{unit}", fontsize=9)
        axis.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    handles = [
        plt.scatter([], [], s=22 + 18 * n, color="#777777", alpha=0.75,
                    edgecolor="white", label=f"{n} frame{'s' if n != 1 else ''}")
        for n in [1, 2, 4]
    ]
    fig.legend(handles=handles, frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    return _save(fig, out / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, out = Path(args.results_root).resolve(), Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "comparison": root / "comparison_summary.csv",
        "predictions": root / "classifier_v1" / "validation_predictions.csv",
        "individual": root / "feature_importance_audit" / "individual_importance_stability.csv",
        "grouped": root / "feature_importance_audit" / "grouped_importance_summary.csv",
        "per_frame": root / "figures" / "per_frame" / "per_frame_plot_data.csv",
        "agreement": root / "counts_ratio_validation" / "agreement_by_frame.csv",
        "summary": root / "counts_ratio_validation" / "agreement_summary.csv",
    }
    for key in ["comparison", "individual", "grouped", "agreement", "summary"]:
        _verify_indexed(paths[key], root)
    _verify_classifier_predictions(paths["predictions"], root / "classifier_v1" / "classifier_manifest.json")
    _verify_output(paths["per_frame"], root / "figures" / "per_frame" / "figure_manifest.json")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    comparison = pd.read_csv(paths["comparison"])
    predictions = pd.read_csv(paths["predictions"])
    individual = pd.read_csv(paths["individual"])
    grouped = pd.read_csv(paths["grouped"])
    per_frame = pd.read_csv(paths["per_frame"])
    agreement = pd.read_csv(paths["agreement"])
    summary = pd.read_csv(paths["summary"])
    outputs: list[Path] = []
    outputs += _confusion(predictions, out)
    outputs += _importance(individual, grouped, out)
    outputs += _detection(comparison, out)
    outputs += _classification(comparison, out)
    outputs += _temporal(per_frame, out)
    outputs += _agreement(agreement, summary, out, ratio=False)
    outputs += _agreement(agreement, summary, out, ratio=True)
    write_figure_manifest(out, Path(__file__), paths.values(), outputs,
                          "Step 19 result figures; ratios are percentages and method colors follow the approved palette.")
    print(json.dumps({"outputs": len(outputs), "manifest": str(out / 'figure_manifest.json')}, indent=2))


if __name__ == "__main__":
    main()
