"""Illustrate the deterministically selected median-error MI05 frame.

Only preserved images, masks, and verified evaluator outputs are read. No model
inference or metric recomputation is performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from code.unified_evaluation import index_masks, sha256_file, write_json
from scripts_figures.build_step20_failure_cases import (
    COLORS, FALSE_NEGATIVE, GROUND_TRUTH, NAMES, SYSTEMS,
    draw_objects, normalize_image, verified_frame_paths,
)


def select_frame(frames: dict[str, pd.DataFrame]) -> tuple[str, float, pd.DataFrame]:
    ids = sorted(set(frames[SYSTEMS[0]]["frame_id"]))
    if any(sorted(set(frames[system]["frame_id"])) != ids for system in SYSTEMS):
        raise ValueError("The three verified files do not cover identical frames")
    rows = []
    for frame_id in ids:
        row = {"frame_id": frame_id}
        for system in SYSTEMS:
            record = frames[system].set_index("frame_id").loc[frame_id]
            row[f"{system}_fp"] = int(record["fp"])
            row[f"{system}_fn"] = int(record["fn"])
        row["aggregate_fp_fn"] = sum(
            row[f"{system}_{kind}"] for system in SYSTEMS for kind in ("fp", "fn")
        )
        rows.append(row)
    selection = pd.DataFrame(rows)
    median = float(selection["aggregate_fp_fn"].median())
    selection["distance_from_median"] = (selection["aggregate_fp_fn"] - median).abs()
    selection = selection.sort_values(["distance_from_median", "frame_id"], kind="stable")
    return str(selection.iloc[0]["frame_id"]), median, selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    args = parser.parse_args()
    root = Path(args.results_root).resolve()
    dataset = Path(args.dataset_root).resolve()
    figure_dir = Path(args.figure_dir).resolve()
    manifest_dir = Path(args.manifest_dir).resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    metrics_paths = {system: root / system / "metrics_by_frame.csv" for system in SYSTEMS}
    match_paths = {system: root / system / "matched_instances.csv" for system in SYSTEMS}
    error_paths = {system: root / system / "error_cases.csv" for system in SYSTEMS}
    eval_paths = {system: root / system / "evaluation_manifest.json" for system in SYSTEMS}
    metrics = {system: pd.read_csv(path) for system, path in metrics_paths.items()}
    frame_id, median, selection = select_frame(metrics)
    chosen = selection.iloc[0]
    image_path = index_masks(dataset / "MI05" / "Images")[frame_id]
    image = normalize_image(np.asarray(Image.open(image_path)))
    gt_path, _ = verified_frame_paths(eval_paths[SYSTEMS[0]], frame_id)
    gt = np.asarray(Image.open(gt_path))
    gt_ids = set(np.unique(gt).astype(int)) - {0}

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.2))
    axes = axes.ravel()
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    draw_objects(axes[0], gt, gt_ids, "white", 2.1, "-")
    draw_objects(axes[0], gt, gt_ids, GROUND_TRUTH, 1.0, "-")
    axes[0].set_title("(a) DIC image + all annotated boundaries")
    inputs = set(metrics_paths.values()) | set(match_paths.values()) | set(error_paths.values()) | set(eval_paths.values()) | {image_path, gt_path}
    panel_counts = {}
    for column, system in enumerate(SYSTEMS, 1):
        verified_gt, pred_path = verified_frame_paths(eval_paths[system], frame_id)
        if verified_gt.resolve() != gt_path.resolve():
            raise ValueError(f"Ground-truth path mismatch: {system}")
        inputs.add(pred_path)
        pred = np.asarray(Image.open(pred_path))
        matches = pd.read_csv(match_paths[system])
        matches = matches.loc[matches["frame_id"].eq(frame_id)]
        errors = pd.read_csv(error_paths[system])
        errors = errors.loc[errors["frame_id"].eq(frame_id)]
        matched_pred = set(matches["pred_id"].dropna().astype(int))
        fn = set(errors.loc[errors["error_type"].eq("false_negative"), "gt_id"].dropna().astype(int))
        fp = set(errors.loc[errors["error_type"].eq("false_positive"), "pred_id"].dropna().astype(int))
        if len(fp) != int(chosen[f"{system}_fp"]) or len(fn) != int(chosen[f"{system}_fn"]):
            raise ValueError(f"Evaluator error counts do not reconcile: {system}")
        panel_counts[system] = {"tp": len(matches), "fp": len(fp), "fn": len(fn)}
        axis = axes[column]
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        draw_objects(axis, gt, gt_ids, "#B8B8B8", 0.65, "--", alpha=0.8)
        draw_objects(axis, pred, matched_pred, COLORS[system], 1.7, "-")
        draw_objects(axis, gt, fn, FALSE_NEGATIVE, 2.0, "--")
        draw_objects(axis, pred, fp, "black", 3.0, ":")
        draw_objects(axis, pred, fp, "white", 1.8, ":")
        axis.set_title(f"({chr(97 + column)}) {NAMES[system]}: TP {len(matches)}, FP {len(fp)}, FN {len(fn)}",
                       color=COLORS[system], fontweight="bold")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    handles = [
        Line2D([0], [0], color=GROUND_TRUTH, lw=1.2, linestyle="--", label="All annotation outlines"),
        Line2D([0], [0], color="#666666", lw=2, label="Matched prediction (method colour)"),
        Line2D([0], [0], color=FALSE_NEGATIVE, lw=2, linestyle="--", label="Unmatched annotation (FN)"),
        Line2D([0], [0], color="black", lw=2, linestyle=":", label="Unmatched prediction (FP)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=2, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"MI05 median-error illustration: {frame_id} (aggregate FP + FN = {int(chosen['aggregate_fp_fn'])})",
                 fontsize=12, y=0.985)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.92, bottom=0.14, wspace=0.045, hspace=0.17)
    pdf_path = figure_dir / "mi05_median_error_comparison.pdf"
    png_path = figure_dir / "mi05_median_error_comparison.png"
    fig.savefig(pdf_path, bbox_inches="tight", metadata={"Creator": str(Path(__file__).name),
                "CreationDate": None, "ModDate": None})
    fig.savefig(png_path, dpi=250, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    write_json(manifest_dir / "median_error_figure_manifest.json", {
        "schema_version": "1.0",
        "selection_rule": "For each common MI05 frame, sum FP+FN across all three configurations at IoU 0.5; choose the frame closest to the median of these sums, breaking ties by lowest frame identifier.",
        "selected_frame": frame_id, "frame_count": len(selection),
        "median_aggregate_fp_fn": median,
        "selected_aggregate_fp_fn": int(chosen["aggregate_fp_fn"]),
        "selected_panel_counts": panel_counts,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in sorted(inputs)],
        "outputs": [{"path": str(path), "sha256": sha256_file(path)} for path in (pdf_path, png_path)],
        "interpretation": "Median-error illustration, not a representative success or a random sample.",
    })
    print(json.dumps({"selected_frame": frame_id, "median": median,
                      "aggregate_fp_fn": int(chosen["aggregate_fp_fn"]), "panels": panel_counts}, indent=2))


if __name__ == "__main__":
    main()
