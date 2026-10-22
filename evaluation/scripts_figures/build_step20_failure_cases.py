"""Build one deterministic cross-method MI05 failure-case comparison."""

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
from code.unified_evaluation import index_masks, sha256_file
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
GROUND_TRUTH = "#333333"
FALSE_NEGATIVE = "#CC79A7"


def normalize_image(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    return np.clip((image.astype(float) - low) / max(high - low, 1.0), 0, 1)


def draw_objects(
    axis: mpl.axes.Axes,
    labels: np.ndarray,
    object_ids: set[int],
    color: str,
    linewidth: float,
    linestyle: str,
    alpha: float = 1.0,
) -> None:
    for object_id in sorted(object_ids):
        mask = labels == object_id
        if mask.any():
            axis.contour(
                mask.astype(float),
                levels=[0.5],
                colors=[color],
                linewidths=linewidth,
                linestyles=linestyle,
                alpha=alpha,
            )


def verified_frame_paths(manifest_path: Path, frame_id: str) -> tuple[Path, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(item for item in manifest["inputs"] if item["frame_id"] == frame_id)
    ground_truth = Path(row["ground_truth_instance"]["path"])
    prediction = Path(row["prediction"]["path"])
    for path, expected in (
        (ground_truth, row["ground_truth_instance"]["sha256"]),
        (prediction, row["prediction"]["sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Evaluation-manifest hash mismatch: {path}")
    return ground_truth, prediction


def verify_prior_error_inputs(paths: list[Path], manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = {Path(item["path"]).resolve(): item["sha256"] for item in manifest["inputs"]}
    for path in paths:
        expected = recorded.get(path.resolve())
        if expected is None or sha256_file(path) != expected:
            raise ValueError(f"Prior error-analysis input is missing or changed: {path}")


def select_frames(errors_by_system: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for system, errors in errors_by_system.items():
        for error_type in ("false_negative", "false_positive"):
            counts = errors.loc[errors["error_type"].eq(error_type)].groupby("frame_id").size()
            rows.extend(
                {
                    "system": system,
                    "frame_id": frame_id,
                    "error_type": error_type,
                    "count": int(count),
                }
                for frame_id, count in counts.items()
            )
    frame_counts = (
        pd.DataFrame(rows)
        .groupby(["frame_id", "error_type"], as_index=False)["count"]
        .sum()
        .pivot(index="frame_id", columns="error_type", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    selected: list[dict[str, object]] = []
    used: set[str] = set()
    rules = [
        ("false_negative", "highest summed false-negative count across the three systems"),
        ("false_positive", "highest summed false-positive count across the three systems, excluding the first frame"),
    ]
    for rank, (column, description) in enumerate(rules, start=1):
        candidate = (
            frame_counts.loc[~frame_counts["frame_id"].isin(used)]
            .sort_values([column, "frame_id"], ascending=[False, True], kind="stable")
            .iloc[0]
        )
        frame_id = str(candidate["frame_id"])
        used.add(frame_id)
        selected.append(
            {
                "selection_rank": rank,
                "frame_id": frame_id,
                "selection_rule": description,
                "summed_false_negatives": int(candidate["false_negative"]),
                "summed_false_positives": int(candidate["false_positive"]),
                "tie_break": "lowest frame identifier",
            }
        )
    return pd.DataFrame(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    error_paths = {system: results_root / system / "error_cases.csv" for system in SYSTEMS}
    match_paths = {system: results_root / system / "matched_instances.csv" for system in SYSTEMS}
    verify_prior_error_inputs(
        [*error_paths.values(), *match_paths.values()],
        results_root / "error_analysis" / "figure_manifest.json",
    )
    errors_by_system = {system: pd.read_csv(path) for system, path in error_paths.items()}
    matches_by_system = {system: pd.read_csv(path) for system, path in match_paths.items()}

    selection = select_frames(errors_by_system)
    selection_path = output_dir / "failure_case_selection.csv"
    selection.to_csv(selection_path, index=False, lineterminator="\n")

    image_index = index_masks(dataset_root / "MI05" / "Images")
    manifest_inputs: set[Path] = set(error_paths.values()) | set(match_paths.values())
    panel_rows: list[dict[str, object]] = []

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.9), constrained_layout=False)
    for row_index, selected in enumerate(selection.itertuples(index=False)):
        frame_id = selected.frame_id
        image_path = image_index[frame_id]
        manifest_inputs.add(image_path)
        image = np.asarray(Image.open(image_path))
        scaled = normalize_image(image)

        ground_truth_path, _ = verified_frame_paths(
            results_root / SYSTEMS[0] / "evaluation_manifest.json", frame_id
        )
        manifest_inputs.add(ground_truth_path)
        ground_truth = np.asarray(Image.open(ground_truth_path))
        gt_ids = set(int(value) for value in np.unique(ground_truth) if value > 0)

        reference = axes[row_index, 0]
        reference.imshow(scaled, cmap="gray", vmin=0, vmax=1)
        draw_objects(reference, ground_truth, gt_ids, "white", 2.2, "-")
        draw_objects(reference, ground_truth, gt_ids, GROUND_TRUTH, 1.1, "-")
        reference.set_title("DIC + all GT boundaries")
        reference.set_ylabel(
            f"{frame_id}\n"
            f"sum FN={selected.summed_false_negatives}; sum FP={selected.summed_false_positives}",
            fontsize=8.2,
        )

        for column_index, system in enumerate(SYSTEMS, start=1):
            axis = axes[row_index, column_index]
            verified_gt, prediction_path = verified_frame_paths(
                results_root / system / "evaluation_manifest.json", frame_id
            )
            if verified_gt.resolve() != ground_truth_path.resolve():
                raise ValueError(f"Ground-truth path mismatch for {system}, {frame_id}")
            manifest_inputs.add(prediction_path)
            prediction = np.asarray(Image.open(prediction_path))
            frame_errors = errors_by_system[system].loc[
                errors_by_system[system]["frame_id"].eq(frame_id)
            ]
            frame_matches = matches_by_system[system].loc[
                matches_by_system[system]["frame_id"].eq(frame_id)
            ]
            matched_prediction_ids = set(frame_matches["pred_id"].dropna().astype(int))
            false_negative_ids = set(
                frame_errors.loc[frame_errors["error_type"].eq("false_negative"), "gt_id"]
                .dropna()
                .astype(int)
            )
            false_positive_ids = set(
                frame_errors.loc[frame_errors["error_type"].eq("false_positive"), "pred_id"]
                .dropna()
                .astype(int)
            )
            axis.imshow(scaled, cmap="gray", vmin=0, vmax=1)
            draw_objects(axis, ground_truth, gt_ids, "#B8B8B8", 0.65, "--", alpha=0.8)
            draw_objects(axis, prediction, matched_prediction_ids, COLORS[system], 1.7, "-")
            draw_objects(axis, ground_truth, false_negative_ids, FALSE_NEGATIVE, 2.0, "--")
            draw_objects(axis, prediction, false_positive_ids, "black", 3.0, ":")
            draw_objects(axis, prediction, false_positive_ids, "white", 1.8, ":")
            tp = len(frame_matches)
            fn = len(false_negative_ids)
            fp = len(false_positive_ids)
            mean_iou = float(frame_matches["iou"].mean()) if tp else float("nan")
            axis.set_title(
                f"{NAMES[system]}\nTP {tp} | FN {fn} | FP {fp} | mean IoU {mean_iou:.2f}",
                color=COLORS[system],
                fontweight="bold",
            )
            panel_rows.append(
                {
                    "frame_id": frame_id,
                    "system": system,
                    "true_positives": tp,
                    "false_negatives": fn,
                    "false_positives": fp,
                    "mean_matched_iou": mean_iou,
                }
            )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    handles = [
        Line2D([0], [0], color=GROUND_TRUTH, lw=1.2, linestyle="--", label="All GT (thin dashed)"),
        Line2D([0], [0], color="#666666", lw=2.0, linestyle="-", label="Matched prediction (method color)"),
        Line2D([0], [0], color=FALSE_NEGATIVE, lw=2.0, linestyle="--", label="FN: unmatched GT"),
        Line2D([0], [0], color="black", lw=2.0, linestyle=":", label="FP: unmatched prediction (white on image)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=4, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "Deterministically selected high-error MI05 frames at IoU 0.5",
        fontsize=12,
        y=0.985,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=0.12, wspace=0.05, hspace=0.12)

    panel_path = output_dir / "failure_case_panel_metrics.csv"
    pd.DataFrame(panel_rows).to_csv(panel_path, index=False, lineterminator="\n")
    pdf_path = output_dir / "mi05_failure_case_comparison.pdf"
    png_path = output_dir / "mi05_failure_case_comparison.png"
    metadata = {
        "Creator": "scripts_figures/build_step20_failure_cases.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf_path, bbox_inches="tight", metadata=metadata)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)

    outputs = [selection_path, panel_path, pdf_path, png_path]
    write_figure_manifest(
        output_dir,
        Path(__file__),
        manifest_inputs,
        outputs,
        (
            "Frames are selected before visualization by summed false-negative and false-positive counts "
            "across all three systems; ties use the lowest frame identifier. FN and FP are defined by "
            "the shared one-to-one IoU 0.5 evaluator."
        ),
    )
    print(json.dumps({"selected_frames": selection["frame_id"].tolist(), "outputs": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
