"""Build ranked error tables and representative MI05 failure-case panels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.segmentation import find_boundaries

from code.unified_evaluation import index_masks
from scripts_figures.provenance import write_figure_manifest


def overlay_boundaries(image: np.ndarray, labels: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    scaled = np.clip((image.astype(float) - low) / max(high - low, 1.0), 0, 1)
    rgb = np.repeat(scaled[..., None], 3, axis=2)
    rgb[find_boundaries(labels, mode="outer")] = color
    return rgb


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sequence", default="MI05")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    results_root = Path(args.results_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence = args.sequence.upper()
    image_index = index_masks(dataset_root / sequence / "Images")
    gt_index = index_masks(dataset_root / sequence / "Masks_converted")
    prediction_roots = {
        "cellpose_auto": dataset_root / sequence / "Cellpose_masks",
        "cellpose_fixed": dataset_root / sequence / "Cellpose_masks_v2",
        "stardist_saved": results_root / "stardist_saved" / "predictions",
    }

    ranked_outputs = []
    selected_outputs = []
    type_outputs = []
    manifest_inputs: set[Path] = set()
    manifest_outputs: list[Path] = []
    for system, prediction_root in prediction_roots.items():
        error_path = results_root / system / "error_cases.csv"
        match_path = results_root / system / "matched_instances.csv"
        if not error_path.is_file() or not match_path.is_file() or not prediction_root.is_dir():
            continue
        errors = pd.read_csv(error_path)
        matches = pd.read_csv(match_path)
        manifest_inputs.update((error_path, match_path))
        errors = errors[errors["frame_id"].str.startswith(f"{sequence}_")]
        counts = errors.groupby("frame_id", as_index=False).size().rename(columns={"size": "error_count"})
        counts.insert(0, "system", system)
        counts = counts.sort_values(["error_count", "frame_id"], ascending=[False, True], kind="stable")
        ranked_outputs.append(counts)
        by_type = errors.groupby("error_type", as_index=False).size().rename(columns={"size": "count"})
        by_type.insert(0, "system", system)
        type_outputs.append(by_type)
        selected_records = []
        used_frames: set[str] = set()

        def select_candidate(candidates: pd.DataFrame, reason: str) -> None:
            for candidate in candidates.itertuples(index=False):
                if candidate.frame_id not in used_frames:
                    used_frames.add(candidate.frame_id)
                    selected_records.append(
                        {
                            "system": system,
                            "frame_id": candidate.frame_id,
                            "error_count": int(
                                counts.loc[counts["frame_id"] == candidate.frame_id, "error_count"].iloc[0]
                            ),
                            "selection_reason": reason,
                        }
                    )
                    return

        for error_type, reason in (
            ("false_negative", "highest false-negative count"),
            ("false_positive", "highest false-positive count"),
        ):
            candidates = (
                errors[errors["error_type"] == error_type]
                .groupby("frame_id", as_index=False)
                .size()
                .sort_values(["size", "frame_id"], ascending=[False, True], kind="stable")
            )
            select_candidate(candidates, reason)
        class_candidates = (
            errors[errors["error_type"].str.contains("class", na=False)]
            .groupby("frame_id", as_index=False)
            .size()
            .sort_values(["size", "frame_id"], ascending=[False, True], kind="stable")
        )
        select_candidate(class_candidates, "highest annotation-class error count")
        low_iou = (
            matches.groupby("frame_id", as_index=False)["iou"]
            .mean()
            .sort_values(["iou", "frame_id"], ascending=[True, True], kind="stable")
        )
        select_candidate(low_iou, "lowest mean IoU among matched instances")
        for candidate in counts.itertuples(index=False):
            if len(selected_records) >= 4:
                break
            if candidate.frame_id not in used_frames:
                used_frames.add(candidate.frame_id)
                selected_records.append(
                    {
                        "system": system,
                        "frame_id": candidate.frame_id,
                        "error_count": int(candidate.error_count),
                        "selection_reason": "next-highest total recorded errors",
                    }
                )
        selected = pd.DataFrame(selected_records)
        selected_outputs.append(selected)

        prediction_index = index_masks(prediction_root)
        fig, axes = plt.subplots(len(selected), 3, figsize=(10, 2.7 * max(len(selected), 1)), squeeze=False)
        for row_index, selected_row in enumerate(selected.itertuples(index=False)):
            frame_id = selected_row.frame_id
            manifest_inputs.update((image_index[frame_id], gt_index[frame_id], prediction_index[frame_id]))
            image = np.asarray(Image.open(image_index[frame_id]))
            gt = np.asarray(Image.open(gt_index[frame_id]))
            pred = np.asarray(Image.open(prediction_index[frame_id]))
            axes[row_index, 0].imshow(image, cmap="gray")
            axes[row_index, 1].imshow(overlay_boundaries(image, gt, (0.0, 0.45, 0.70)))
            axes[row_index, 2].imshow(overlay_boundaries(image, pred, (0.84, 0.37, 0.0)))
            axes[row_index, 0].set_ylabel(f"{frame_id}\nerrors={selected_row.error_count}")
            for column in range(3):
                axes[row_index, column].set_xticks([])
                axes[row_index, column].set_yticks([])
        for column, title in enumerate(("DIC image", "Ground truth boundary", "Prediction boundary")):
            axes[0, column].set_title(title)
        fig.suptitle(f"Representative failure frames: {system}")
        fig.tight_layout()
        fig.savefig(output_dir / f"failure_cases_{system}.png", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / f"failure_cases_{system}.pdf", bbox_inches="tight")
        manifest_outputs.extend(
            (
                output_dir / f"failure_cases_{system}.png",
                output_dir / f"failure_cases_{system}.pdf",
            )
        )
        plt.close(fig)

    if not ranked_outputs:
        raise ValueError("No system error tables were found")
    pd.concat(ranked_outputs, ignore_index=True).to_csv(
        output_dir / "ranked_error_frames.csv", index=False, lineterminator="\n"
    )
    pd.concat(type_outputs, ignore_index=True).sort_values(["system", "error_type"]).to_csv(
        output_dir / "error_type_counts.csv", index=False, lineterminator="\n"
    )
    pd.concat(selected_outputs, ignore_index=True).to_csv(
        output_dir / "failure_case_selection.csv", index=False, lineterminator="\n"
    )
    manifest_outputs.extend(
        (
            output_dir / "ranked_error_frames.csv",
            output_dir / "error_type_counts.csv",
            output_dir / "failure_case_selection.csv",
        )
    )
    write_figure_manifest(
        output_dir,
        Path(__file__),
        manifest_inputs,
        manifest_outputs,
        "Selections are deterministic and source frames are hashed; panels show DIC images with verified ground-truth and prediction boundaries.",
    )


if __name__ == "__main__":
    main()
