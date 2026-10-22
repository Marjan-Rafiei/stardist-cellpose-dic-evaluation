"""Validate count and ALFI annotation-class ratios against ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from code.unified_evaluation import sha256_file, write_json


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence = args.sequence.upper()

    combined_rows: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    for system_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        metrics_path = system_dir / "metrics_by_frame.csv"
        matches_path = system_dir / "matched_instances.csv"
        errors_path = system_dir / "error_cases.csv"
        if not (metrics_path.is_file() and matches_path.is_file() and errors_path.is_file()):
            continue
        metrics = pd.read_csv(metrics_path)
        metrics = metrics[metrics["frame_id"].str.startswith(f"{sequence}_")].copy()
        if metrics.empty:
            continue
        metrics.insert(0, "system", system_dir.name)
        combined_rows.append(metrics)
        matches = pd.read_csv(matches_path)
        errors = pd.read_csv(errors_path)

        gt_total_sum = int(metrics["gt_total"].sum())
        pred_total_sum = int(metrics["pred_total"].sum())
        gt_mitotic_sum = int(metrics["gt_mitotic"].sum())
        pred_mitotic_sum = int(metrics["pred_mitotic"].sum())
        error_counts = (
            errors["error_type"].value_counts().sort_index().astype(int).to_dict()
            if not errors.empty
            else {}
        )
        summaries[system_dir.name] = {
            "frames": int(len(metrics)),
            "ground_truth_total": gt_total_sum,
            "predicted_total": pred_total_sum,
            "total_count_error": pred_total_sum - gt_total_sum,
            "mean_absolute_frame_count_error": float(metrics["total_count_error"].abs().mean()),
            "ground_truth_pooled_mitotic_ratio": gt_mitotic_sum / gt_total_sum if gt_total_sum else 0.0,
            "predicted_pooled_mitotic_ratio": pred_mitotic_sum / pred_total_sum if pred_total_sum else 0.0,
            "mean_frame_mitotic_ratio_error": float(metrics["mitotic_ratio_error"].mean()),
            "mean_absolute_frame_mitotic_ratio_error": float(metrics["mitotic_ratio_error"].abs().mean()),
            "matched_instances": int(len(matches)),
            "matched_class_accuracy": float(matches["class_correct"].mean()) if len(matches) else None,
            "ground_truth_shape_classifier_accuracy_on_matches": (
                float(matches["gt_classifier_correct"].mean()) if len(matches) else None
            ),
            "segmentation_induced_class_flips": (
                int(matches["segmentation_induced_flip"].sum()) if len(matches) else 0
            ),
            "error_type_counts": error_counts,
            "source_hashes": {
                "metrics_by_frame": sha256_file(metrics_path),
                "matched_instances": sha256_file(matches_path),
                "error_cases": sha256_file(errors_path),
            },
        }

    if not combined_rows:
        raise ValueError(f"No evaluated systems found under {results_root}")
    combined = pd.concat(combined_rows, ignore_index=True).sort_values(
        ["system", "frame_id"], kind="stable"
    )
    combined.to_csv(
        output_dir / "analytics_by_frame.csv",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )
    payload = {
        "schema_version": "1.0",
        "sequence": sequence,
        "systems": summaries,
        "interpretation": "descriptive ALFI annotation-class analytics; not a causal treatment analysis",
        "annotation_compartment_limit": "interphase masks are nuclei and mitotic masks are whole-cell regions",
    }
    write_json(output_dir / "analytics_validation.json", payload)


if __name__ == "__main__":
    main()
