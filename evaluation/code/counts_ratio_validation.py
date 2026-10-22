"""Audit per-frame count and mitotic annotation-ratio agreement on MI05."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from code.unified_evaluation import sha256_file, write_json


SYSTEMS = ("stardist_saved", "cellpose_auto", "cellpose_fixed")
SEGMENT_LABELS = ("early", "middle", "late")


def safe_correlation(left: pd.Series, right: pd.Series, method: str) -> float | None:
    valid = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(valid) < 2 or valid["left"].nunique() < 2 or valid["right"].nunique() < 2:
        return None
    if method == "pearson":
        return float(pearsonr(valid["left"], valid["right"]).statistic)
    if method == "spearman":
        return float(spearmanr(valid["left"], valid["right"]).statistic)
    raise ValueError(f"Unsupported correlation method: {method}")


def cancellation_fraction(errors: pd.Series) -> float:
    values = pd.to_numeric(errors, errors="coerce").dropna().to_numpy(dtype=float)
    gross = float(np.abs(values).sum())
    if gross == 0:
        return 0.0
    return float(1.0 - abs(values.sum()) / gross)


def temporal_segment_ids(size: int) -> np.ndarray:
    if size < 3:
        raise ValueError("At least three frames are required for temporal thirds")
    result = np.empty(size, dtype=object)
    for label, indices in zip(SEGMENT_LABELS, np.array_split(np.arange(size), 3)):
        result[indices] = label
    return result


def add_case_flags(frame: pd.DataFrame, ratio_close_threshold: float) -> pd.DataFrame:
    result = frame.copy()
    result["count_wrong"] = result["count_error"] != 0
    result["ratio_close"] = result["ratio_absolute_error"] <= ratio_close_threshold
    result["count_wrong_ratio_close"] = result["count_wrong"] & result["ratio_close"]
    result["final_class_wrong_ratio_close"] = (
        result["final_matched_classification_errors"] > 0
    ) & result["ratio_close"]
    result["gt_shape_class_wrong_ratio_close"] = (
        result["gt_shape_classification_errors"] > 0
    ) & result["ratio_close"]
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--analytics-by-frame", required=True)
    parser.add_argument("--analytics-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ratio-close-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)
    if not 0 < args.ratio_close_threshold < 1:
        raise ValueError("Ratio-close threshold must be between zero and one")

    results_root = Path(args.results_root).resolve()
    analytics_path = Path(args.analytics_by_frame).resolve()
    analytics_summary_path = Path(args.analytics_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    analytics = pd.read_csv(analytics_path)
    required = {
        "system",
        "frame_id",
        "gt_total",
        "pred_total",
        "gt_mitotic",
        "pred_mitotic",
        "gt_mitotic_ratio",
        "pred_mitotic_ratio",
        "total_count_error",
        "mitotic_ratio_error",
    }
    missing = sorted(required - set(analytics.columns))
    if missing:
        raise ValueError(f"Analytics input is missing columns: {missing}")
    if set(analytics["system"].unique()) != set(SYSTEMS):
        raise ValueError("Analytics input does not contain exactly the three approved systems")

    reference_frames: list[str] | None = None
    combined_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    temporal_rows: list[dict[str, object]] = []
    matched_paths: list[Path] = []
    summary_systems: dict[str, object] = {}

    for system in SYSTEMS:
        frame = analytics[analytics["system"] == system].sort_values("frame_id", kind="stable").copy()
        frame = frame.reset_index(drop=True)
        frames = frame["frame_id"].tolist()
        if reference_frames is None:
            reference_frames = frames
        elif frames != reference_frames:
            raise ValueError("Systems do not share the same ordered MI05 frame set")
        if frame["frame_id"].duplicated().any():
            raise ValueError(f"Duplicate frame rows for {system}")
        if (frame["gt_total"] <= 0).any():
            raise ValueError("Ground-truth ratio is undefined for a zero-count frame")

        matches_path = results_root / system / "matched_instances.csv"
        matched_paths.append(matches_path)
        matches = pd.read_csv(matches_path)
        if matches.empty:
            classification_by_frame = pd.DataFrame(
                columns=[
                    "frame_id",
                    "final_matched_classification_errors",
                    "gt_shape_classification_errors",
                    "segmentation_induced_class_flips",
                ]
            )
        else:
            classification_by_frame = (
                matches.assign(
                    final_matched_classification_errors=(~matches["class_correct"].astype(bool)).astype(int),
                    gt_shape_classification_errors=(
                        ~matches["gt_classifier_correct"].astype(bool)
                    ).astype(int),
                    segmentation_induced_class_flips=matches["segmentation_induced_flip"].astype(int),
                )
                .groupby("frame_id", as_index=False)
                .agg(
                    final_matched_classification_errors=(
                        "final_matched_classification_errors",
                        "sum",
                    ),
                    gt_shape_classification_errors=("gt_shape_classification_errors", "sum"),
                    segmentation_induced_class_flips=("segmentation_induced_class_flips", "sum"),
                )
            )
        frame = frame.merge(classification_by_frame, on="frame_id", how="left", validate="one_to_one")
        for column in (
            "final_matched_classification_errors",
            "gt_shape_classification_errors",
            "segmentation_induced_class_flips",
        ):
            frame[column] = frame[column].fillna(0).astype(int)

        frame["frame_index"] = np.arange(1, len(frame) + 1)
        frame["temporal_third"] = temporal_segment_ids(len(frame))
        frame["count_error"] = frame["pred_total"] - frame["gt_total"]
        frame["count_absolute_error"] = frame["count_error"].abs()
        frame["predicted_ratio_defined"] = frame["pred_total"] > 0
        frame["ratio_signed_error"] = np.where(
            frame["predicted_ratio_defined"],
            frame["pred_mitotic_ratio"] - frame["gt_mitotic_ratio"],
            np.nan,
        )
        frame["ratio_absolute_error"] = frame["ratio_signed_error"].abs()
        frame = add_case_flags(frame, args.ratio_close_threshold)
        combined_rows.append(frame)

        ratio_valid = frame[frame["predicted_ratio_defined"]].copy()
        count_errors = frame["count_error"].astype(float)
        ratio_errors = ratio_valid["ratio_signed_error"].astype(float)
        gt_total = int(frame["gt_total"].sum())
        pred_total = int(frame["pred_total"].sum())
        gt_mitotic = int(frame["gt_mitotic"].sum())
        pred_mitotic = int(frame["pred_mitotic"].sum())
        gt_pooled_ratio = gt_mitotic / gt_total
        pred_pooled_ratio = pred_mitotic / pred_total if pred_total else None
        pooled_ratio_error = pred_pooled_ratio - gt_pooled_ratio if pred_pooled_ratio is not None else None
        zero_tolerance = 1e-12

        summary_row = {
            "system": system,
            "frames": len(frame),
            "ground_truth_total": gt_total,
            "predicted_total": pred_total,
            "aggregate_count_error": int(count_errors.sum()),
            "count_bias": float(count_errors.mean()),
            "count_mae": float(count_errors.abs().mean()),
            "count_median_absolute_error": float(count_errors.abs().median()),
            "count_rmse": float(np.sqrt(np.mean(np.square(count_errors)))),
            "count_pearson_correlation": safe_correlation(frame["gt_total"], frame["pred_total"], "pearson"),
            "count_spearman_correlation": safe_correlation(frame["gt_total"], frame["pred_total"], "spearman"),
            "count_error_positive_frames": int((count_errors > 0).sum()),
            "count_error_negative_frames": int((count_errors < 0).sum()),
            "count_error_zero_frames": int((count_errors == 0).sum()),
            "ground_truth_pooled_ratio": gt_pooled_ratio,
            "predicted_pooled_ratio": pred_pooled_ratio,
            "pooled_ratio_error": pooled_ratio_error,
            "pooled_ratio_absolute_error": abs(pooled_ratio_error) if pooled_ratio_error is not None else None,
            "ratio_defined_frames": len(ratio_valid),
            "ratio_undefined_frames": int((~frame["predicted_ratio_defined"]).sum()),
            "ratio_bias": float(ratio_errors.mean()),
            "ratio_mae": float(ratio_errors.abs().mean()),
            "ratio_median_absolute_error": float(ratio_errors.abs().median()),
            "ratio_rmse": float(np.sqrt(np.mean(np.square(ratio_errors)))),
            "ratio_pearson_correlation": safe_correlation(
                ratio_valid["gt_mitotic_ratio"], ratio_valid["pred_mitotic_ratio"], "pearson"
            ),
            "ratio_spearman_correlation": safe_correlation(
                ratio_valid["gt_mitotic_ratio"], ratio_valid["pred_mitotic_ratio"], "spearman"
            ),
            "ratio_error_positive_frames": int((ratio_errors > zero_tolerance).sum()),
            "ratio_error_negative_frames": int((ratio_errors < -zero_tolerance).sum()),
            "ratio_error_zero_frames": int((ratio_errors.abs() <= zero_tolerance).sum()),
            "ratio_error_cancellation_fraction": cancellation_fraction(ratio_errors),
            "ratio_close_threshold": args.ratio_close_threshold,
            "ratio_close_frames": int(frame["ratio_close"].sum()),
            "count_wrong_ratio_close_frames": int(frame["count_wrong_ratio_close"].sum()),
            "final_class_wrong_ratio_close_frames": int(frame["final_class_wrong_ratio_close"].sum()),
            "gt_shape_class_wrong_ratio_close_frames": int(
                frame["gt_shape_class_wrong_ratio_close"].sum()
            ),
            "final_matched_classification_errors": int(
                frame["final_matched_classification_errors"].sum()
            ),
            "gt_shape_classification_errors": int(frame["gt_shape_classification_errors"].sum()),
            "segmentation_induced_class_flips": int(frame["segmentation_induced_class_flips"].sum()),
            "final_classification_errors_in_ratio_close_frames": int(
                frame.loc[frame["ratio_close"], "final_matched_classification_errors"].sum()
            ),
            "gt_shape_classification_errors_in_ratio_close_frames": int(
                frame.loc[frame["ratio_close"], "gt_shape_classification_errors"].sum()
            ),
        }
        summary_rows.append(summary_row)

        for segment in SEGMENT_LABELS:
            subset = frame[frame["temporal_third"] == segment]
            valid_subset = subset[subset["predicted_ratio_defined"]]
            segment_count_error = subset["count_error"].astype(float)
            segment_ratio_error = valid_subset["ratio_signed_error"].astype(float)
            temporal_rows.append(
                {
                    "system": system,
                    "temporal_third": segment,
                    "first_frame": subset["frame_id"].iloc[0],
                    "last_frame": subset["frame_id"].iloc[-1],
                    "frames": len(subset),
                    "count_bias": float(segment_count_error.mean()),
                    "count_mae": float(segment_count_error.abs().mean()),
                    "ratio_bias": float(segment_ratio_error.mean()),
                    "ratio_mae": float(segment_ratio_error.abs().mean()),
                    "ratio_error_cancellation_fraction": cancellation_fraction(segment_ratio_error),
                    "ratio_close_frames": int(subset["ratio_close"].sum()),
                }
            )

        summary_systems[system] = {
            "count": {
                "aggregate_error": summary_row["aggregate_count_error"],
                "bias": summary_row["count_bias"],
                "mae": summary_row["count_mae"],
                "pearson_correlation_supplementary": summary_row["count_pearson_correlation"],
            },
            "ratio": {
                "pooled_error": summary_row["pooled_ratio_error"],
                "frame_bias": summary_row["ratio_bias"],
                "frame_mae": summary_row["ratio_mae"],
                "cancellation_fraction": summary_row["ratio_error_cancellation_fraction"],
                "pearson_correlation_supplementary": summary_row["ratio_pearson_correlation"],
            },
            "ratio_close_cases": {
                "threshold": args.ratio_close_threshold,
                "frames": summary_row["ratio_close_frames"],
                "with_wrong_count": summary_row["count_wrong_ratio_close_frames"],
                "with_final_matched_classification_error": summary_row[
                    "final_class_wrong_ratio_close_frames"
                ],
            },
        }

    by_frame = pd.concat(combined_rows, ignore_index=True).sort_values(
        ["system", "frame_index"], kind="stable"
    )
    summary_frame = pd.DataFrame(summary_rows)
    temporal_frame = pd.DataFrame(temporal_rows)
    cases = by_frame[
        by_frame["count_wrong_ratio_close"]
        | by_frame["final_class_wrong_ratio_close"]
        | by_frame["gt_shape_class_wrong_ratio_close"]
    ].copy()

    output_frames = {
        "agreement_by_frame.csv": by_frame,
        "agreement_summary.csv": summary_frame,
        "temporal_segment_summary.csv": temporal_frame,
        "cancellation_cases.csv": cases,
    }
    for name, output_frame in output_frames.items():
        output_frame.to_csv(
            output_dir / name,
            index=False,
            lineterminator="\n",
            float_format="%.10g",
        )

    validations = {
        "systems_exact": set(summary_frame["system"]) == set(SYSTEMS),
        "common_frame_set": all(
            by_frame[by_frame["system"] == system]["frame_id"].tolist() == reference_frames
            for system in SYSTEMS
        ),
        "all_ground_truth_ratios_defined": bool((by_frame["gt_total"] > 0).all()),
        "all_predicted_ratios_defined": bool(by_frame["predicted_ratio_defined"].all()),
        "stored_count_errors_match": bool(
            (by_frame["count_error"] == by_frame["total_count_error"]).all()
        ),
        "stored_ratio_errors_match": bool(
            np.allclose(
                by_frame["ratio_signed_error"],
                by_frame["mitotic_ratio_error"],
                atol=5e-10,
                rtol=0,
            )
        ),
        "temporal_thirds_cover_all_rows": len(temporal_frame) == len(SYSTEMS) * 3,
        "mi05_only": bool(by_frame["frame_id"].str.startswith("MI05_").all()),
    }
    if not all(validations.values()):
        raise ValueError(f"One or more validation checks failed: {validations}")

    summary_path = output_dir / "analysis_summary.json"
    write_json(
        summary_path,
        {
            "schema_version": "1.0",
            "sequence": "MI05",
            "systems": summary_systems,
            "ratio_close_definition": (
                f"absolute per-frame ratio error at most {100 * args.ratio_close_threshold:.1f} percentage points; "
                "descriptive tolerance, not a biological accuracy standard"
            ),
            "interpretation": {
                "count": "bias and MAE measure agreement; correlation is supplementary and cannot establish agreement",
                "ratio": "pooled and signed errors can look small through opposing per-frame errors and denominator weighting",
                "classification": "matched classification-error counts do not label false-positive classes but demonstrate that close aggregate ratios can coexist with known class errors",
                "temporal": "early, middle, and late thirds are descriptive summaries of one related sequence, not independent experimental groups",
            },
            "validations": validations,
        },
    )

    output_paths = [output_dir / name for name in output_frames] + [summary_path]
    manifest_path = output_dir / "analysis_manifest.json"
    input_paths = [analytics_path, analytics_summary_path, *matched_paths]
    write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "status": "verified MI05 per-frame count and mitotic annotation-ratio agreement audit",
            "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
            "parameters": {
                "sequence": "MI05",
                "ratio_close_threshold": args.ratio_close_threshold,
                "temporal_segments": list(SEGMENT_LABELS),
            },
            "actions_performed": {
                "model_inference": False,
                "classifier_training": False,
                "threshold_tuning": False,
                "descriptive_post_evaluation_analysis": True,
            },
            "inputs": [
                {"path": str(path), "sha256": sha256_file(path)} for path in input_paths
            ],
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path)} for path in output_paths
            ],
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        },
    )


if __name__ == "__main__":
    main()
