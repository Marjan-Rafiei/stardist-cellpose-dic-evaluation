"""Build machine-readable headline tables only from verified result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from code.unified_evaluation import sha256_file, write_json


SYSTEMS = ("stardist_saved", "cellpose_auto", "cellpose_fixed")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.results_root).resolve()
    analytics = read_json(root / "analytics_validation" / "analytics_validation.json")["systems"]
    temporal_bootstrap_path = root / "temporal_bootstrap" / "analysis_summary.json"
    bootstrap = read_json(temporal_bootstrap_path)["bootstrap"]["chosen_f1_intervals"]
    rows = []
    source_files: list[Path] = []
    for system in SYSTEMS:
        metric_path = root / system / "metrics_summary.json"
        class_path = root / system / "classifier_metrics.json"
        metrics = read_json(metric_path)
        classes = read_json(class_path)
        source_files.extend((metric_path, class_path))
        rows.append(
            {
                "system": system,
                "sequence": metrics["sequence"],
                "iou_threshold": metrics["iou_threshold"],
                "pooled_precision": metrics["pooled"]["precision"],
                "pooled_recall": metrics["pooled"]["recall"],
                "pooled_f1": metrics["pooled"]["f1"],
                "pooled_pq": metrics["pooled"]["pq"],
                "pooled_matched_iou": metrics["pooled"]["matched_iou"],
                "frame_macro_f1": metrics["per_frame_macro"]["f1"]["mean"],
                "frame_macro_pq": metrics["per_frame_macro"]["pq"]["mean"],
                "classification_only_accuracy": classes["classification_only"]["accuracy"],
                "classification_only_balanced_accuracy": classes["classification_only"]["balanced_accuracy"],
                "classification_only_macro_f1": classes["classification_only"]["macro_f1"],
                "end_to_end_macro_f1": classes["end_to_end"]["macro_f1"],
                "end_to_end_interphase_f1": classes["end_to_end"]["per_class"]["interphase"]["f1"],
                "end_to_end_mitotic_f1": classes["end_to_end"]["per_class"]["mitotic"]["f1"],
                "ground_truth_total": analytics[system]["ground_truth_total"],
                "predicted_total": analytics[system]["predicted_total"],
                "ground_truth_pooled_mitotic_ratio": analytics[system]["ground_truth_pooled_mitotic_ratio"],
                "predicted_pooled_mitotic_ratio": analytics[system]["predicted_pooled_mitotic_ratio"],
                "segmentation_induced_class_flips": analytics[system]["segmentation_induced_class_flips"],
                "block_bootstrap_f1_lower": bootstrap[system]["percentile_2_5"],
                "block_bootstrap_f1_upper": bootstrap[system]["percentile_97_5"],
            }
        )
    comparison_path = root / "comparison_summary.csv"
    pd.DataFrame(rows).to_csv(
        comparison_path,
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    validation_path = root / "classifier_v1" / "classifier_validation.json"
    validation = read_json(validation_path)
    report = validation["classification_report"]
    classifier_summary_path = root / "classifier_validation_summary.csv"
    pd.DataFrame(
        [
            {
                "training_rows": validation["training_rows"],
                "validation_rows": validation["validation_rows"],
                "accuracy": report["accuracy"],
                "balanced_accuracy": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
                "interphase_f1": report["interphase"]["f1-score"],
                "mitotic_f1": report["mitotic"]["f1-score"],
            }
        ]
    ).to_csv(classifier_summary_path, index=False, lineterminator="\n", float_format="%.10g")
    source_files.extend(
        (
            root / "analytics_validation" / "analytics_validation.json",
            temporal_bootstrap_path,
            validation_path,
            root / "classifier_v1" / "classifier_manifest.json",
            comparison_path,
            classifier_summary_path,
        )
    )
    optional_verified_paths = (
        root / "iou_sensitivity" / "analysis_manifest.json",
        root / "iou_sensitivity" / "iou_sensitivity_metrics.csv",
        root / "iou_sensitivity" / "iou_sensitivity_table.tex",
        root / "iou_sensitivity" / "figures" / "figure_manifest.json",
        root / "temporal_bootstrap" / "analysis_manifest.json",
        root / "temporal_bootstrap" / "analysis_summary.json",
        root / "temporal_bootstrap" / "autocorrelation.csv",
        root / "temporal_bootstrap" / "bootstrap_interval_sensitivity.csv",
        root / "temporal_bootstrap" / "bootstrap_interval_table.tex",
        root / "temporal_bootstrap" / "figures" / "figure_manifest.json",
        root / "segmentation_analytics_error_chain" / "analysis_manifest.json",
        root / "segmentation_analytics_error_chain" / "analysis_summary.json",
        root / "segmentation_analytics_error_chain" / "error_chain_summary.csv",
        root / "segmentation_analytics_error_chain" / "mitotic_ratio_bridge.csv",
        root / "segmentation_analytics_error_chain" / "count_decomposition.csv",
        root / "segmentation_analytics_error_chain" / "error_chain_table.tex",
        root / "segmentation_analytics_error_chain" / "figures" / "figure_manifest.json",
        root / "classification_transitions" / "analysis_manifest.json",
        root / "classification_transitions" / "analysis_summary.json",
        root / "classification_transitions" / "transition_summary.csv",
        root / "classification_transitions" / "transition_counts.csv",
        root / "classification_transitions" / "transition_by_class.csv",
        root / "classification_transitions" / "transition_matrix_table.tex",
        root / "classification_transitions" / "figures" / "figure_manifest.json",
        root / "morphology_domain_shift" / "analysis_manifest.json",
        root / "morphology_domain_shift" / "analysis_summary.json",
        root / "morphology_domain_shift" / "feature_shift_summary.csv",
        root / "morphology_domain_shift" / "common_subset_feature_shift_summary.csv",
        root / "morphology_domain_shift" / "paired_feature_differences.csv",
        root / "morphology_domain_shift" / "median_relative_shift_table.tex",
        root / "morphology_domain_shift" / "figures" / "figure_manifest.json",
        root / "random_forest_audit" / "analysis_manifest.json",
        root / "random_forest_audit" / "audit_summary.json",
        root / "random_forest_audit" / "benchmark_metrics.csv",
        root / "random_forest_audit" / "benchmark_predictions.csv",
        root / "random_forest_audit" / "class_balance.csv",
        root / "random_forest_audit" / "feature_distributions.csv",
        root / "random_forest_audit" / "feature_correlations.csv",
        root / "random_forest_audit" / "logistic_coefficients.csv",
        root / "random_forest_audit" / "benchmark_table.tex",
        root / "random_forest_audit" / "figures" / "figure_manifest.json",
        root / "feature_importance_audit" / "analysis_manifest.json",
        root / "feature_importance_audit" / "audit_summary.json",
        root / "feature_importance_audit" / "individual_importance_runs.csv",
        root / "feature_importance_audit" / "individual_importance_draws.csv",
        root / "feature_importance_audit" / "individual_importance_stability.csv",
        root / "feature_importance_audit" / "grouped_importance_runs.csv",
        root / "feature_importance_audit" / "grouped_importance_draws.csv",
        root / "feature_importance_audit" / "grouped_importance_summary.csv",
        root / "feature_importance_audit" / "feature_correlations.csv",
        root / "feature_importance_audit" / "high_correlation_pairs.csv",
        root / "feature_importance_audit" / "figures" / "figure_manifest.json",
        root / "counts_ratio_validation" / "analysis_manifest.json",
        root / "counts_ratio_validation" / "analysis_summary.json",
        root / "counts_ratio_validation" / "agreement_by_frame.csv",
        root / "counts_ratio_validation" / "agreement_summary.csv",
        root / "counts_ratio_validation" / "temporal_segment_summary.csv",
        root / "counts_ratio_validation" / "cancellation_cases.csv",
        root / "counts_ratio_validation" / "figures" / "figure_manifest.json",
        root / "dataset_quality_audit" / "quality_manifest.json",
        root / "dataset_quality_audit" / "quality_summary.json",
        root / "dataset_quality_audit" / "sequence_summary.csv",
        root / "dataset_quality_audit" / "sequence_hashes.csv",
        root / "dataset_quality_audit" / "duplicate_groups.csv",
        root / "dataset_quality_audit" / "quality_issues.csv",
        root / "mask_conversion_audit" / "analysis_manifest.json",
        root / "mask_conversion_audit" / "conversion_summary.json",
        root / "mask_conversion_audit" / "mask_comparison.csv",
        root / "mask_conversion_audit" / "value_mapping.csv",
        root / "mask_conversion_audit" / "figures" / "figure_manifest.json",
    )
    source_files.extend(path for path in optional_verified_paths if path.is_file())
    write_json(
        root / "results_index.json",
        {
            "schema_version": "1.0",
            "status": "verified analytical outputs; approved for thesis integration",
            "sources": [
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in sorted(source_files)
            ],
            "determinism_check": {
                "relative_path": "../determinism_check.json",
                "sha256": sha256_file(root.parent / "determinism_check.json"),
            },
        },
    )


if __name__ == "__main__":
    main()
