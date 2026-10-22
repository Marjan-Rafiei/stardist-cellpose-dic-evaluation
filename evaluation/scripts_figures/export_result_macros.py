"""Export verified thesis values as LaTeX macros.

This script is intentionally small and deterministic.  It reads only the
machine-readable files under results/verified, verifies their hashes against
results_index.json where entries are available, and writes the numerical
macros used by the thesis prose.  Figures and tables are generated separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: str | float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}\\%"


def command(name: str, value: str | int) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    root = args.results_root.resolve()
    sources = {
        "comparison": root / "comparison_summary.csv",
        "classifier": root / "classifier_validation_summary.csv",
        "classifier_manifest": root / "classifier_v1" / "classifier_manifest.json",
        "dataset": root / "dataset_audit" / "dataset_manifest.json",
        "analytics": root / "analytics_validation" / "analytics_validation.json",
        "bootstrap": root / "temporal_bootstrap" / "analysis_summary.json",
        "stardist_metrics": root / "stardist_saved" / "metrics_summary.json",
        "stardist_classes": root / "stardist_saved" / "classifier_metrics.json",
        "cellpose_auto_metrics": root / "cellpose_auto" / "metrics_summary.json",
        "cellpose_auto_classes": root / "cellpose_auto" / "classifier_metrics.json",
        "cellpose_fixed_metrics": root / "cellpose_fixed" / "metrics_summary.json",
        "cellpose_fixed_classes": root / "cellpose_fixed" / "classifier_metrics.json",
        "stardist_predictions": root / "stardist_saved" / "predictions" / "prediction_manifest.json",
        "mask_conversion": root / "mask_conversion_audit" / "conversion_summary.json",
        "error_chain": root / "segmentation_analytics_error_chain" / "error_chain_summary.csv",
        "transitions": root / "classification_transitions" / "transition_summary.csv",
        "morphology": root / "morphology_domain_shift" / "feature_shift_summary.csv",
        "morphology_common": root / "morphology_domain_shift" / "common_subset_feature_shift_summary.csv",
        "agreement": root / "counts_ratio_validation" / "agreement_summary.csv",
        "rf_audit": root / "random_forest_audit" / "audit_summary.json",
        "iou_sensitivity": root / "iou_sensitivity" / "iou_sensitivity_metrics.csv",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing verified sources: " + ", ".join(missing))

    index = read_json(root / "results_index.json")
    indexed_hashes = {
        entry["relative_path"].replace("\\", "/"): entry["sha256"]
        for entry in index["sources"]
    }
    verified_hashes: dict[str, str] = {}
    for label, path in sources.items():
        actual = sha256(path)
        relative = path.relative_to(root).as_posix()
        expected = indexed_hashes.get(relative)
        if expected is not None and actual != expected:
            raise ValueError(f"Hash mismatch for {relative}: {actual} != {expected}")
        verified_hashes[label] = actual

    comparison = {row["system"]: row for row in read_csv_rows(sources["comparison"])}
    classifier = read_csv_rows(sources["classifier"])[0]
    classifier_manifest = read_json(sources["classifier_manifest"])
    dataset = read_json(sources["dataset"])
    analytics = read_json(sources["analytics"])
    bootstrap = read_json(sources["bootstrap"])["bootstrap"]
    prediction_manifest = read_json(sources["stardist_predictions"])
    mask_conversion = read_json(sources["mask_conversion"])
    error_chain = {row["system"]: row for row in read_csv_rows(sources["error_chain"])}
    transitions = {row["system"]: row for row in read_csv_rows(sources["transitions"])}
    morphology = {
        (row["system"], row["feature"]): row
        for row in read_csv_rows(sources["morphology"])
    }
    morphology_common = read_csv_rows(sources["morphology_common"])
    agreement = {row["system"]: row for row in read_csv_rows(sources["agreement"])}
    rf_audit = read_json(sources["rf_audit"])
    iou_sensitivity = read_csv_rows(sources["iou_sensitivity"])

    sd = comparison["stardist_saved"]
    ca = comparison["cellpose_auto"]
    cf = comparison["cellpose_fixed"]
    sd_metrics = read_json(sources["stardist_metrics"])["pooled"]
    ca_metrics = read_json(sources["cellpose_auto_metrics"])["pooled"]
    cf_metrics = read_json(sources["cellpose_fixed_metrics"])["pooled"]
    sd_classes = read_json(sources["stardist_classes"])
    ca_classes = read_json(sources["cellpose_auto_classes"])
    cf_classes = read_json(sources["cellpose_fixed_classes"])
    sd_analytics = analytics["systems"]["stardist_saved"]
    ca_analytics = analytics["systems"]["cellpose_auto"]
    cf_analytics = analytics["systems"]["cellpose_fixed"]

    training_frames = sum(dataset["counts"][seq]["aligned_core"] for seq in ("MI01", "MI02", "MI03"))
    lines = [
        "% Generated by scripts_figures/export_result_macros.py.",
        "% Do not edit by hand. Regenerate from results/verified.",
        command("TotalAlignedFrames", dataset["total_aligned_images"]),
        command("TrainingAlignedFrames", training_frames),
        command("ValidationFrames", dataset["counts"]["MI04"]["aligned_core"]),
        command("TestFrames", dataset["counts"]["MI05"]["aligned_core"]),
        command("MIOneImages", dataset["counts"]["MI01"]["image"]),
        command("MIOneAnnotationFiles", dataset["counts"]["MI01"]["raw_mask"]),
        command("MaskAuditFiles", mask_conversion["total_masks"]),
        command("MaskAuditMasksWithValueTwoFiveFive", mask_conversion["masks_with_value_255"]),
        command("MaskAuditValueTwoFiveFivePixels", f"{mask_conversion['total_value_255_pixels']:,}"),
        command("MaskAuditInstanceMatches", mask_conversion["corrected_instance_exact_matches"]),
        command("MaskAuditClassMatches", mask_conversion["corrected_class_exact_matches"]),
        command("MaskAuditFaultOutput", mask_conversion["fault_consequence"]["value_255_after_fault"]),
        command("ClassifierTrainingRows", classifier["training_rows"]),
        command("ClassifierValidationRows", classifier["validation_rows"]),
        command("ClassifierValidationAccuracy", fmt(classifier["accuracy"])),
        command("ClassifierValidationBalancedAccuracy", fmt(classifier["balanced_accuracy"])),
        command("ClassifierValidationMacroFOne", fmt(classifier["macro_f1"])),
        command("ClassifierValidationInterphaseFOne", fmt(classifier["interphase_f1"])),
        command("ClassifierValidationMitoticFOne", fmt(classifier["mitotic_f1"])),
        command("LogisticBenchmarkMacroFOne", fmt(rf_audit["comparison"]["logistic_regression_macro_f1"])),
        command("ClassifierTrees", classifier_manifest["classifier_parameters"]["n_estimators"]),
        command("ClassifierSeed", classifier_manifest["classifier_parameters"]["random_state"]),
        command("ClassifierJobs", classifier_manifest["classifier_parameters"]["n_jobs"]),
        command("EvaluationIoUThreshold", fmt(sd["iou_threshold"], 1)),
        command("StarDistSavedProbabilityThreshold", fmt(prediction_manifest["thresholds"]["probability"], 4)),
        command("StarDistPooledPrecision", fmt(sd["pooled_precision"])),
        command("StarDistPooledRecall", fmt(sd["pooled_recall"])),
        command("StarDistPooledFOne", fmt(sd["pooled_f1"])),
        command("StarDistPooledPQ", fmt(sd["pooled_pq"])),
        command("StarDistMatchedIoU", fmt(sd["pooled_matched_iou"])),
        command("CellposeAutoPooledPrecision", fmt(ca["pooled_precision"])),
        command("CellposeAutoPooledRecall", fmt(ca["pooled_recall"])),
        command("CellposeAutoPooledFOne", fmt(ca["pooled_f1"])),
        command("CellposeAutoPooledPQ", fmt(ca["pooled_pq"])),
        command("CellposeAutoMatchedIoU", fmt(ca["pooled_matched_iou"])),
        command("CellposeFixedPooledPrecision", fmt(cf["pooled_precision"])),
        command("CellposeFixedPooledRecall", fmt(cf["pooled_recall"])),
        command("CellposeFixedPooledFOne", fmt(cf["pooled_f1"])),
        command("CellposeFixedPooledPQ", fmt(cf["pooled_pq"])),
        command("CellposeFixedMatchedIoU", fmt(cf["pooled_matched_iou"])),
        command("StarDistTruePositives", sd_metrics["tp"]),
        command("StarDistFalsePositives", sd_metrics["fp"]),
        command("StarDistFalseNegatives", sd_metrics["fn"]),
        command("CellposeAutoTruePositives", ca_metrics["tp"]),
        command("CellposeAutoFalsePositives", ca_metrics["fp"]),
        command("CellposeAutoFalseNegatives", ca_metrics["fn"]),
        command("CellposeFixedTruePositives", cf_metrics["tp"]),
        command("CellposeFixedFalsePositives", cf_metrics["fp"]),
        command("CellposeFixedFalseNegatives", cf_metrics["fn"]),
        command("StarDistClassificationMacroFOne", fmt(sd_classes["classification_only"]["macro_f1"])),
        command("CellposeAutoClassificationMacroFOne", fmt(ca_classes["classification_only"]["macro_f1"])),
        command("CellposeFixedClassificationMacroFOne", fmt(cf_classes["classification_only"]["macro_f1"])),
        command("StarDistEndToEndMacroFOne", fmt(sd_classes["end_to_end"]["macro_f1"])),
        command("CellposeAutoEndToEndMacroFOne", fmt(ca_classes["end_to_end"]["macro_f1"])),
        command("CellposeFixedEndToEndMacroFOne", fmt(cf_classes["end_to_end"]["macro_f1"])),
        command("StarDistEndToEndInterphaseFOne", fmt(sd_classes["end_to_end"]["per_class"]["interphase"]["f1"])),
        command("StarDistEndToEndMitoticFOne", fmt(sd_classes["end_to_end"]["per_class"]["mitotic"]["f1"])),
        command("CellposeAutoEndToEndInterphaseFOne", fmt(ca_classes["end_to_end"]["per_class"]["interphase"]["f1"])),
        command("CellposeAutoEndToEndMitoticFOne", fmt(ca_classes["end_to_end"]["per_class"]["mitotic"]["f1"])),
        command("CellposeFixedEndToEndInterphaseFOne", fmt(cf_classes["end_to_end"]["per_class"]["interphase"]["f1"])),
        command("CellposeFixedEndToEndMitoticFOne", fmt(cf_classes["end_to_end"]["per_class"]["mitotic"]["f1"])),
        command("TestGroundTruthObjects", sd_analytics["ground_truth_total"]),
        command("TestGroundTruthMitoticRatio", pct(sd_analytics["ground_truth_pooled_mitotic_ratio"])),
        command("StarDistPredictedObjects", sd_analytics["predicted_total"]),
        command("CellposeAutoPredictedObjects", ca_analytics["predicted_total"]),
        command("CellposeFixedPredictedObjects", cf_analytics["predicted_total"]),
        command("StarDistPredictedMitoticRatio", pct(sd_analytics["predicted_pooled_mitotic_ratio"])),
        command("CellposeAutoPredictedMitoticRatio", pct(ca_analytics["predicted_pooled_mitotic_ratio"])),
        command("CellposeFixedPredictedMitoticRatio", pct(cf_analytics["predicted_pooled_mitotic_ratio"])),
        command("StarDistSegmentationClassFlips", sd_analytics["segmentation_induced_class_flips"]),
        command("CellposeAutoSegmentationClassFlips", ca_analytics["segmentation_induced_class_flips"]),
        command("CellposeFixedSegmentationClassFlips", cf_analytics["segmentation_induced_class_flips"]),
        command("StarDistClassOutputChanges", transitions["stardist_saved"]["classifier_output_changes"]),
        command("CellposeAutoClassOutputChanges", transitions["cellpose_auto"]["classifier_output_changes"]),
        command("CellposeFixedClassOutputChanges", transitions["cellpose_fixed"]["classifier_output_changes"]),
        command("StarDistIncorrectToCorrect", transitions["stardist_saved"]["incorrect_to_correct"]),
        command("CellposeAutoIncorrectToCorrect", transitions["cellpose_auto"]["incorrect_to_correct"]),
        command("CellposeFixedIncorrectToCorrect", transitions["cellpose_fixed"]["incorrect_to_correct"]),
        command("StarDistMorphologyAreaShift", fmt(morphology[("stardist_saved", "area")]["median_relative_difference_percent"], 1)),
        command("CellposeAutoMorphologyAreaShift", fmt(morphology[("cellpose_auto", "area")]["median_relative_difference_percent"], 1)),
        command("CellposeFixedMorphologyAreaShift", fmt(morphology[("cellpose_fixed", "area")]["median_relative_difference_percent"], 1)),
        command("CommonMatchedInstances", morphology_common[0]["matched_instances"]),
        command("CellposeAutoInterphaseFalseNegatives", error_chain["cellpose_auto"]["fn_interphase"]),
        command("CellposeFixedInterphaseFalseNegatives", error_chain["cellpose_fixed"]["fn_interphase"]),
        command("CellposeAutoMitoticFalseNegatives", error_chain["cellpose_auto"]["fn_mitotic"]),
        command("CellposeFixedMitoticFalseNegatives", error_chain["cellpose_fixed"]["fn_mitotic"]),
        command("CellposeAutoMatchedGroundTruthMitoticRatio", pct(error_chain["cellpose_auto"]["matched_gt_mitotic_ratio"])),
        command("CellposeFixedMatchedGroundTruthMitoticRatio", pct(error_chain["cellpose_fixed"]["matched_gt_mitotic_ratio"])),
        command("CellposeFixedCountWrongRatioCloseFrames", agreement["cellpose_fixed"]["count_wrong_ratio_close_frames"]),
        command("CellposeFixedRatioCloseFrames", agreement["cellpose_fixed"]["ratio_close_frames"]),
        command("CellposeFixedRatioFrameMAE", fmt(100 * float(agreement["cellpose_fixed"]["ratio_mae"]), 2)),
        command("CellposeFixedRatioCancellation", pct(agreement["cellpose_fixed"]["ratio_error_cancellation_fraction"])),
        command("IoUSensitivityMinimum", fmt(min(float(row["iou_threshold"]) for row in iou_sensitivity), 1)),
        command("IoUSensitivityMaximum", fmt(max(float(row["iou_threshold"]) for row in iou_sensitivity), 1)),
        command("BootstrapReplicates", bootstrap["replicates"]),
        command("BootstrapBlockLength", bootstrap["chosen_block_length"]),
        command("BootstrapSeed", bootstrap["seed"]),
        command("StarDistBootstrapFOneLower", fmt(bootstrap["chosen_f1_intervals"]["stardist_saved"]["percentile_2_5"])),
        command("StarDistBootstrapFOneUpper", fmt(bootstrap["chosen_f1_intervals"]["stardist_saved"]["percentile_97_5"])),
        command("CellposeAutoBootstrapFOneLower", fmt(bootstrap["chosen_f1_intervals"]["cellpose_auto"]["percentile_2_5"])),
        command("CellposeAutoBootstrapFOneUpper", fmt(bootstrap["chosen_f1_intervals"]["cellpose_auto"]["percentile_97_5"])),
        command("CellposeFixedBootstrapFOneLower", fmt(bootstrap["chosen_f1_intervals"]["cellpose_fixed"]["percentile_2_5"])),
        command("CellposeFixedBootstrapFOneUpper", fmt(bootstrap["chosen_f1_intervals"]["cellpose_fixed"]["percentile_97_5"])),
        "",
    ]

    content = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="\n")

    manifest = {
        "generator": "scripts_figures/export_result_macros.py",
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "results_index_sha256": sha256(root / "results_index.json"),
        "sources": {
            label: {"path": str(path), "sha256": verified_hashes[label]}
            for label, path in sorted(sources.items())
        },
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
