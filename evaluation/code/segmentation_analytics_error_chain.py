"""Quantify how MI05 segmentation errors propagate into cell analytics.

The analysis uses accepted evaluator tables only. It performs no inference,
training, threshold tuning, or classifier fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from code.unified_evaluation import package_versions, sha256_file, write_json


SYSTEM_ORDER = ("stardist_saved", "cellpose_auto", "cellpose_fixed")
DISPLAY_NAMES = {
    "stardist_saved": "StarDist",
    "cellpose_auto": "Cellpose automatic",
    "cellpose_fixed": "Cellpose fixed",
}


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("ratio denominator must be positive")
    return numerator / denominator


def build_ratio_bridge(
    system: str,
    gt_mitotic: int,
    gt_total: int,
    matched_gt_mitotic: int,
    matched_total: int,
    false_positive_mitotic: int,
    false_positives: int,
    gt_shape_classifier_mitotic: int,
    pred_shape_classifier_mitotic: int,
) -> list[dict[str, Any]]:
    stages = (
        (
            "ground_truth",
            "Ground truth",
            gt_mitotic,
            gt_total,
            "All annotated instances and annotation classes.",
        ),
        (
            "matched_gt_only",
            "Matched GT only",
            matched_gt_mitotic,
            matched_total,
            "Unmatched annotated instances are removed; matched instances keep annotation classes.",
        ),
        (
            "false_positives_added",
            "Add false positives",
            matched_gt_mitotic + false_positive_mitotic,
            matched_total + false_positives,
            "Unmatched predictions are added with their predicted annotation classes.",
        ),
        (
            "gt_shape_classifier",
            "GT-shape classifier",
            gt_shape_classifier_mitotic + false_positive_mitotic,
            matched_total + false_positives,
            "Matched annotation classes are replaced by classifier outputs from ground-truth morphology.",
        ),
        (
            "pred_shape_classifier",
            "Pred-shape classifier",
            pred_shape_classifier_mitotic + false_positive_mitotic,
            matched_total + false_positives,
            "Matched classifier outputs use predicted-mask morphology, yielding the observed prediction ratio.",
        ),
    )
    rows: list[dict[str, Any]] = []
    previous: float | None = None
    for order, (stage, label, numerator, denominator, description) in enumerate(stages):
        ratio = safe_ratio(numerator, denominator)
        rows.append(
            {
                "system": system,
                "stage_order": order,
                "stage": stage,
                "stage_label": label,
                "mitotic_numerator": numerator,
                "total_denominator": denominator,
                "mitotic_ratio": ratio,
                "change_from_prior": None if previous is None else ratio - previous,
                "description": description,
            }
        )
        previous = ratio
    return rows


def build_count_decomposition(
    system: str,
    gt_total: int,
    false_negatives: int,
    false_positives: int,
    gt_mitotic: int,
    false_negative_mitotic: int,
    false_positive_mitotic: int,
    baseline_classifier_net_mitotic: int,
    predicted_shape_net_mitotic: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_components = (
        ("ground_truth", gt_total),
        ("false_negatives", -false_negatives),
        ("false_positives", false_positives),
    )
    running = 0
    for order, (component, contribution) in enumerate(total_components):
        running = contribution if order == 0 else running + contribution
        rows.append(
            {
                "system": system,
                "quantity": "total_count",
                "component_order": order,
                "component": component,
                "contribution": contribution,
                "running_total": running,
            }
        )

    mitotic_components = (
        ("ground_truth", gt_mitotic),
        ("false_negative_mitotic", -false_negative_mitotic),
        ("false_positive_mitotic", false_positive_mitotic),
        ("gt_shape_classifier_net", baseline_classifier_net_mitotic),
        ("pred_shape_transition_net", predicted_shape_net_mitotic),
    )
    running = 0
    for order, (component, contribution) in enumerate(mitotic_components):
        running = contribution if order == 0 else running + contribution
        rows.append(
            {
                "system": system,
                "quantity": "mitotic_count",
                "component_order": order,
                "component": component,
                "contribution": contribution,
                "running_total": running,
            }
        )
    return rows


def build_tex_table(frame: pd.DataFrame) -> str:
    lines = [
        "% Generated by code/segmentation_analytics_error_chain.py.",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Method & FN & FP & Count error & FN rate I & FN rate M & Matched macro F1 & End-to-end macro F1 \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{DISPLAY_NAMES[row.system]} & {row.false_negatives:d} & {row.false_positives:d} & "
            f"{row.total_count_error:+d} & {row.fn_rate_interphase:.3f} & "
            f"{row.fn_rate_mitotic:.3f} & {row.classification_only_macro_f1:.3f} & "
            f"{row.end_to_end_macro_f1:.3f} \\\\" 
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--sequence", default="MI05")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    root = Path(args.results_root).resolve()
    sequence = args.sequence.upper()
    output_dir = Path(args.output_dir).resolve()
    input_paths: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    validations: dict[str, Any] = {}
    reference_gt: tuple[int, int, int] | None = None

    for system in SYSTEM_ORDER:
        system_dir = root / system
        metrics_path = system_dir / "metrics_by_frame.csv"
        matches_path = system_dir / "matched_instances.csv"
        errors_path = system_dir / "error_cases.csv"
        classifier_path = system_dir / "classifier_metrics.json"
        input_paths.extend((metrics_path, matches_path, errors_path, classifier_path))

        metrics = pd.read_csv(metrics_path)
        metrics = metrics.loc[metrics["frame_id"].str.startswith(f"{sequence}_")].copy()
        matches = pd.read_csv(matches_path)
        matches = matches.loc[matches["frame_id"].str.startswith(f"{sequence}_")].copy()
        errors = pd.read_csv(errors_path)
        errors = errors.loc[errors["frame_id"].str.startswith(f"{sequence}_")].copy()
        classifier = json.loads(classifier_path.read_text(encoding="utf-8"))

        if len(metrics) != 69:
            raise ValueError(f"Expected 69 {sequence} frames for {system}, found {len(metrics)}")
        if matches.duplicated(["frame_id", "gt_id"]).any():
            raise ValueError(f"Ground-truth match duplication for {system}")
        if matches.duplicated(["frame_id", "pred_id"]).any():
            raise ValueError(f"Prediction match duplication for {system}")

        false_negatives_frame = errors.loc[errors["error_type"] == "false_negative"]
        false_positives_frame = errors.loc[errors["error_type"] == "false_positive"]
        tp = int(metrics["tp"].sum())
        fp = int(metrics["fp"].sum())
        fn = int(metrics["fn"].sum())
        gt_total = int(metrics["gt_total"].sum())
        pred_total = int(metrics["pred_total"].sum())
        gt_interphase = int(metrics["gt_interphase"].sum())
        gt_mitotic = int(metrics["gt_mitotic"].sum())
        pred_mitotic = int(metrics["pred_mitotic"].sum())

        gt_signature = (gt_total, gt_interphase, gt_mitotic)
        if reference_gt is None:
            reference_gt = gt_signature
        elif reference_gt != gt_signature:
            raise ValueError(f"Ground-truth totals differ for {system}")

        fn_interphase = int((false_negatives_frame["gt_class"] == 1).sum())
        fn_mitotic = int((false_negatives_frame["gt_class"] == 2).sum())
        fp_interphase = int((false_positives_frame["pred_class"] == 1).sum())
        fp_mitotic = int((false_positives_frame["pred_class"] == 2).sum())
        matched_gt_mitotic = int((matches["gt_class"] == 2).sum())
        gt_shape_classifier_mitotic = int((matches["gt_classifier_class"] == 2).sum())
        pred_shape_classifier_mitotic = int((matches["pred_class"] == 2).sum())
        interphase_to_mitotic = int(
            ((matches["gt_classifier_class"] == 1) & (matches["pred_class"] == 2)).sum()
        )
        mitotic_to_interphase = int(
            ((matches["gt_classifier_class"] == 2) & (matches["pred_class"] == 1)).sum()
        )
        morphology_class_changes = interphase_to_mitotic + mitotic_to_interphase
        gt_shape_classifier_errors = int(
            (matches["gt_classifier_class"] != matches["gt_class"]).sum()
        )
        pred_shape_classifier_errors = int((matches["pred_class"] != matches["gt_class"]).sum())
        baseline_classifier_net_mitotic = gt_shape_classifier_mitotic - matched_gt_mitotic
        predicted_shape_net_mitotic = (
            pred_shape_classifier_mitotic - gt_shape_classifier_mitotic
        )

        system_bridge = build_ratio_bridge(
            system=system,
            gt_mitotic=gt_mitotic,
            gt_total=gt_total,
            matched_gt_mitotic=matched_gt_mitotic,
            matched_total=tp,
            false_positive_mitotic=fp_mitotic,
            false_positives=fp,
            gt_shape_classifier_mitotic=gt_shape_classifier_mitotic,
            pred_shape_classifier_mitotic=pred_shape_classifier_mitotic,
        )
        bridge_rows.extend(system_bridge)
        decomposition_rows.extend(
            build_count_decomposition(
                system=system,
                gt_total=gt_total,
                false_negatives=fn,
                false_positives=fp,
                gt_mitotic=gt_mitotic,
                false_negative_mitotic=fn_mitotic,
                false_positive_mitotic=fp_mitotic,
                baseline_classifier_net_mitotic=baseline_classifier_net_mitotic,
                predicted_shape_net_mitotic=predicted_shape_net_mitotic,
            )
        )

        bridge_changes = [
            float(row["change_from_prior"])
            for row in system_bridge
            if row["change_from_prior"] is not None
        ]
        final_ratio = safe_ratio(pred_mitotic, pred_total)
        gt_ratio = safe_ratio(gt_mitotic, gt_total)
        classification_only_macro_f1 = float(classifier["classification_only"]["macro_f1"])
        end_to_end_macro_f1 = float(classifier["end_to_end"]["macro_f1"])
        summary_rows.append(
            {
                "system": system,
                "frames": len(metrics),
                "ground_truth_total": gt_total,
                "matched_instances": tp,
                "false_negatives": fn,
                "false_positives": fp,
                "predicted_total": pred_total,
                "total_count_error": pred_total - gt_total,
                "gt_interphase": gt_interphase,
                "gt_mitotic": gt_mitotic,
                "fn_interphase": fn_interphase,
                "fn_mitotic": fn_mitotic,
                "fn_rate_interphase": safe_ratio(fn_interphase, gt_interphase),
                "fn_rate_mitotic": safe_ratio(fn_mitotic, gt_mitotic),
                "fn_rate_mitotic_minus_interphase": (
                    safe_ratio(fn_mitotic, gt_mitotic)
                    - safe_ratio(fn_interphase, gt_interphase)
                ),
                "fp_pred_interphase": fp_interphase,
                "fp_pred_mitotic": fp_mitotic,
                "gt_mitotic_ratio": gt_ratio,
                "matched_gt_mitotic_ratio": system_bridge[1]["mitotic_ratio"],
                "predicted_mitotic_ratio": final_ratio,
                "mitotic_ratio_error": final_ratio - gt_ratio,
                "ratio_bridge_gross_absolute_movement": float(sum(abs(x) for x in bridge_changes)),
                "gt_shape_classifier_errors_on_matches": gt_shape_classifier_errors,
                "pred_shape_classifier_errors_on_matches": pred_shape_classifier_errors,
                "morphology_induced_class_changes": morphology_class_changes,
                "gt_shape_interphase_to_pred_shape_mitotic": interphase_to_mitotic,
                "gt_shape_mitotic_to_pred_shape_interphase": mitotic_to_interphase,
                "morphology_net_mitotic_count_change": predicted_shape_net_mitotic,
                "classification_only_accuracy": float(classifier["classification_only"]["accuracy"]),
                "classification_only_macro_f1": classification_only_macro_f1,
                "end_to_end_macro_f1": end_to_end_macro_f1,
                "matched_minus_end_to_end_macro_f1": (
                    classification_only_macro_f1 - end_to_end_macro_f1
                ),
            }
        )

        checks = {
            "matched_rows_equal_tp": len(matches) == tp,
            "false_negative_rows_equal_fn": len(false_negatives_frame) == fn,
            "false_positive_rows_equal_fp": len(false_positives_frame) == fp,
            "gt_total_equals_tp_plus_fn": gt_total == tp + fn,
            "pred_total_equals_tp_plus_fp": pred_total == tp + fp,
            "gt_classes_reconcile": gt_interphase + gt_mitotic == gt_total,
            "false_negative_classes_reconcile": fn_interphase + fn_mitotic == fn,
            "false_positive_classes_reconcile": fp_interphase + fp_mitotic == fp,
            "predicted_mitotic_reconciles": pred_shape_classifier_mitotic + fp_mitotic == pred_mitotic,
            "mitotic_decomposition_reconciles": (
                gt_mitotic
                - fn_mitotic
                + fp_mitotic
                + baseline_classifier_net_mitotic
                + predicted_shape_net_mitotic
                == pred_mitotic
            ),
            "ratio_bridge_final_matches_metrics": bool(
                np.isclose(system_bridge[-1]["mitotic_ratio"], final_ratio, rtol=0, atol=1e-12)
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"Error-chain reconciliation failed for {system}: {checks}")
        validations[system] = checks

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame["system"] = pd.Categorical(
        summary_frame["system"], categories=SYSTEM_ORDER, ordered=True
    )
    summary_frame = summary_frame.sort_values("system", kind="stable").reset_index(drop=True)
    summary_frame["system"] = summary_frame["system"].astype("object")
    bridge_frame = pd.DataFrame(bridge_rows)
    decomposition_frame = pd.DataFrame(decomposition_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "error_chain_summary.csv"
    bridge_path = output_dir / "mitotic_ratio_bridge.csv"
    decomposition_path = output_dir / "count_decomposition.csv"
    tex_path = output_dir / "error_chain_table.tex"
    summary_json_path = output_dir / "analysis_summary.json"
    summary_frame.to_csv(summary_csv_path, index=False, lineterminator="\n", float_format="%.10g")
    bridge_frame.to_csv(bridge_path, index=False, lineterminator="\n", float_format="%.10g")
    decomposition_frame.to_csv(
        decomposition_path, index=False, lineterminator="\n", float_format="%.10g"
    )
    tex_path.write_text(build_tex_table(summary_frame), encoding="utf-8", newline="\n")

    summary_records = {
        row["system"]: {key: value for key, value in row.items() if key != "system"}
        for row in summary_frame.to_dict(orient="records")
    }
    write_json(
        summary_json_path,
        {
            "schema_version": "1.0",
            "sequence": sequence,
            "analysis": "segmentation-to-analytics accounting chain",
            "definitions": {
                "false_negative": "annotated instance without an eligible one-to-one prediction match at IoU 0.50; this includes misses and severe merge, split, or boundary failures",
                "false_positive": "predicted instance without an eligible one-to-one annotation match at IoU 0.50; this includes extras and severe merge, split, or fragmentation failures",
                "morphology_induced_class_change": "different output from the same locked classifier when using predicted-mask rather than ground-truth-mask morphology for a matched instance",
                "ratio_bridge": "sequential accounting bridge; step contributions are path-dependent because the ratio is nonlinear",
            },
            "systems": summary_records,
            "validations": validations,
            "interpretation_limits": [
                "The decomposition is accounting evidence, not a biological causal model.",
                "Classification targets are ALFI annotation classes; interphase masks describe nuclei and mitotic masks describe whole-cell regions.",
                "Matched-only classification metrics are conditioned on successful detection and can hide selection bias.",
                "A small aggregate ratio error can result from cancellation among large opposing cell-level errors.",
            ],
        },
    )

    output_paths = (
        summary_csv_path,
        bridge_path,
        decomposition_path,
        tex_path,
        summary_json_path,
    )
    write_json(
        output_dir / "analysis_manifest.json",
        {
            "schema_version": "1.0",
            "status": "verified MI05 segmentation-to-analytics accounting analysis",
            "sequence": sequence,
            "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "packages": package_versions(),
            "inputs": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in sorted(input_paths)
            ],
            "outputs": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in output_paths
            ],
            "actions_performed": {
                "model_inference": False,
                "model_training": False,
                "threshold_tuning": False,
                "classifier_fitting": False,
            },
        },
    )


if __name__ == "__main__":
    main()
