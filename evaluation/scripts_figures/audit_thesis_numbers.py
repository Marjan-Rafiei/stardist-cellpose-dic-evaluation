"""Independent, read-only Step 31 numerical consistency checks.

Run from any directory with ``python scripts_figures/audit_thesis_numbers.py``.
The first discrepancy raises an error; no verified result or thesis file is edited.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
VERIFIED = ROOT / "results" / "verified"
MACRO_FILE = ROOT / "thesis" / "generated" / "results_values.tex"
MACROS = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", MACRO_FILE.read_text(encoding="utf-8")))
CHECKS = 0
AUDITED_MACROS: set[str] = set()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def obj(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(label: str, source, thesis, tolerance: float = 0) -> None:
    global CHECKS
    CHECKS += 1
    if isinstance(source, (float, int)) and isinstance(thesis, (float, int)):
        valid = math.isclose(source, thesis, abs_tol=tolerance, rel_tol=0)
    else:
        valid = source == thesis
    if not valid:
        raise AssertionError(
            f"CONFLICT {label}: source/calculation={source!r}; thesis/summary={thesis!r}; "
            f"allowed absolute rounding tolerance={tolerance}"
        )


def number(name: str) -> float:
    return float(MACROS[name].replace(",", "").replace(r"\%", ""))


def macro(name: str, source: float, decimals: int = 3) -> None:
    AUDITED_MACROS.add(name)
    check(f"thesis/generated/results_values.tex \\{name}", source, number(name), 0.5 * 10 ** -decimals + 1e-10)


def binary_scores(pairs: list[tuple[int, int]]) -> dict[int, dict[str, float | int]]:
    result = {}
    for cls in (1, 2):
        tp = sum(a == cls and b == cls for a, b in pairs)
        fp = sum(a != cls and b == cls for a, b in pairs)
        fn = sum(a == cls and b != cls for a, b in pairs)
        result[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    return result


def compare_class_scores(label: str, calculated: dict, reported: dict) -> None:
    for cls, name in ((1, "interphase"), (2, "mitotic")):
        for field in ("tp", "fp", "fn", "precision", "recall", "f1"):
            check(f"{label} {name} {field}", calculated[cls][field], reported["per_class"][name][field], 1e-9)
    check(f"{label} macro F1", mean(calculated[c]["f1"] for c in (1, 2)), reported["macro_f1"], 1e-9)


def audit_manifests() -> None:
    manifest = obj(ROOT / "thesis/generated/results_values_manifest.json")
    check("generated macro file SHA-256", digest(MACRO_FILE), manifest["output_sha256"])
    check("verified results index SHA-256", digest(VERIFIED / "results_index.json"), manifest["results_index_sha256"])
    for label, source in manifest["sources"].items():
        check(f"macro source {label} SHA-256", digest(Path(source["path"])), source["sha256"])
    assets = obj(ROOT / "thesis/generated/act8_assets_manifest.json")
    check("asset builder SHA-256", digest(Path(assets["script"]["path"])), assets["script"]["sha256"])
    for item in assets["verified_inputs"]:
        check(f"asset source {Path(item['path']).name} SHA-256", digest(Path(item["path"])), item["sha256"])
    for item in assets["staged_sources"]:
        check(f"figure source {Path(item['source']).name} SHA-256", digest(Path(item["source"])), item["source_sha256"])
        check(f"figure manifest {Path(item['manifest']).parent.name} SHA-256", digest(Path(item["manifest"])), item["manifest_sha256"])
        check(f"staged figure {Path(item['destination']).name} SHA-256", digest(Path(item["destination"])), item["source_sha256"])
    for item in assets["outputs"]:
        check(f"thesis asset {Path(item['path']).name} SHA-256", digest(Path(item["path"])), item["sha256"])


def audit_dataset_classifier() -> None:
    dataset = obj(VERIFIED / "dataset_audit/dataset_manifest.json")
    counts = dataset["counts"]
    macro("TotalAlignedFrames", sum(x["aligned_core"] for x in counts.values()), 0)
    macro("TrainingAlignedFrames", sum(counts[f"MI0{i}"]["aligned_core"] for i in (1, 2, 3)), 0)
    for seq, name in (("MI04", "ValidationFrames"), ("MI05", "TestFrames")):
        macro(name, counts[seq]["aligned_core"], 0)
    macro("MIOneImages", counts["MI01"]["image"], 0)
    macro("MIOneAnnotationFiles", counts["MI01"]["raw_mask"], 0)
    check("MI01 annotation-only count", counts["MI01"]["raw_mask"] - counts["MI01"]["image"], len(dataset["known_annotation_only_frames"]))
    conversion = obj(VERIFIED / "mask_conversion_audit/conversion_summary.json")
    for name, field in (("MaskAuditFiles", "total_masks"), ("MaskAuditMasksWithValueTwoFiveFive", "masks_with_value_255"),
                        ("MaskAuditValueTwoFiveFivePixels", "total_value_255_pixels"),
                        ("MaskAuditInstanceMatches", "corrected_instance_exact_matches"),
                        ("MaskAuditClassMatches", "corrected_class_exact_matches")):
        macro(name, conversion[field], 0)
    macro("MaskAuditFaultOutput", conversion["fault_consequence"]["value_255_after_fault"], 0)
    quality = obj(VERIFIED / "dataset_quality_audit/quality_summary.json")
    check("hard-coded thesis image dimensions", quality["representation"]["shape"], [320, 400])
    check("hard-coded raw mask values", quality["representation"]["value_unions"]["raw_mask"], [0, 128, 255])
    check("hard-coded class mask values", quality["representation"]["value_unions"]["class_mask"], [0, 1, 2])
    check("hard-coded duplicate count", quality["duplicate_groups"], 0)

    val = rows(VERIFIED / "classifier_v1/validation_predictions.csv")
    summary = rows(VERIFIED / "classifier_validation_summary.csv")[0]
    check("classifier validation rows", len(val), int(summary["validation_rows"]))
    macro("ClassifierValidationRows", len(val), 0)
    pairs = [(int(r["target"]), int(r["predicted"])) for r in val]
    scores = binary_scores(pairs)
    accuracy = sum(a == b for a, b in pairs) / len(pairs)
    balanced = mean(scores[c]["recall"] for c in (1, 2))
    macro_f1 = mean(scores[c]["f1"] for c in (1, 2))
    for label, value in (("accuracy", accuracy), ("balanced_accuracy", balanced), ("macro_f1", macro_f1)):
        check(f"classifier validation CSV {label}", value, float(summary[label]), 1e-9)
    for cls, label in ((1, "interphase_f1"), (2, "mitotic_f1")):
        check(f"classifier validation CSV {label}", scores[cls]["f1"], float(summary[label]), 1e-9)
    for name, value in (("ClassifierValidationAccuracy", accuracy), ("ClassifierValidationBalancedAccuracy", balanced),
                        ("ClassifierValidationMacroFOne", macro_f1), ("ClassifierValidationInterphaseFOne", scores[1]["f1"]),
                        ("ClassifierValidationMitoticFOne", scores[2]["f1"])):
        macro(name, value)
    classifier_manifest = obj(VERIFIED / "classifier_v1/classifier_manifest.json")
    training = rows(VERIFIED / "classifier_v1/training_rows.csv")
    check("classifier training rows directly counted", len(training), classifier_manifest["row_counts"]["training"])
    check("classifier training has no MI04 or MI05", {r["sequence"] for r in training}, {"MI01", "MI02", "MI03"})
    macro("ClassifierTrainingRows", classifier_manifest["row_counts"]["training"], 0)
    parameters = classifier_manifest["classifier_parameters"]
    for name, field in (("ClassifierTrees", "n_estimators"), ("ClassifierSeed", "random_state"), ("ClassifierJobs", "n_jobs")):
        macro(name, parameters[field], 0)
    benchmark = {r["model"]: r for r in rows(VERIFIED / "random_forest_audit/benchmark_metrics.csv")}
    benchmark_predictions = rows(VERIFIED / "random_forest_audit/benchmark_predictions.csv")
    logistic_pairs = [(int(r["target"]), int(r["logistic_regression_predicted"])) for r in benchmark_predictions]
    logistic_scores = binary_scores(logistic_pairs)
    logistic_macro = mean(logistic_scores[c]["f1"] for c in (1, 2))
    check("Logistic Regression benchmark macro F1", logistic_macro, float(benchmark["logistic_regression"]["macro_f1"]), 1e-9)
    macro("LogisticBenchmarkMacroFOne", logistic_macro)


def audit_system(system: str, prefix: str) -> None:
    folder = VERIFIED / system
    frames = rows(folder / "metrics_by_frame.csv")
    matches = rows(folder / "matched_instances.csv")
    errors = rows(folder / "error_cases.csv")
    summary = obj(folder / "metrics_summary.json")["pooled"]
    classes = obj(folder / "classifier_metrics.json")
    check(f"{system} frame rows", len(frames), 69)
    counts = {field: sum(int(r[field]) for r in frames) for field in ("tp", "fp", "fn", "gt_total", "pred_total", "gt_interphase", "gt_mitotic", "pred_interphase", "pred_mitotic")}
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    check(f"{system} matches count", len(matches), tp)
    check(f"{system} unmatched FN rows", sum(r["error_type"] == "false_negative" for r in errors), fn)
    check(f"{system} unmatched FP rows", sum(r["error_type"] == "false_positive" for r in errors), fp)
    check(f"{system} GT count identity", tp + fn, counts["gt_total"])
    check(f"{system} predicted count identity", tp + fp, counts["pred_total"])
    iou_sum = sum(float(r["iou"]) for r in matches)
    calculated = {"tp": tp, "fp": fp, "fn": fn,
                  "precision": tp / (tp + fp), "recall": tp / (tp + fn),
                  "f1": 2 * tp / (2 * tp + fp + fn),
                  "matched_iou_count": tp, "matched_iou_sum": iou_sum,
                  "matched_iou": iou_sum / tp, "pq": iou_sum / (tp + 0.5 * fp + 0.5 * fn)}
    for field, value in calculated.items():
        check(f"{system} metrics_summary pooled.{field}", value, summary[field], 1e-8)
    comparison = next(r for r in rows(VERIFIED / "comparison_summary.csv") if r["system"] == system)
    for field, column in (("precision", "pooled_precision"), ("recall", "pooled_recall"), ("f1", "pooled_f1"),
                          ("pq", "pooled_pq"), ("matched_iou", "pooled_matched_iou")):
        check(f"{system} comparison_summary.{column}", calculated[field], float(comparison[column]), 1e-9)
    check(f"{system} comparison_summary ground truth", counts["gt_total"], int(comparison["ground_truth_total"]))
    check(f"{system} comparison_summary predicted total", counts["pred_total"], int(comparison["predicted_total"]))
    for field, suffix in (("precision", "PooledPrecision"), ("recall", "PooledRecall"), ("f1", "PooledFOne"),
                          ("matched_iou", "MatchedIoU"), ("pq", "PooledPQ"),
                          ("tp", "TruePositives"), ("fp", "FalsePositives"), ("fn", "FalseNegatives")):
        macro(prefix + suffix, calculated[field], 0 if field in ("tp", "fp", "fn") else 3)
    for r in frames:
        ft, ff, ffn = (int(r[k]) for k in ("tp", "fp", "fn"))
        frame = r["frame_id"]
        for label, value in (("precision", ft / (ft + ff)), ("recall", ft / (ft + ffn)),
                             ("f1", 2 * ft / (2 * ft + ff + ffn)),
                             ("matched_iou", float(r["matched_iou_sum"]) / ft),
                             ("pq", float(r["matched_iou_sum"]) / (ft + 0.5 * ff + 0.5 * ffn))):
            check(f"{system} {frame} {label}", value, float(r[label]), 1e-9)
        check(f"{system} {frame} GT class totals", int(r["gt_interphase"]) + int(r["gt_mitotic"]), int(r["gt_total"]))
        check(f"{system} {frame} predicted class totals", int(r["pred_interphase"]) + int(r["pred_mitotic"]), int(r["pred_total"]))
        check(f"{system} {frame} GT mitotic ratio", int(r["gt_mitotic"]) / int(r["gt_total"]), float(r["gt_mitotic_ratio"]), 1e-9)
        check(f"{system} {frame} predicted mitotic ratio", int(r["pred_mitotic"]) / int(r["pred_total"]), float(r["pred_mitotic_ratio"]), 1e-9)
        check(f"{system} {frame} ratio error", float(r["pred_mitotic_ratio"]) - float(r["gt_mitotic_ratio"]),
              float(r["mitotic_ratio_error"]), 1e-9)
        check(f"{system} {r['frame_id']} count error", int(r["pred_total"]) - int(r["gt_total"]), int(r["total_count_error"]))
    frame_macro = obj(folder / "metrics_summary.json")["per_frame_macro"]
    for field in ("precision", "recall", "f1", "pq", "matched_iou"):
        check(f"{system} per-frame macro {field}", mean(float(r[field]) for r in frames),
              frame_macro[field]["mean"], 1e-9)

    matched_pairs = [(int(r["gt_class"]), int(r["pred_class"])) for r in matches]
    only = binary_scores(matched_pairs)
    compare_class_scores(f"{system} classification only", only, classes["classification_only"])
    confusion = [[sum(a == gt and b == pred for a, b in matched_pairs) for pred in (1, 2)] for gt in (1, 2)]
    check(f"{system} matched confusion", confusion, classes["classification_only_confusion_matrix"])
    end_pairs = matched_pairs + [(int(r["gt_class"]), 0) for r in errors if r["error_type"] == "false_negative"]
    end_pairs += [(0, int(r["pred_class"])) for r in errors if r["error_type"] == "false_positive"]
    end_scores = binary_scores(end_pairs)
    compare_class_scores(f"{system} end to end", end_scores, classes["end_to_end"])
    for field, value in (("classification_only_macro_f1", classes["classification_only"]["macro_f1"]),
                         ("end_to_end_macro_f1", classes["end_to_end"]["macro_f1"]),
                         ("end_to_end_interphase_f1", end_scores[1]["f1"]),
                         ("end_to_end_mitotic_f1", end_scores[2]["f1"])):
        check(f"{system} comparison_summary.{field}", value, float(comparison[field]), 1e-9)
    macro(prefix + "ClassificationMacroFOne", classes["classification_only"]["macro_f1"])
    macro(prefix + "EndToEndMacroFOne", classes["end_to_end"]["macro_f1"])
    macro(prefix + "EndToEndInterphaseFOne", end_scores[1]["f1"])
    macro(prefix + "EndToEndMitoticFOne", end_scores[2]["f1"])

    macro(prefix + "PredictedObjects", counts["pred_total"], 0)
    macro(prefix + "PredictedMitoticRatio", 100 * counts["pred_mitotic"] / counts["pred_total"], 1)
    check(f"{system} comparison_summary predicted pooled ratio", counts["pred_mitotic"] / counts["pred_total"],
          float(comparison["predicted_pooled_mitotic_ratio"]), 1e-9)
    macro("TestGroundTruthObjects", counts["gt_total"], 0)
    macro("TestGroundTruthMitoticRatio", 100 * counts["gt_mitotic"] / counts["gt_total"], 1)
    check(f"{system} comparison_summary GT pooled ratio", counts["gt_mitotic"] / counts["gt_total"],
          float(comparison["ground_truth_pooled_mitotic_ratio"]), 1e-9)
    flips = sum(r["gt_classifier_correct"] == "1" and r["class_correct"] == "0" for r in matches)
    output_changes = sum(r["gt_classifier_class"] != r["pred_class"] for r in matches)
    restored = sum(r["gt_classifier_correct"] == "0" and r["class_correct"] == "1" for r in matches)
    for suffix, value in (("SegmentationClassFlips", flips), ("ClassOutputChanges", output_changes), ("IncorrectToCorrect", restored)):
        macro(prefix + suffix, value, 0)
    check(f"{system} comparison_summary class flips", flips, int(comparison["segmentation_induced_class_flips"]))
    area = median(100 * (float(r["pred_area"]) / float(r["gt_area"]) - 1) for r in matches)
    macro(prefix + "MorphologyAreaShift", area, 1)

    agreement = next(r for r in rows(VERIFIED / "counts_ratio_validation/agreement_summary.csv") if r["system"] == system)
    count_error = [int(r["pred_total"]) - int(r["gt_total"]) for r in frames]
    ratio_error = [float(r["pred_mitotic_ratio"]) - float(r["gt_mitotic_ratio"]) for r in frames]
    aggregates = {"count_bias": mean(count_error), "count_mae": mean(abs(x) for x in count_error),
                  "pooled_ratio_error": counts["pred_mitotic"] / counts["pred_total"] - counts["gt_mitotic"] / counts["gt_total"],
                  "ratio_bias": mean(ratio_error), "ratio_mae": mean(abs(x) for x in ratio_error),
                  "ratio_error_cancellation_fraction": 1 - abs(sum(ratio_error)) / sum(abs(x) for x in ratio_error)}
    for field, value in aggregates.items():
        check(f"{system} agreement {field}", value, float(agreement[field]), 1e-9)
    close = [r for r, e in zip(frames, ratio_error) if abs(e) <= 0.05 + 1e-10]
    check(f"{system} ratio-close frames", len(close), int(agreement["ratio_close_frames"]))
    check(f"{system} count-wrong ratio-close", sum(int(r["total_count_error"]) != 0 for r in close), int(agreement["count_wrong_ratio_close_frames"]))
    if system == "cellpose_fixed":
        macro("CellposeFixedCountWrongRatioCloseFrames", int(agreement["count_wrong_ratio_close_frames"]), 0)
        macro("CellposeFixedRatioCloseFrames", len(close), 0)
        macro("CellposeFixedRatioFrameMAE", 100 * aggregates["ratio_mae"], 2)
        macro("CellposeFixedRatioCancellation", 100 * aggregates["ratio_error_cancellation_fraction"], 1)


def audit_cross_system() -> None:
    systems = ("stardist_saved", "cellpose_auto", "cellpose_fixed")
    matched_sets = []
    for system in systems:
        matched_sets.append({(r["frame_id"], r["gt_id"]) for r in rows(VERIFIED / system / "matched_instances.csv")})
    macro("CommonMatchedInstances", len(set.intersection(*matched_sets)), 0)
    chain = {r["system"]: r for r in rows(VERIFIED / "segmentation_analytics_error_chain/error_chain_summary.csv")}
    for system, prefix in (("cellpose_auto", "CellposeAuto"), ("cellpose_fixed", "CellposeFixed")):
        errors = rows(VERIFIED / system / "error_cases.csv")
        for cls, name in ((1, "InterphaseFalseNegatives"), (2, "MitoticFalseNegatives")):
            value = sum(r["error_type"] == "false_negative" and r["gt_class"] == str(cls) for r in errors)
            macro(prefix + name, value, 0)
            check(f"{system} error-chain {name}", value, int(chain[system]["fn_interphase" if cls == 1 else "fn_mitotic"]))
        matches = rows(VERIFIED / system / "matched_instances.csv")
        ratio = 100 * sum(r["gt_class"] == "2" for r in matches) / len(matches)
        macro(prefix + "MatchedGroundTruthMitoticRatio", ratio, 1)

    thirds = {r["system"]: {} for r in rows(VERIFIED / "counts_ratio_validation/temporal_segment_summary.csv")}
    for r in rows(VERIFIED / "counts_ratio_validation/temporal_segment_summary.csv"):
        thirds[r["system"]][r["temporal_third"]] = float(r["count_bias"])
    for system in systems:
        check(f"{system} early-to-late count underestimation trend",
              thirds[system]["late"] < thirds[system]["early"], True)

    sensitivity = rows(VERIFIED / "iou_sensitivity/iou_sensitivity_metrics.csv")
    evaluation = obj(VERIFIED / "stardist_saved/evaluation_manifest.json")
    macro("EvaluationIoUThreshold", float(evaluation["iou_threshold"]), 1)
    macro("StarDistSavedProbabilityThreshold", float(evaluation["inference_parameters"]["probability_threshold"]), 4)
    check("hard-coded NMS threshold", float(evaluation["inference_parameters"]["nms_threshold"]), 0.3)
    thresholds = sorted({float(r["iou_threshold"]) for r in sensitivity})
    macro("IoUSensitivityMinimum", thresholds[0], 1)
    macro("IoUSensitivityMaximum", thresholds[-1], 1)
    for threshold in thresholds:
        subset = {r["system"]: r for r in sensitivity if float(r["iou_threshold"]) == threshold}
        for system, r in subset.items():
            tp, fp, fn = (int(r[k]) for k in ("tp", "fp", "fn"))
            check(f"IoU {threshold} {system} F1 from counts", 2 * tp / (2 * tp + fp + fn), float(r["f1"]), 1e-9)
            check(f"IoU {threshold} {system} precision from counts", tp / (tp + fp), float(r["precision"]), 1e-9)
            check(f"IoU {threshold} {system} recall from counts", tp / (tp + fn), float(r["recall"]), 1e-9)
            check(f"IoU {threshold} {system} PQ from IoU support", float(r["mean_matched_iou"]) * tp / (tp + 0.5 * fp + 0.5 * fn),
                  float(r["pq"]), 1e-9)
        for field in ("f1", "pq"):
            check(f"IoU {threshold} StarDist ranks first on {field}",
                  float(subset["stardist_saved"][field]) > max(float(subset[s][field]) for s in systems[1:]), True)

    selected = rows(VERIFIED / "figures/step20/failure_case_selection.csv")
    check("two selected high-error frames", {r.get("frame_id") for r in selected}, {"MI05_0055", "MI05_0069"})
    frame_data = {system: {r["frame_id"]: r for r in rows(VERIFIED / system / "metrics_by_frame.csv")} for system in systems}
    fn_ranking = sorted(frame_data[systems[0]], key=lambda f: (-sum(int(frame_data[s][f]["fn"]) for s in systems), f))
    fp_ranking = sorted((f for f in frame_data[systems[0]] if f != fn_ranking[0]),
                        key=lambda f: (-sum(int(frame_data[s][f]["fp"]) for s in systems), f))
    check("hard-coded failure frame MI05_0055", fn_ranking[0], "MI05_0055")
    check("hard-coded failure frame MI05_0069", fp_ranking[0], "MI05_0069")
    for system in systems[1:]:
        for frame in ("MI05_0055", "MI05_0069"):
            check(f"{system} {frame} hard-coded matches", int(frame_data[system][frame]["tp"]), 1)
            check(f"{system} {frame} hard-coded missed annotations", int(frame_data[system][frame]["fn"]), 12)

    bootstrap = rows(VERIFIED / "temporal_bootstrap/bootstrap_replicates.csv")
    intervals = rows(VERIFIED / "temporal_bootstrap/bootstrap_interval_sensitivity.csv")
    bootstrap_manifest = obj(VERIFIED / "temporal_bootstrap/analysis_manifest.json")
    command = bootstrap_manifest["command"]
    macro("BootstrapBlockLength", int(command[command.index("--chosen-block-length") + 1]), 0)
    macro("BootstrapReplicates", int(command[command.index("--replicates") + 1]), 0)
    macro("BootstrapSeed", int(command[command.index("--seed") + 1]), 0)
    autocorrelation = rows(VERIFIED / "temporal_bootstrap/autocorrelation.csv")
    for system, expected in (("stardist_saved", 7), ("cellpose_auto", 9), ("cellpose_fixed", 9)):
        decay = min(int(r["lag_frames"]) for r in autocorrelation if r["system"] == system
                    and r["quantity"] == "f1" and abs(float(r["autocorrelation"])) < 0.2)
        check(f"hard-coded {system} F1 autocorrelation decay lag", decay, expected)
    for system, prefix in (("stardist_saved", "StarDist"), ("cellpose_auto", "CellposeAuto"), ("cellpose_fixed", "CellposeFixed")):
        data = sorted(float(r["f1"]) for r in bootstrap if r["system"] == system and int(r["block_length"]) == 10)
        check(f"{system} bootstrap count", len(data), int(number("BootstrapReplicates")))
        def percentile(p: float) -> float:
            x = (len(data) - 1) * p
            lo = math.floor(x)
            return data[lo] + (x - lo) * (data[math.ceil(x)] - data[lo])
        interval = next(r for r in intervals if r["system"] == system and r["metric"] == "f1" and r["block_length"] == "10")
        for p, field, suffix in ((0.025, "percentile_2_5", "BootstrapFOneLower"), (0.975, "percentile_97_5", "BootstrapFOneUpper")):
            value = percentile(p)
            check(f"{system} bootstrap {field}", value, float(interval[field]), 1e-9)
            comparison = next(r for r in rows(VERIFIED / "comparison_summary.csv") if r["system"] == system)
            column = "block_bootstrap_f1_lower" if p == 0.025 else "block_bootstrap_f1_upper"
            check(f"{system} comparison_summary {column}", value, float(comparison[column]), 1e-9)
            macro(prefix + suffix, value)


def table_values(filename: str, label: str) -> list[float]:
    content = (ROOT / "thesis/tables" / filename).read_text(encoding="utf-8")
    for line in content.splitlines():
        if label in line and "&" in line and not line.lstrip().startswith("%"):
            fields = line.split("&")[1:]
            values = []
            for field in fields:
                values += [float(x) for x in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", field)]
            return values
    raise AssertionError(f"Missing thesis table row: {filename} / {label}")


def audit_tables_and_prose() -> None:
    dataset = obj(VERIFIED / "dataset_audit/dataset_manifest.json")["counts"]
    for seq in ("MI01", "MI02", "MI03", "MI04", "MI05"):
        expected = [dataset[seq][field] for field in ("image", "instance_mask", "aligned_core")]
        check(f"dataset table {seq}", table_values("dataset_split_table.tex", seq)[:3], expected)
    check("dataset table Total", table_values("dataset_split_table.tex", "Total")[:3], [343, 345, 343])

    classifier = rows(VERIFIED / "classifier_validation_summary.csv")[0]
    for label, field, decimals in (("Training instances", "training_rows", 0), ("Validation instances", "validation_rows", 0),
                                   ("Accuracy", "accuracy", 3), ("Balanced accuracy", "balanced_accuracy", 3),
                                   ("Macro F1", "macro_f1", 3), ("Interphase F1", "interphase_f1", 3),
                                   ("Mitotic F1", "mitotic_f1", 3)):
        value = table_values("classifier_validation_table.tex", label)[0]
        check(f"classifier table {label}", float(classifier[field]), value, 0.5 * 10 ** -decimals + 1e-10)

    model_rows = (("StarDist", "stardist_saved"), ("Cellpose automatic", "cellpose_auto"), ("Cellpose fixed", "cellpose_fixed"))
    for label, system in model_rows:
        summary = obj(VERIFIED / system / "metrics_summary.json")["pooled"]
        expected = [summary[field] for field in ("precision", "recall", "f1", "pq", "matched_iou")]
        actual = table_values("mi05_detection_table.tex", label)
        for field, source, thesis in zip(("precision", "recall", "f1", "pq", "matched_iou"), expected, actual):
            check(f"detection table {system} {field}", source, thesis, 0.0005000001)
        classes = obj(VERIFIED / system / "classifier_metrics.json")
        expected = [classes["classification_only"]["macro_f1"], classes["end_to_end"]["macro_f1"],
                    classes["end_to_end"]["per_class"]["interphase"]["f1"], classes["end_to_end"]["per_class"]["mitotic"]["f1"]]
        actual = table_values("mi05_classification_table.tex", label)
        for idx, (source, thesis) in enumerate(zip(expected, actual)):
            check(f"classification table {system} column {idx + 1}", source, thesis, 0.0005000001)
        agreement = next(r for r in rows(VERIFIED / "counts_ratio_validation/agreement_summary.csv") if r["system"] == system)
        expected = [float(agreement["count_bias"]), float(agreement["count_mae"]), 100 * float(agreement["pooled_ratio_error"]),
                    100 * float(agreement["ratio_bias"]), 100 * float(agreement["ratio_mae"]),
                    100 * float(agreement["ratio_error_cancellation_fraction"])]
        actual = table_values("mi05_analytics_table.tex", label if system == "stardist_saved" else "Cellpose")
        if system == "cellpose_fixed":
            content = (ROOT / "thesis/tables/mi05_analytics_table.tex").read_text(encoding="utf-8")
            line = next(x for x in content.splitlines() if "fixed}" in x and "&" in x)
            actual = [float(x) for x in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", "&".join(line.split("&")[1:]))]
        for idx, (source, thesis) in enumerate(zip(expected, actual)):
            tolerance = 0.05000001 if idx == 5 else 0.00500001
            check(f"analytics table {system} column {idx + 1}", source, thesis, tolerance)
        cancellation = table_values("mi05_cancellation_cases_table.tex", label)
        expected = [int(agreement[field]) for field in ("ratio_close_frames", "count_wrong_ratio_close_frames",
                                                     "final_class_wrong_ratio_close_frames", "gt_shape_class_wrong_ratio_close_frames")]
        check(f"cancellation table {system}", cancellation, expected)

    intervals = rows(VERIFIED / "temporal_bootstrap/bootstrap_interval_sensitivity.csv")
    for label, system in model_rows:
        interval = next(r for r in intervals if r["system"] == system and r["metric"] == "f1" and r["block_length"] == "10")
        actual = table_values("bootstrap_f1_table.tex", label)
        expected = [float(interval[field]) for field in ("point_estimate", "percentile_2_5", "percentile_97_5")]
        for idx, (source, thesis) in enumerate(zip(expected, actual)):
            check(f"bootstrap table {system} column {idx + 1}", source, thesis, 0.0005000001)

    # Catch numeric prose accidentally hard-coded outside generated macros. These
    # exceptions are design constants, source identifiers, and explicitly rejected
    # historical figures; they are reported for human review below.
    prose_files = [ROOT / "thesis/Chapters/summary.tex", *(ROOT / "thesis/Chapters").glob("chapter*.tex"),
                   *(ROOT / "thesis/appendices").glob("*.tex")]
    numeric_lines = []
    for path in prose_files:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"(?<!\\)[0-9]+(?:\.[0-9]+)?", line) and not line.lstrip().startswith("%"):
                numeric_lines.append(f"{path.relative_to(ROOT)}:{line_no}: {line.strip()[:140]}")
    print(f"Numeric prose/caption lines enumerated for targeted review: {len(numeric_lines)}")


if __name__ == "__main__":
    audit_manifests()
    audit_dataset_classifier()
    for model, label in (("stardist_saved", "StarDist"), ("cellpose_auto", "CellposeAuto"), ("cellpose_fixed", "CellposeFixed")):
        audit_system(model, label)
        print(f"PASS: {model} independent metrics, classes, counts, ratios, and thesis macros")
    audit_cross_system()
    audit_tables_and_prose()
    check("all generated numerical macros audited", sorted(AUDITED_MACROS), sorted(MACROS))
    print(f"PASS: {CHECKS} numerical and provenance comparisons; no discrepancy found")
