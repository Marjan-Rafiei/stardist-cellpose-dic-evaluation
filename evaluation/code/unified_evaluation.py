"""Shared instance, class, and analytics evaluation for StarDist and Cellpose.

The command consumes instance-label masks from one prediction source, ALFI
ground-truth instance/class masks, and one locked morphology classifier. It
does not train models or tune thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops


FEATURE_NAMES = (
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "major_axis_length",
    "minor_axis_length",
    "aspect_ratio",
)
CLASS_NAMES = {1: "interphase", 2: "mitotic"}
SPECIFICATION_VERSION = "2.0"


@dataclass(frozen=True)
class Match:
    gt_id: int
    pred_id: int
    iou: float


@dataclass(frozen=True)
class DetectionCounts:
    tp: int
    fp: int
    fn: int
    matched_iou_sum: float
    matched_iou_count: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mask(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask at {path}, got shape {array.shape}")
    return array.astype(np.int64, copy=False)


def positive_ids(mask: np.ndarray) -> np.ndarray:
    ids = np.unique(mask)
    return ids[ids > 0].astype(np.int64, copy=False)


def pairwise_iou(
    gt_mask: np.ndarray, pred_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(f"Mask shapes differ: {gt_mask.shape} vs {pred_mask.shape}")

    gt_ids = positive_ids(gt_mask)
    pred_ids = positive_ids(pred_mask)
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    if matrix.size == 0:
        return gt_ids, pred_ids, matrix

    gt_index = {int(label): index for index, label in enumerate(gt_ids)}
    pred_index = {int(label): index for index, label in enumerate(pred_ids)}
    intersections = np.zeros_like(matrix, dtype=np.int64)

    overlap = (gt_mask > 0) & (pred_mask > 0)
    pairs = np.column_stack((gt_mask[overlap], pred_mask[overlap]))
    if pairs.size:
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for (gt_id, pred_id), count in zip(unique_pairs, counts, strict=True):
            intersections[gt_index[int(gt_id)], pred_index[int(pred_id)]] = int(count)

    gt_areas = np.array([(gt_mask == label).sum() for label in gt_ids], dtype=np.int64)
    pred_areas = np.array([(pred_mask == label).sum() for label in pred_ids], dtype=np.int64)
    unions = gt_areas[:, None] + pred_areas[None, :] - intersections
    np.divide(intersections, unions, out=matrix, where=unions > 0)
    return gt_ids, pred_ids, matrix


def match_iou_matrix(
    gt_ids: Sequence[int],
    pred_ids: Sequence[int],
    iou_matrix: np.ndarray,
    threshold: float,
) -> list[Match]:
    if not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must be in [0, 1]")
    expected_shape = (len(gt_ids), len(pred_ids))
    if iou_matrix.shape != expected_shape:
        raise ValueError(f"Expected IoU matrix {expected_shape}, got {iou_matrix.shape}")
    if iou_matrix.size == 0:
        return []

    # Lexicographic objective: maximize the number of threshold-eligible
    # pairs first, then maximize their total IoU.  The bonus is larger than
    # the maximum possible difference in summed IoU between assignments.
    max_pairs = min(len(gt_ids), len(pred_ids))
    eligible = iou_matrix >= threshold
    bonus = float(max_pairs + 1)
    score = np.where(eligible, bonus + iou_matrix, 0.0)
    rows, columns = linear_sum_assignment(-score)
    accepted = [
        Match(int(gt_ids[row]), int(pred_ids[column]), float(iou_matrix[row, column]))
        for row, column in zip(rows, columns, strict=True)
        if iou_matrix[row, column] >= threshold
    ]
    return sorted(accepted, key=lambda item: (item.gt_id, item.pred_id))


def match_instances(
    gt_mask: np.ndarray, pred_mask: np.ndarray, threshold: float = 0.5
) -> tuple[list[Match], np.ndarray, np.ndarray]:
    gt_ids, pred_ids, matrix = pairwise_iou(gt_mask, pred_mask)
    return match_iou_matrix(gt_ids, pred_ids, matrix, threshold), gt_ids, pred_ids


def detection_counts(
    gt_mask: np.ndarray, pred_mask: np.ndarray, threshold: float = 0.5
) -> tuple[DetectionCounts, list[Match]]:
    matches, gt_ids, pred_ids = match_instances(gt_mask, pred_mask, threshold)
    count = DetectionCounts(
        tp=len(matches),
        fp=len(pred_ids) - len(matches),
        fn=len(gt_ids) - len(matches),
        matched_iou_sum=float(sum(match.iou for match in matches)),
        matched_iou_count=len(matches),
    )
    return count, matches


def _ratio(numerator: float, denominator: float, both_empty: bool) -> float:
    if denominator:
        return float(numerator / denominator)
    return 1.0 if both_empty else 0.0


def detection_metrics(counts: DetectionCounts) -> dict[str, float | int | None]:
    both_empty = counts.tp == counts.fp == counts.fn == 0
    precision = _ratio(counts.tp, counts.tp + counts.fp, both_empty)
    recall = _ratio(counts.tp, counts.tp + counts.fn, both_empty)
    f1 = _ratio(2 * counts.tp, 2 * counts.tp + counts.fp + counts.fn, both_empty)
    pq_denominator = counts.tp + 0.5 * counts.fp + 0.5 * counts.fn
    pq = _ratio(counts.matched_iou_sum, pq_denominator, both_empty)
    matched_iou = (
        counts.matched_iou_sum / counts.matched_iou_count
        if counts.matched_iou_count
        else None
    )
    return {
        **asdict(counts),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_iou": matched_iou,
        "pq": pq,
    }


def pooled_metrics(frame_counts: Iterable[DetectionCounts]) -> dict[str, Any]:
    rows = list(frame_counts)
    totals = DetectionCounts(
        tp=sum(row.tp for row in rows),
        fp=sum(row.fp for row in rows),
        fn=sum(row.fn for row in rows),
        matched_iou_sum=sum(row.matched_iou_sum for row in rows),
        matched_iou_count=sum(row.matched_iou_count for row in rows),
    )
    return detection_metrics(totals)


def macro_summary(frame_metrics: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_frames": int(len(frame_metrics))}
    for name in ("precision", "recall", "f1", "matched_iou", "pq"):
        values = pd.to_numeric(frame_metrics[name], errors="coerce").dropna()
        summary[name] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None,
            "median": float(values.median()) if len(values) else None,
            "q1": float(values.quantile(0.25)) if len(values) else None,
            "q3": float(values.quantile(0.75)) if len(values) else None,
            "support": int(len(values)),
        }
    return summary


def class_for_instance(instance_mask: np.ndarray, class_mask: np.ndarray, instance_id: int) -> int:
    values = class_mask[instance_mask == instance_id]
    values = values[np.isin(values, list(CLASS_NAMES))]
    if values.size == 0:
        raise ValueError(f"Instance {instance_id} has no valid class pixels")
    counts = np.bincount(values.astype(np.int64), minlength=max(CLASS_NAMES) + 1)
    return int(np.argmax(counts[1:]) + 1)


def extract_features(instance_mask: np.ndarray) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    for region in regionprops(instance_mask.astype(np.int64, copy=False)):
        perimeter = float(region.perimeter)
        area = float(region.area)
        major_axis = float(region.major_axis_length)
        minor_axis = float(region.minor_axis_length)
        aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0.0
        records.append(
            {
                "instance_id": int(region.label),
                "area": area,
                "perimeter": perimeter,
                "eccentricity": float(region.eccentricity),
                "solidity": float(region.solidity),
                "extent": float(region.extent),
                "major_axis_length": major_axis,
                "minor_axis_length": minor_axis,
                "aspect_ratio": aspect_ratio,
            }
        )
    return pd.DataFrame.from_records(records, columns=("instance_id", *FEATURE_NAMES))


def predict_instance_classes(instance_mask: np.ndarray, classifier: Any) -> dict[int, int]:
    features = extract_features(instance_mask)
    if features.empty:
        return {}
    predictions = classifier.predict(features.loc[:, FEATURE_NAMES])
    return {
        int(instance_id): int(prediction)
        for instance_id, prediction in zip(features["instance_id"], predictions, strict=True)
    }


def classification_counts(
    matches: Sequence[Match],
    gt_mask: np.ndarray,
    gt_class_mask: np.ndarray,
    pred_classes: Mapping[int, int],
) -> tuple[np.ndarray, dict[int, dict[str, int]]]:
    confusion = np.zeros((2, 2), dtype=np.int64)
    for match in matches:
        true_class = class_for_instance(gt_mask, gt_class_mask, match.gt_id)
        predicted_class = int(pred_classes[match.pred_id])
        confusion[true_class - 1, predicted_class - 1] += 1

    metrics: dict[int, dict[str, int]] = {}
    for class_id in CLASS_NAMES:
        index = class_id - 1
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        metrics[class_id] = {"tp": tp, "fp": fp, "fn": fn}
    return confusion, metrics


def end_to_end_class_counts(
    matches: Sequence[Match],
    gt_ids: Sequence[int],
    pred_ids: Sequence[int],
    gt_mask: np.ndarray,
    gt_class_mask: np.ndarray,
    pred_classes: Mapping[int, int],
) -> dict[int, dict[str, int]]:
    counts = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in CLASS_NAMES}
    matched_gt = {match.gt_id for match in matches}
    matched_pred = {match.pred_id for match in matches}

    for match in matches:
        true_class = class_for_instance(gt_mask, gt_class_mask, match.gt_id)
        predicted_class = int(pred_classes[match.pred_id])
        if true_class == predicted_class:
            counts[true_class]["tp"] += 1
        else:
            counts[true_class]["fn"] += 1
            counts[predicted_class]["fp"] += 1

    for gt_id in set(map(int, gt_ids)) - matched_gt:
        counts[class_for_instance(gt_mask, gt_class_mask, gt_id)]["fn"] += 1
    for pred_id in set(map(int, pred_ids)) - matched_pred:
        counts[int(pred_classes[pred_id])]["fp"] += 1
    return counts


def class_metric_row(counts: Mapping[str, int]) -> dict[str, float | int]:
    both_empty = counts["tp"] == counts["fp"] == counts["fn"] == 0
    return {
        **counts,
        "precision": _ratio(counts["tp"], counts["tp"] + counts["fp"], both_empty),
        "recall": _ratio(counts["tp"], counts["tp"] + counts["fn"], both_empty),
        "f1": _ratio(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"], both_empty),
    }


def confusion_metrics(confusion: np.ndarray) -> dict[str, Any]:
    if confusion.shape != (2, 2):
        raise ValueError(f"Expected a 2 x 2 confusion matrix, got {confusion.shape}")
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    recalls: list[float] = []
    per_class: dict[str, dict[str, float | int]] = {}
    for class_id, class_name in CLASS_NAMES.items():
        index = class_id - 1
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        row = class_metric_row({"tp": tp, "fp": fp, "fn": fn})
        per_class[class_name] = row
        recalls.append(float(row["recall"]))
    return {
        "per_class": per_class,
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean([float(row["f1"]) for row in per_class.values()])),
        "support": total,
    }


def extract_frame_key(path: Path) -> str:
    match = re.search(r"(MI\d{2})[_-](\d+)(?!.*\d)", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot extract sequence/frame key from {path.name}")
    return f"{match.group(1).upper()}_{int(match.group(2)):04d}"


def index_masks(directory: Path) -> dict[str, Path]:
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".tif", ".tiff"})
    indexed: dict[str, Path] = {}
    for path in files:
        key = extract_frame_key(path)
        if key in indexed:
            raise ValueError(f"Duplicate frame key {key} in {directory}")
        indexed[key] = path
    return indexed


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_ready) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    result: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "scikit-image", "scikit-learn", "Pillow", "joblib"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def evaluate(args: argparse.Namespace) -> None:
    prediction_root = Path(args.prediction_root).resolve()
    gt_instance_root = Path(args.ground_truth_root).resolve()
    gt_class_root = Path(args.ground_truth_class_root).resolve()
    classifier_path = Path(args.classifier).resolve()
    classifier_config_path = Path(args.classifier_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = index_masks(prediction_root)
    ground_truth = index_masks(gt_instance_root)
    ground_truth_classes = index_masks(gt_class_root)
    keys = sorted(set(predictions) & set(ground_truth) & set(ground_truth_classes))
    keys = [key for key in keys if key.startswith(f"{args.sequence.upper()}_")]
    if not keys:
        raise ValueError("No aligned frames found for the requested sequence")
    expected = {key for key in ground_truth if key.startswith(f"{args.sequence.upper()}_")}
    if set(keys) != expected:
        missing = sorted(expected - set(keys))
        raise ValueError(f"Prediction/class alignment incomplete; missing keys: {missing}")

    classifier_config = json.loads(classifier_config_path.read_text(encoding="utf-8"))
    if tuple(classifier_config.get("features", ())) != FEATURE_NAMES:
        raise ValueError("Classifier configuration feature order does not match evaluator")
    inference_parameters = json.loads(args.inference_parameters_json)
    if not isinstance(inference_parameters, dict):
        raise ValueError("--inference-parameters-json must decode to an object")

    classifier = joblib.load(classifier_path)
    frame_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    all_counts: list[DetectionCounts] = []
    pooled_confusion = np.zeros((2, 2), dtype=np.int64)
    pooled_e2e = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in CLASS_NAMES}

    for key in keys:
        gt_mask = read_mask(ground_truth[key])
        gt_class_mask = read_mask(ground_truth_classes[key])
        pred_mask = read_mask(predictions[key])
        if not (gt_mask.shape == gt_class_mask.shape == pred_mask.shape):
            raise ValueError(f"Shape mismatch for {key}")

        counts, matches = detection_counts(gt_mask, pred_mask, args.iou_threshold)
        metrics = detection_metrics(counts)
        all_counts.append(counts)
        gt_ids = positive_ids(gt_mask)
        pred_ids = positive_ids(pred_mask)
        pred_classes = predict_instance_classes(pred_mask, classifier)
        gt_classifier_classes = predict_instance_classes(gt_mask, classifier)
        gt_features = extract_features(gt_mask).set_index("instance_id")
        pred_features = extract_features(pred_mask).set_index("instance_id")
        confusion, _ = classification_counts(matches, gt_mask, gt_class_mask, pred_classes)
        pooled_confusion += confusion
        e2e = end_to_end_class_counts(matches, gt_ids, pred_ids, gt_mask, gt_class_mask, pred_classes)
        for class_id in CLASS_NAMES:
            for field in ("tp", "fp", "fn"):
                pooled_e2e[class_id][field] += e2e[class_id][field]

        gt_class_by_id = {int(label): class_for_instance(gt_mask, gt_class_mask, int(label)) for label in gt_ids}
        gt_mitotic = sum(value == 2 for value in gt_class_by_id.values())
        pred_mitotic = sum(value == 2 for value in pred_classes.values())
        gt_total = len(gt_ids)
        pred_total = len(pred_ids)
        gt_ratio = gt_mitotic / gt_total if gt_total else 0.0
        pred_ratio = pred_mitotic / pred_total if pred_total else 0.0

        frame_rows.append(
            {
                "frame_id": key,
                **metrics,
                "gt_total": gt_total,
                "pred_total": pred_total,
                "gt_interphase": gt_total - gt_mitotic,
                "pred_interphase": pred_total - pred_mitotic,
                "gt_mitotic": gt_mitotic,
                "pred_mitotic": pred_mitotic,
                "gt_mitotic_ratio": gt_ratio,
                "pred_mitotic_ratio": pred_ratio,
                "mitotic_ratio_error": pred_ratio - gt_ratio,
                "total_count_error": pred_total - gt_total,
                "interphase_count_error": (pred_total - pred_mitotic) - (gt_total - gt_mitotic),
                "mitotic_count_error": pred_mitotic - gt_mitotic,
            }
        )

        for match in matches:
            true_class = gt_class_by_id[match.gt_id]
            predicted_class = pred_classes[match.pred_id]
            gt_classifier_class = gt_classifier_classes[match.gt_id]
            row: dict[str, Any] = {
                "frame_id": key,
                "gt_id": match.gt_id,
                "pred_id": match.pred_id,
                "iou": match.iou,
                "gt_class": true_class,
                "gt_classifier_class": gt_classifier_class,
                "pred_class": predicted_class,
                "gt_classifier_correct": int(true_class == gt_classifier_class),
                "class_correct": int(true_class == predicted_class),
                "segmentation_induced_flip": int(
                    gt_classifier_class == true_class and predicted_class != true_class
                ),
            }
            for feature_name in FEATURE_NAMES:
                gt_value = float(gt_features.loc[match.gt_id, feature_name])
                pred_value = float(pred_features.loc[match.pred_id, feature_name])
                row[f"gt_{feature_name}"] = gt_value
                row[f"pred_{feature_name}"] = pred_value
                row[f"delta_{feature_name}"] = pred_value - gt_value
            match_rows.append(row)
            if true_class != predicted_class:
                error_type = (
                    "segmentation_induced_class_flip"
                    if gt_classifier_class == true_class
                    else "classifier_error_present_on_ground_truth_shape"
                )
                error_rows.append(
                    {
                        "frame_id": key,
                        "error_type": error_type,
                        "gt_id": match.gt_id,
                        "pred_id": match.pred_id,
                        "iou": match.iou,
                        "gt_class": true_class,
                        "gt_classifier_class": gt_classifier_class,
                        "pred_class": predicted_class,
                    }
                )
        matched_gt = {match.gt_id for match in matches}
        matched_pred = {match.pred_id for match in matches}
        for gt_id in sorted(set(map(int, gt_ids)) - matched_gt):
            error_rows.append({"frame_id": key, "error_type": "false_negative", "gt_id": gt_id, "pred_id": None, "iou": None, "gt_class": gt_class_by_id[gt_id], "gt_classifier_class": gt_classifier_classes[gt_id], "pred_class": None})
        for pred_id in sorted(set(map(int, pred_ids)) - matched_pred):
            error_rows.append({"frame_id": key, "error_type": "false_positive", "gt_id": None, "pred_id": pred_id, "iou": None, "gt_class": None, "gt_classifier_class": None, "pred_class": pred_classes[pred_id]})

    frame_df = pd.DataFrame(frame_rows).sort_values("frame_id", kind="stable")
    match_columns = (
        "frame_id", "gt_id", "pred_id", "iou", "gt_class",
        "gt_classifier_class", "pred_class", "gt_classifier_correct",
        "class_correct", "segmentation_induced_flip",
        *(f"gt_{name}" for name in FEATURE_NAMES),
        *(f"pred_{name}" for name in FEATURE_NAMES),
        *(f"delta_{name}" for name in FEATURE_NAMES),
    )
    error_columns = (
        "frame_id", "error_type", "gt_id", "pred_id", "iou",
        "gt_class", "gt_classifier_class", "pred_class",
    )
    match_df = pd.DataFrame.from_records(match_rows, columns=match_columns).sort_values(
        ["frame_id", "gt_id", "pred_id"], kind="stable"
    )
    error_df = pd.DataFrame.from_records(error_rows, columns=error_columns).sort_values(
        ["frame_id", "error_type", "gt_id", "pred_id"], kind="stable", na_position="last"
    )

    frame_df.to_csv(output_dir / "metrics_by_frame.csv", index=False, lineterminator="\n", float_format="%.10g")
    match_df.to_csv(output_dir / "matched_instances.csv", index=False, lineterminator="\n", float_format="%.10g")
    error_df.to_csv(output_dir / "error_cases.csv", index=False, lineterminator="\n", float_format="%.10g")

    detection_summary = {
        "model_name": args.model_name,
        "sequence": args.sequence.upper(),
        "iou_threshold": args.iou_threshold,
        "pooled": pooled_metrics(all_counts),
        "per_frame_macro": macro_summary(frame_df),
    }
    write_json(output_dir / "metrics_summary.json", detection_summary)

    classification_only = confusion_metrics(pooled_confusion)
    end_to_end_rows = {
        CLASS_NAMES[class_id]: class_metric_row(pooled_e2e[class_id])
        for class_id in CLASS_NAMES
    }
    classifier_metrics = {
        "classes": CLASS_NAMES,
        "classification_only_confusion_matrix": pooled_confusion.tolist(),
        "classification_only": classification_only,
        "end_to_end": {
            "per_class": end_to_end_rows,
            "macro_f1": float(np.mean([float(row["f1"]) for row in end_to_end_rows.values()])),
        },
    }
    write_json(output_dir / "classifier_metrics.json", classifier_metrics)

    manifest = {
        "specification_version": SPECIFICATION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "model_name": args.model_name,
        "sequence": args.sequence.upper(),
        "iou_threshold": args.iou_threshold,
        "seed": 42,
        "feature_definitions": list(FEATURE_NAMES),
        "class_task": "ALFI annotation-class recovery; interphase masks are nuclei and mitotic masks are whole-cell regions",
        "test_status": "retrospective held-out-by-sequence; MI05 was consulted by historical development scripts",
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "packages": package_versions(),
        "classifier": {"path": str(classifier_path), "sha256": sha256_file(classifier_path)},
        "classifier_config": {
            "path": str(classifier_config_path),
            "sha256": sha256_file(classifier_config_path),
        },
        "model_artifacts": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path).resolve())}
            for path in args.model_artifact
        ],
        "inference_parameters": inference_parameters,
        "prediction_source_manifest": (
            {
                "path": str(Path(args.prediction_manifest).resolve()),
                "sha256": sha256_file(Path(args.prediction_manifest).resolve()),
            }
            if args.prediction_manifest
            else None
        ),
        "inputs": [
            {
                "frame_id": key,
                "prediction": {"path": str(predictions[key]), "sha256": sha256_file(predictions[key])},
                "ground_truth_instance": {"path": str(ground_truth[key]), "sha256": sha256_file(ground_truth[key])},
                "ground_truth_class": {"path": str(ground_truth_classes[key]), "sha256": sha256_file(ground_truth_classes[key])},
            }
            for key in keys
        ],
    }
    write_json(output_dir / "evaluation_manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--ground-truth-root", required=True)
    parser.add_argument("--ground-truth-class-root", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--classifier", required=True, help="Locked joblib classifier pipeline")
    parser.add_argument("--classifier-config", required=True)
    parser.add_argument("--model-artifact", action="append", default=[], help="Model/config/weights file to hash; repeat as needed")
    parser.add_argument("--inference-parameters-json", default="{}", help="JSON object with threshold, diameter, and other locked inference settings")
    parser.add_argument("--prediction-manifest", help="Optional upstream prediction-generation manifest")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
