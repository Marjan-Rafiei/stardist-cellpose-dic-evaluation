"""Train, validate, and lock the morphology classifier under the fixed split."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from code.unified_evaluation import (
    CLASS_NAMES,
    FEATURE_NAMES,
    class_for_instance,
    extract_features,
    index_masks,
    read_mask,
    sha256_file,
    write_json,
)


TRAIN_SEQUENCES = ("MI01", "MI02", "MI03")
VALIDATION_SEQUENCE = "MI04"
FORBIDDEN_TRAINING_SEQUENCE = "MI05"


def validate_split(training_sequences: Sequence[str], validation_sequence: str) -> None:
    normalized = tuple(item.upper() for item in training_sequences)
    validation = validation_sequence.upper()
    if set(normalized) != set(TRAIN_SEQUENCES):
        raise ValueError(f"Training split must be exactly {TRAIN_SEQUENCES}")
    if validation != VALIDATION_SEQUENCE:
        raise ValueError(f"Validation sequence must be {VALIDATION_SEQUENCE}")
    if FORBIDDEN_TRAINING_SEQUENCE in normalized or validation == FORBIDDEN_TRAINING_SEQUENCE:
        raise ValueError("MI05 is reserved for final testing")


def collect_rows(dataset_root: Path, sequences: Sequence[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sequence in sequences:
        image_files = index_masks(dataset_root / sequence / "Images")
        instance_files = index_masks(dataset_root / sequence / "Masks_converted")
        class_files = index_masks(dataset_root / sequence / "Masks_classes")
        keys = sorted(set(image_files) & set(instance_files) & set(class_files))
        if not keys:
            raise ValueError(f"No aligned image/instance/class rows for {sequence}")
        for frame_id in keys:
            instance_mask = read_mask(instance_files[frame_id])
            class_mask = read_mask(class_files[frame_id])
            features = extract_features(instance_mask)
            if features.empty:
                continue
            features.insert(0, "frame_id", frame_id)
            features.insert(0, "sequence", sequence)
            features["target"] = [
                class_for_instance(instance_mask, class_mask, int(instance_id))
                for instance_id in features["instance_id"]
            ]
            rows.append(features)
    if not rows:
        raise ValueError("No classifier rows were collected")
    result = pd.concat(rows, ignore_index=True)
    if any(result["sequence"].str.upper() == FORBIDDEN_TRAINING_SEQUENCE):
        raise ValueError("Leakage guard: MI05 rows are forbidden")
    return result.sort_values(["sequence", "frame_id", "instance_id"], kind="stable").reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_sequences = tuple(config["training_sequences"])
    validation_sequence = config["validation_sequences"][0]
    validate_split(training_sequences, validation_sequence)

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = collect_rows(dataset_root, training_sequences)
    validation = collect_rows(dataset_root, (validation_sequence,))

    pipeline = build_pipeline()
    pipeline.fit(train.loc[:, FEATURE_NAMES], train["target"])
    predicted = pipeline.predict(validation.loc[:, FEATURE_NAMES])

    model_path = output_dir / "classifier.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    train.loc[:, ["sequence", "frame_id", "instance_id", "target"]].to_csv(
        output_dir / "training_rows.csv", index=False, lineterminator="\n"
    )
    validation_output = validation.loc[:, ["sequence", "frame_id", "instance_id", *FEATURE_NAMES, "target"]].assign(
        predicted=predicted
    )
    validation_output.to_csv(
        output_dir / "validation_predictions.csv", index=False, lineterminator="\n", float_format="%.10g"
    )

    metrics = {
        "training_sequences": list(training_sequences),
        "validation_sequence": validation_sequence,
        "features": list(FEATURE_NAMES),
        "seed": 42,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "confusion_matrix_labels": list(CLASS_NAMES),
        "confusion_matrix": confusion_matrix(validation["target"], predicted, labels=list(CLASS_NAMES)).tolist(),
        "classification_report": classification_report(
            validation["target"],
            predicted,
            labels=list(CLASS_NAMES),
            target_names=[CLASS_NAMES[item] for item in CLASS_NAMES],
            output_dict=True,
            zero_division=0,
        ),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "classifier": {"path": str(model_path), "sha256": sha256_file(model_path)},
    }
    write_json(output_dir / "classifier_validation.json", metrics)

    packages: dict[str, str] = {}
    for package in ("numpy", "pandas", "scikit-image", "scikit-learn", "joblib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "unavailable"
    manifest = {
        "schema_version": "1.0",
        "specification_version": "2.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": packages,
        "training_sequences": list(training_sequences),
        "validation_sequence": validation_sequence,
        "forbidden_sequence": FORBIDDEN_TRAINING_SEQUENCE,
        "features": list(FEATURE_NAMES),
        "class_task": "ALFI annotation-class recovery; class masks represent different compartments",
        "classifier_parameters": {
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 1,
            "imputer": "median fitted on MI01-MI03 only",
        },
        "row_counts": {"training": int(len(train)), "validation": int(len(validation))},
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "classifier": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "training_rows_sha256": sha256_file(output_dir / "training_rows.csv"),
        "validation_predictions_sha256": sha256_file(output_dir / "validation_predictions.csv"),
    }
    write_json(output_dir / "classifier_manifest.json", manifest)


if __name__ == "__main__":
    main()
