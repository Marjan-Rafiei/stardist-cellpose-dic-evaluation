"""Create deterministic MI04 diagnostics for the locked classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from code.unified_evaluation import CLASS_NAMES, FEATURE_NAMES, sha256_file, write_json


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    classifier_path = Path(args.classifier).resolve()
    validation_path = Path(args.validation).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier = joblib.load(classifier_path)
    frame = pd.read_csv(validation_path)
    missing = sorted(set((*FEATURE_NAMES, "target")) - set(frame.columns))
    if missing:
        raise ValueError(f"Validation file is missing columns: {missing}")
    x = frame.loc[:, FEATURE_NAMES]
    y = frame["target"].astype(int)
    predicted = classifier.predict(x)
    labels = list(CLASS_NAMES)
    confusion = confusion_matrix(y, predicted, labels=labels)
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_totals,
        out=np.zeros_like(confusion, dtype=float),
        where=row_totals > 0,
    )

    importance = permutation_importance(
        classifier,
        x,
        y,
        scoring="balanced_accuracy",
        n_repeats=50,
        random_state=args.seed,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values(["importance_mean", "feature"], ascending=[False, True], kind="stable")
    importance_frame.to_csv(
        output_dir / "permutation_importance.csv",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    distributions = []
    for class_id, class_name in CLASS_NAMES.items():
        subset = frame[frame["target"] == class_id]
        for feature in FEATURE_NAMES:
            values = pd.to_numeric(subset[feature], errors="coerce").dropna()
            distributions.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "feature": feature,
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(values.median()),
                    "q1": float(values.quantile(0.25)),
                    "q3": float(values.quantile(0.75)),
                }
            )
    pd.DataFrame(distributions).to_csv(
        output_dir / "feature_distributions.csv",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )

    payload = {
        "schema_version": "1.0",
        "sequence": "MI04",
        "seed": args.seed,
        "classifier_sha256": sha256_file(classifier_path),
        "validation_sha256": sha256_file(validation_path),
        "support": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
        "classification_report": classification_report(
            y,
            predicted,
            labels=labels,
            target_names=[CLASS_NAMES[item] for item in labels],
            output_dict=True,
            zero_division=0,
        ),
        "interpretation_limit": "ALFI interphase masks are nuclei and mitotic masks are whole-cell regions",
    }
    write_json(output_dir / "classifier_diagnostics.json", payload)


if __name__ == "__main__":
    main()
