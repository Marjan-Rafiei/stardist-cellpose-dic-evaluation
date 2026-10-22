"""Audit MI04 permutation importance under correlation and repeated shuffling."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score

from code.train_classifier import TRAIN_SEQUENCES, collect_rows
from code.unified_evaluation import FEATURE_NAMES, sha256_file, write_json


FEATURE_GROUPS = OrderedDict(
    (
        ("elongation", ("eccentricity", "aspect_ratio")),
        ("size", ("area", "perimeter", "major_axis_length", "minor_axis_length")),
        ("compactness", ("solidity", "extent")),
    )
)


def correlation_rows(frame: pd.DataFrame, split: str) -> list[dict[str, object]]:
    matrix = frame.loc[:, FEATURE_NAMES].corr(method="spearman")
    return [
        {
            "split": split,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "spearman_correlation": float(matrix.loc[feature_a, feature_b]),
            "absolute_spearman_correlation": float(abs(matrix.loc[feature_a, feature_b])),
        }
        for feature_a in FEATURE_NAMES
        for feature_b in FEATURE_NAMES
    ]


def summarize_runs(run_frame: pd.DataFrame, draws: pd.DataFrame, key: str) -> pd.DataFrame:
    run_summary = run_frame.groupby(key, sort=False).agg(
        between_run_std=("importance_mean", "std"),
        minimum_run_mean=("importance_mean", "min"),
        maximum_run_mean=("importance_mean", "max"),
        median_rank=("rank", "median"),
        minimum_rank=("rank", "min"),
        maximum_rank=("rank", "max"),
    )
    draw_summary = draws.groupby(key, sort=False).agg(
        importance_mean=("importance_drop", "mean"),
        importance_std=("importance_drop", lambda values: float(np.std(values, ddof=0))),
        permutations=("importance_drop", "size"),
    )
    return draw_summary.join(run_summary).reset_index()


def high_correlation_pairs(correlations: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    order = {feature: index for index, feature in enumerate(FEATURE_NAMES)}
    selected = correlations[
        correlations.apply(
            lambda row: order[row["feature_a"]] < order[row["feature_b"]]
            and row["absolute_spearman_correlation"] >= threshold,
            axis=1,
        )
    ].copy()
    return selected.sort_values(
        ["split", "absolute_spearman_correlation", "feature_a", "feature_b"],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--training-rows", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--current-importance", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args(argv)
    if args.runs < 2:
        raise ValueError("At least two runs are required for a stability audit")
    if args.repeats < 2:
        raise ValueError("At least two repeats per run are required")

    dataset_root = Path(args.dataset_root).resolve()
    classifier_path = Path(args.classifier).resolve()
    training_path = Path(args.training_rows).resolve()
    validation_path = Path(args.validation).resolve()
    current_importance_path = Path(args.current_importance).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier = joblib.load(classifier_path)
    saved_training_rows = pd.read_csv(training_path)
    training = collect_rows(dataset_root, TRAIN_SEQUENCES)
    validation = pd.read_csv(validation_path)
    current = pd.read_csv(current_importance_path)
    required = set(FEATURE_NAMES) | {"target"}
    expected_training_rows = training.loc[:, ["sequence", "frame_id", "instance_id", "target"]]
    if not expected_training_rows.equals(saved_training_rows):
        raise ValueError("Reconstructed training identifiers or targets do not match the saved rows")
    for name, frame in (("training", training), ("validation", validation)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} data are missing columns: {missing}")
        if not np.isfinite(frame.loc[:, FEATURE_NAMES].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} features contain nonfinite values")
    if set(current.columns) != {"feature", "importance_mean", "importance_std"}:
        raise ValueError("Current importance file has an unexpected schema")
    if set(current["feature"]) != set(FEATURE_NAMES):
        raise ValueError("Current importance file does not contain the fixed feature set")

    x = validation.loc[:, FEATURE_NAMES]
    y = validation["target"].astype(int)
    predicted = classifier.predict(x).astype(int)
    if "predicted" in validation and not np.array_equal(predicted, validation["predicted"].astype(int)):
        raise ValueError("Classifier predictions do not match the saved MI04 predictions")
    baseline_score = float(balanced_accuracy_score(y, predicted))
    seeds = list(range(args.seed_start, args.seed_start + args.runs))

    individual_run_rows: list[dict[str, object]] = []
    individual_draw_rows: list[dict[str, object]] = []
    for seed in seeds:
        result = permutation_importance(
            classifier,
            x,
            y,
            scoring="balanced_accuracy",
            n_repeats=args.repeats,
            random_state=seed,
            n_jobs=1,
        )
        run = pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        run["rank"] = run["importance_mean"].rank(method="min", ascending=False).astype(int)
        for row in run.itertuples(index=False):
            individual_run_rows.append(
                {
                    "seed": seed,
                    "feature": row.feature,
                    "importance_mean": float(row.importance_mean),
                    "importance_std": float(row.importance_std),
                    "rank": int(row.rank),
                }
            )
        for feature_index, feature in enumerate(FEATURE_NAMES):
            for repeat, drop in enumerate(result.importances[feature_index], start=1):
                individual_draw_rows.append(
                    {
                        "seed": seed,
                        "repeat": repeat,
                        "feature": feature,
                        "importance_drop": float(drop),
                    }
                )

    group_run_rows: list[dict[str, object]] = []
    group_draw_rows: list[dict[str, object]] = []
    for seed in seeds:
        seed_group_means: list[dict[str, object]] = []
        for group_index, (group, features) in enumerate(FEATURE_GROUPS.items()):
            rng = np.random.default_rng(np.random.SeedSequence((seed, group_index)))
            drops = []
            for repeat in range(1, args.repeats + 1):
                permutation = rng.permutation(len(x))
                permuted = x.copy()
                permuted.loc[:, list(features)] = x.iloc[permutation][list(features)].to_numpy()
                score = balanced_accuracy_score(y, classifier.predict(permuted))
                drop = float(baseline_score - score)
                drops.append(drop)
                group_draw_rows.append(
                    {
                        "seed": seed,
                        "repeat": repeat,
                        "group": group,
                        "features": ";".join(features),
                        "importance_drop": drop,
                    }
                )
            seed_group_means.append(
                {
                    "seed": seed,
                    "group": group,
                    "features": ";".join(features),
                    "importance_mean": float(np.mean(drops)),
                    "importance_std": float(np.std(drops, ddof=0)),
                }
            )
        ranked = pd.DataFrame(seed_group_means)
        ranked["rank"] = ranked["importance_mean"].rank(method="min", ascending=False).astype(int)
        group_run_rows.extend(ranked.to_dict(orient="records"))

    individual_runs = pd.DataFrame(individual_run_rows)
    individual_draws = pd.DataFrame(individual_draw_rows)
    group_runs = pd.DataFrame(group_run_rows)
    group_draws = pd.DataFrame(group_draw_rows)
    individual_summary = summarize_runs(individual_runs, individual_draws, "feature")
    group_summary = summarize_runs(group_runs, group_draws, "group")
    group_features = group_runs[["group", "features"]].drop_duplicates()
    group_summary = group_summary.merge(group_features, on="group", how="left", validate="one_to_one")

    current_ranked = current.copy()
    current_ranked["current_rank"] = current_ranked["importance_mean"].rank(
        method="min", ascending=False
    ).astype(int)
    current_ranked = current_ranked.rename(
        columns={
            "importance_mean": "current_importance_mean",
            "importance_std": "current_importance_std",
        }
    )
    individual_summary = individual_summary.merge(
        current_ranked, on="feature", how="left", validate="one_to_one"
    ).sort_values(["importance_mean", "feature"], ascending=[False, True], kind="stable")
    group_summary = group_summary.sort_values(
        ["importance_mean", "group"], ascending=[False, True], kind="stable"
    )

    correlations = pd.DataFrame(
        correlation_rows(training, "training") + correlation_rows(validation, "validation")
    )
    high_correlations = high_correlation_pairs(correlations)

    outputs = {
        "individual_importance_runs.csv": individual_runs,
        "individual_importance_draws.csv": individual_draws,
        "individual_importance_stability.csv": individual_summary,
        "grouped_importance_runs.csv": group_runs,
        "grouped_importance_draws.csv": group_draws,
        "grouped_importance_summary.csv": group_summary,
        "feature_correlations.csv": correlations,
        "high_correlation_pairs.csv": high_correlations,
    }
    for name, frame in outputs.items():
        frame.to_csv(
            output_dir / name,
            index=False,
            lineterminator="\n",
            float_format="%.10g",
        )

    top_individual = individual_summary.iloc[0]
    top_group = group_summary.iloc[0]
    stability_exact = bool((individual_summary["minimum_rank"] == individual_summary["maximum_rank"]).all())
    summary = {
        "schema_version": "1.0",
        "sequence": "MI04",
        "scoring": "balanced_accuracy",
        "baseline_balanced_accuracy": baseline_score,
        "stability_design": {
            "seeds": seeds,
            "runs": args.runs,
            "permutations_per_run": args.repeats,
            "total_permutations_per_feature_or_group": args.runs * args.repeats,
        },
        "individual_importance": {
            "top_feature": top_individual["feature"],
            "top_mean_drop": float(top_individual["importance_mean"]),
            "all_rankings_identical_across_runs": stability_exact,
            "interpretation": "predictive reliance within this fitted classifier and feature set; not causal or uniquely attributable biological importance",
        },
        "grouped_importance": {
            "top_group": top_group["group"],
            "top_mean_drop": float(top_group["importance_mean"]),
            "groups": {group: list(features) for group, features in FEATURE_GROUPS.items()},
            "method": "all features in a group receive the same row permutation, preserving their within-group joint structure",
            "limit": "group permutation can still create unrealistic combinations between groups and is not conditional permutation importance",
        },
        "correlation": {
            "threshold_for_high_pair": 0.8,
            "high_pair_count": int(len(high_correlations)),
            "strongest_pairs": high_correlations.to_dict(orient="records"),
        },
        "claim_correction": {
            "avoid": "Area is the most important biological determinant of mitosis.",
            "preferred": "Within this fitted classifier and feature set, eccentricity produced the largest mean permutation-based decrease in MI04 balanced accuracy.",
            "required_caveat": "Correlated features make individual rankings non-unique and can split, mask, or distort apparent importance.",
        },
        "validations": {
            "training_rows_match_saved": bool(expected_training_rows.equals(saved_training_rows)),
            "feature_set_exact": set(current["feature"]) == set(FEATURE_NAMES),
            "saved_predictions_match": bool(np.array_equal(predicted, validation["predicted"].astype(int))),
            "all_individual_drops_finite": bool(np.isfinite(individual_draws["importance_drop"]).all()),
            "all_group_drops_finite": bool(np.isfinite(group_draws["importance_drop"]).all()),
            "all_features_present_each_run": bool(
                (individual_runs.groupby("seed")["feature"].nunique() == len(FEATURE_NAMES)).all()
            ),
            "all_groups_present_each_run": bool(
                (group_runs.groupby("seed")["group"].nunique() == len(FEATURE_GROUPS)).all()
            ),
            "mi05_used": False,
        },
    }
    summary_path = output_dir / "audit_summary.json"
    write_json(summary_path, summary)

    output_paths = [output_dir / name for name in outputs] + [summary_path]
    manifest_path = output_dir / "analysis_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "1.0",
            "status": "verified MI04 permutation-importance correlation and stability audit",
            "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
            "parameters": {
                "dataset_root": str(dataset_root),
                "training_sequences": list(TRAIN_SEQUENCES),
                "seed_start": args.seed_start,
                "runs": args.runs,
                "repeats": args.repeats,
                "scoring": "balanced_accuracy",
            },
            "actions_performed": {
                "classifier_training": False,
                "feature_selection": False,
                "hyperparameter_tuning": False,
                "mi05_use": False,
                "validation_only_permutation": True,
            },
            "inputs": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in (
                    classifier_path,
                    training_path,
                    validation_path,
                    current_importance_path,
                )
            ],
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path)} for path in output_paths
            ],
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        },
    )


if __name__ == "__main__":
    main()
