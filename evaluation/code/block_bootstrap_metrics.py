"""Moving-block bootstrap for descriptive MI05 detection metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from code.unified_evaluation import DetectionCounts, detection_metrics, write_json


METRICS = ("precision", "recall", "f1", "matched_iou", "pq")


def pooled_from_rows(frame: pd.DataFrame, indices: np.ndarray) -> dict[str, float | int | None]:
    selected = frame.iloc[indices]
    counts = DetectionCounts(
        tp=int(selected["tp"].sum()),
        fp=int(selected["fp"].sum()),
        fn=int(selected["fn"].sum()),
        matched_iou_sum=float(selected["matched_iou_sum"].sum()),
        matched_iou_count=int(selected["matched_iou_count"].sum()),
    )
    return detection_metrics(counts)


def sample_indices(rng: np.random.Generator, n: int, block_length: int) -> np.ndarray:
    if block_length < 1 or block_length > n:
        raise ValueError("block length must be between 1 and the number of frames")
    starts = rng.integers(0, n - block_length + 1, size=int(np.ceil(n / block_length)))
    sampled = np.concatenate([np.arange(start, start + block_length) for start in starts])
    return sampled[:n]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--block-length", type=int, default=7)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    root = Path(args.results_root).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    replicate_rows: list[dict[str, float | int | str | None]] = []
    summary: dict[str, object] = {}

    systems = sorted(
        directory for directory in root.iterdir()
        if directory.is_dir() and (directory / "metrics_by_frame.csv").is_file()
    )
    if not systems:
        raise ValueError(f"No evaluated systems found under {root}")

    for system_dir in systems:
        frame = pd.read_csv(system_dir / "metrics_by_frame.csv").sort_values("frame_id", kind="stable")
        frame = frame[frame["frame_id"].str.startswith(f"{args.sequence.upper()}_")].reset_index(drop=True)
        if frame.empty:
            continue
        point = pooled_from_rows(frame, np.arange(len(frame)))
        system_rows: list[dict[str, float | int | str | None]] = []
        for replicate in range(args.replicates):
            indices = sample_indices(rng, len(frame), args.block_length)
            metrics = pooled_from_rows(frame, indices)
            row = {"system": system_dir.name, "replicate": replicate, **{name: metrics[name] for name in METRICS}}
            replicate_rows.append(row)
            system_rows.append(row)
        system_frame = pd.DataFrame(system_rows)
        summary[system_dir.name] = {
            name: {
                "point_estimate": point[name],
                "percentile_2_5": float(system_frame[name].quantile(0.025)),
                "percentile_97_5": float(system_frame[name].quantile(0.975)),
            }
            for name in METRICS
        }

    pd.DataFrame(replicate_rows).to_csv(
        output_dir / "bootstrap_replicates.csv",
        index=False,
        lineterminator="\n",
        float_format="%.10g",
    )
    write_json(
        output_dir / "bootstrap_summary.json",
        {
            "schema_version": "1.0",
            "sequence": args.sequence.upper(),
            "method": "moving-block frame bootstrap",
            "block_length_frames": args.block_length,
            "replicates": args.replicates,
            "seed": args.seed,
            "interval": "2.5th and 97.5th percentiles; descriptive only",
            "limitation": "one temporally dependent sequence; no between-experiment uncertainty",
            "systems": summary,
        },
    )


if __name__ == "__main__":
    main()
