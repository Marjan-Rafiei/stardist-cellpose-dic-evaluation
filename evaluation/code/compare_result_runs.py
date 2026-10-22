"""Compare deterministic analytical outputs from two evaluator runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from code.unified_evaluation import sha256_file, write_json


ANALYTICAL_FILES = {
    "metrics_by_frame.csv",
    "metrics_summary.json",
    "matched_instances.csv",
    "classifier_metrics.json",
    "error_cases.csv",
}


def collect(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name in ANALYTICAL_FILES
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    first_root = Path(args.first).resolve()
    second_root = Path(args.second).resolve()
    first = collect(first_root)
    second = collect(second_root)
    keys = sorted(set(first) | set(second))
    comparisons = []
    for key in keys:
        first_hash = sha256_file(first[key]) if key in first else None
        second_hash = sha256_file(second[key]) if key in second else None
        comparisons.append(
            {
                "relative_path": key,
                "first_sha256": first_hash,
                "second_sha256": second_hash,
                "byte_identical": first_hash is not None and first_hash == second_hash,
            }
        )
    passed = bool(comparisons) and all(item["byte_identical"] for item in comparisons)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output_path,
        {
            "schema_version": "1.0",
            "first_root": str(first_root),
            "second_root": str(second_root),
            "files_compared": len(comparisons),
            "status": "pass" if passed else "fail",
            "comparisons": comparisons,
        },
    )
    if not passed:
        raise SystemExit("Determinism comparison failed")


if __name__ == "__main__":
    main()
