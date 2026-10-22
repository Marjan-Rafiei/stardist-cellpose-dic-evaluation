"""Shared deterministic provenance helper for generated figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from code.unified_evaluation import sha256_file, write_json


def write_figure_manifest(
    output_dir: Path,
    script_path: Path,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    notes: str,
) -> None:
    input_paths = sorted({path.resolve() for path in inputs})
    output_paths = sorted({path.resolve() for path in outputs})
    write_json(
        output_dir / "figure_manifest.json",
        {
            "schema_version": "1.0",
            "script": {"path": str(script_path.resolve()), "sha256": sha256_file(script_path.resolve())},
            "inputs": [
                {"path": str(path), "sha256": sha256_file(path)} for path in input_paths
            ],
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path)} for path in output_paths
            ],
            "notes": notes,
        },
    )
