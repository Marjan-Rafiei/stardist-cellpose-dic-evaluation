"""Verify preserved prediction masks and write immutable source manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from code.unified_evaluation import index_masks, read_mask, sha256_file, write_json


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset_root"]).resolve()
    sequence = str(config["sequence"]).upper()
    expected_frames = int(config["expected_frames"])
    expected_shape = tuple(int(value) for value in config["expected_shape"])
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for system_name, directory_name in sorted(config["sources"].items()):
        source = dataset_root / sequence / directory_name
        indexed = index_masks(source)
        errors: list[str] = []
        records: list[dict[str, Any]] = []
        if len(indexed) != expected_frames:
            errors.append(f"expected {expected_frames} frames, found {len(indexed)}")
        for frame_id, path in sorted(indexed.items()):
            mask = read_mask(path)
            labels = np.unique(mask)
            positive = labels[labels > 0]
            contiguous = np.array_equal(positive, np.arange(1, len(positive) + 1))
            if mask.shape != expected_shape:
                errors.append(f"{frame_id}: expected shape {expected_shape}, found {mask.shape}")
            if np.any(mask < 0):
                errors.append(f"{frame_id}: negative labels found")
            if not contiguous:
                errors.append(f"{frame_id}: instance labels are not contiguous")
            records.append(
                {
                    "frame_id": frame_id,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "shape": list(mask.shape),
                    "dtype": str(mask.dtype),
                    "instances": int(len(positive)),
                    "contiguous": bool(contiguous),
                }
            )
        manifest = {
            "schema_version": "1.0",
            "system_name": system_name,
            "sequence": sequence,
            "source_directory": str(source),
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "frame_count": len(records),
            "total_instances": int(sum(item["instances"] for item in records)),
            "empty_frames": [item["frame_id"] for item in records if item["instances"] == 0],
            "errors": errors,
            "status": "pass" if not errors else "fail",
            "files": records,
        }
        destination = output_root / f"{system_name}_prediction_manifest.json"
        write_json(destination, manifest)
        if errors:
            raise SystemExit(f"{system_name} prediction integrity failed: {errors[0]}")


if __name__ == "__main__":
    main()
