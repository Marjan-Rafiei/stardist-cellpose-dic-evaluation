"""Generate missing StarDist instance masks from the preserved saved model.

This command performs inference only. It refuses sequences other than MI05 so
that it cannot accidentally become a training or tuning entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

from csbdeep.utils import normalize
from PIL import Image
from stardist.models import StarDist2D

from code.unified_evaluation import extract_frame_key, sha256_file, write_json


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("numpy", "Pillow", "stardist", "csbdeep", "tensorflow"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    sequence = args.sequence.upper()
    if sequence != "MI05":
        raise ValueError("Locked inference command accepts MI05 only")

    dataset_root = Path(args.dataset_root).resolve()
    image_root = dataset_root / sequence / "Images"
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_masks = sorted(output_dir.glob("*.png"))
    if existing_masks:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing_masks)} existing prediction masks in {output_dir}"
        )

    random.seed(42)
    np.random.seed(42)

    thresholds_path = model_dir / "thresholds.json"
    config_path = model_dir / "config.json"
    weights_path = model_dir / "weights_best.h5"
    for required in (thresholds_path, config_path, weights_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    probability_threshold = float(thresholds["prob"])
    nms_threshold = float(thresholds["nms"])

    model = StarDist2D(None, name=model_dir.name, basedir=str(model_dir.parent))
    model.load_weights("weights_best.h5")
    images = sorted(path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".tif", ".tiff"})
    if len(images) != 69:
        raise ValueError(f"Expected 69 MI05 frames, found {len(images)}")

    input_records: list[dict[str, str]] = []
    output_records: list[dict[str, str | int]] = []
    for image_path in images:
        frame_id = extract_frame_key(image_path)
        image = np.asarray(Image.open(image_path)).astype(np.float32)
        normalized = normalize(image, 1, 99.8, axis=(0, 1))
        labels, _ = model.predict_instances(
            normalized,
            prob_thresh=probability_threshold,
            nms_thresh=nms_threshold,
        )
        if int(labels.max()) > np.iinfo(np.uint16).max:
            raise ValueError(f"Instance labels exceed uint16 for {frame_id}")
        output_path = output_dir / f"S_{frame_id}.png"
        Image.fromarray(labels.astype(np.uint16, copy=False)).save(output_path)
        input_records.append({"frame_id": frame_id, "path": str(image_path), "sha256": sha256_file(image_path)})
        output_records.append(
            {
                "frame_id": frame_id,
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "instances": int(labels.max()),
            }
        )

    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "operation": "inference_only",
        "seed": 42,
        "determinism_environment": {
            key: os.environ.get(key)
            for key in (
                "TF_DETERMINISTIC_OPS",
                "TF_ENABLE_ONEDNN_OPTS",
                "OMP_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS",
            )
        },
        "sequence": sequence,
        "normalization": {"method": "csbdeep.normalize", "percentile_low": 1.0, "percentile_high": 99.8, "axis": [0, 1]},
        "thresholds": {"probability": probability_threshold, "nms": nms_threshold},
        "model_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (config_path, thresholds_path, weights_path)
        ],
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "packages": package_versions(),
        "inputs": input_records,
        "outputs": output_records,
    }
    write_json(output_dir / "prediction_manifest.json", manifest)


if __name__ == "__main__":
    main()
