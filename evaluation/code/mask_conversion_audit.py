"""Audit the historical mask-conversion fault against every ALFI mask.

The audit reproduces the documented unconditional uint8 multiplication in
memory, regenerates the corrected instance and class masks, and compares the
regenerated arrays with the currently stored outputs. Source data are never
modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import skimage
from skimage import io, measure


SEQUENCES = ("MI01", "MI02", "MI03", "MI04", "MI05")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_hash(paths: Sequence[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        record = f"{path.relative_to(base).as_posix()}|{sha256_file(path)}\n"
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_mask_uint8(mask: np.ndarray) -> np.ndarray:
    """Apply the corrected conversion branch used by convert_masks.py."""
    if mask.size == 0:
        raise ValueError("Mask is empty")
    if float(mask.max()) <= 1.0:
        return (mask * 255).astype(np.uint8)
    return mask.astype(np.uint8)


def faulty_unconditional_uint8(mask: np.ndarray) -> np.ndarray:
    """Reproduce the documented pre-fix expression under NumPy semantics."""
    return (mask * 255).astype(np.uint8)


def convert_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Generate instance and class masks with the corrected logic."""
    prepared = prepare_mask_uint8(mask)
    interphase_pixels = (prepared >= 50) & (prepared <= 200)
    mitotic_pixels = prepared > 200
    interphase = measure.label(interphase_pixels, connectivity=2)
    mitotic = measure.label(mitotic_pixels, connectivity=2)
    mitotic_offset = mitotic.copy()
    mitotic_offset[mitotic > 0] += int(interphase.max())
    instances = interphase + mitotic_offset
    classes = np.zeros_like(instances, dtype=np.uint8)
    classes[interphase_pixels] = 1
    classes[mitotic_pixels] = 2
    return instances, classes, int(interphase.max()), int(mitotic.max())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--converter", required=True)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--raw-check", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root).resolve()
    converter = Path(args.converter).resolve()
    diagnostic = Path(args.diagnostic).resolve()
    raw_check = Path(args.raw_check).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    evidence_files = (converter, diagnostic, raw_check)
    for path in evidence_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_paths: list[Path] = []
    instance_paths: list[Path] = []
    class_paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    all_dtypes: set[str] = set()
    all_values: set[int] = set()
    masks_with_255 = 0
    total_255_pixels = 0
    total_interphase_components = 0
    total_mitotic_components = 0
    instance_matches = 0
    class_matches = 0
    example_frame = ""

    for sequence in SEQUENCES:
        mask_dir = dataset_root / sequence / "Masks"
        stored_instance_dir = dataset_root / sequence / "Masks_converted"
        stored_class_dir = dataset_root / sequence / "Masks_classes"
        masks = sorted(mask_dir.glob("*.png"))
        if not masks:
            raise FileNotFoundError(f"No raw masks found in {mask_dir}")
        for raw_path in masks:
            instance_path = stored_instance_dir / raw_path.name
            class_path = stored_class_dir / raw_path.name
            if not instance_path.is_file() or not class_path.is_file():
                raise FileNotFoundError(f"Missing converted output for {raw_path}")

            raw = io.imread(raw_path, as_gray=True)
            old = faulty_unconditional_uint8(raw)
            regenerated_instance, regenerated_class, n_interphase, n_mitotic = convert_mask(raw)
            stored_instance = io.imread(instance_path)
            stored_class = io.imread(class_path)
            values = np.unique(raw).astype(int).tolist()
            old_values = np.unique(old).astype(int).tolist()
            pixel_255 = int(np.count_nonzero(raw == 255))
            faulty_mitotic_pixels = int(np.count_nonzero(old > 200))
            instance_equal = bool(np.array_equal(regenerated_instance, stored_instance))
            class_equal = bool(np.array_equal(regenerated_class, stored_class))

            all_dtypes.add(str(raw.dtype))
            all_values.update(values)
            masks_with_255 += int(pixel_255 > 0)
            total_255_pixels += pixel_255
            total_interphase_components += n_interphase
            total_mitotic_components += n_mitotic
            instance_matches += int(instance_equal)
            class_matches += int(class_equal)
            if not example_frame and pixel_255 > 0 and 128 in values:
                example_frame = raw_path.stem

            raw_paths.append(raw_path)
            instance_paths.append(instance_path)
            class_paths.append(class_path)
            rows.append(
                {
                    "sequence": sequence,
                    "frame": raw_path.stem,
                    "as_gray_dtype": str(raw.dtype),
                    "as_gray_values": ";".join(map(str, values)),
                    "faulty_values": ";".join(map(str, old_values)),
                    "raw_255_pixels": pixel_255,
                    "faulty_pixels_above_200": faulty_mitotic_pixels,
                    "corrected_interphase_components": n_interphase,
                    "corrected_mitotic_components": n_mitotic,
                    "instance_exact_match": str(instance_equal).lower(),
                    "class_exact_match": str(class_equal).lower(),
                    "raw_sha256": sha256_file(raw_path),
                    "stored_instance_sha256": sha256_file(instance_path),
                    "stored_class_sha256": sha256_file(class_path),
                }
            )

    total_masks = len(rows)
    if not example_frame:
        raise ValueError("No mask containing both 128 and 255 was found")

    mapping_rows = []
    for value in (0, 128, 255):
        raw_value = np.asarray([value], dtype=np.uint8)
        faulty_value = int(faulty_unconditional_uint8(raw_value)[0])
        corrected_value = int(prepare_mask_uint8(raw_value)[0])
        mapping_rows.append(
            {
                "input_uint8": value,
                "faulty_unconditional_output": faulty_value,
                "corrected_output": corrected_value,
                "passes_mitotic_threshold_after_fault": str(faulty_value > 200).lower(),
            }
        )

    normalized_probe = np.asarray([0.0, 128.0 / 255.0, 1.0], dtype=np.float64)
    uint8_probe = np.asarray([0, 128, 255], dtype=np.uint8)
    summary = {
        "schema_version": "1.0",
        "status": "verified computational fault mechanism and verified current outputs",
        "scope": "all raw masks and current converted masks in MI01 to MI05",
        "total_masks": total_masks,
        "as_gray_dtypes": sorted(all_dtypes),
        "as_gray_values": sorted(all_values),
        "masks_with_value_255": masks_with_255,
        "total_value_255_pixels": total_255_pixels,
        "corrected_interphase_components": total_interphase_components,
        "corrected_mitotic_components": total_mitotic_components,
        "corrected_instance_exact_matches": instance_matches,
        "corrected_class_exact_matches": class_matches,
        "example_frame": example_frame,
        "faulty_value_mapping": {str(row["input_uint8"]): row["faulty_unconditional_output"] for row in mapping_rows},
        "fault_consequence": {
            "mitotic_threshold": "prepared mask > 200",
            "value_255_after_fault": 1,
            "value_255_pixels_remaining_above_200": sum(
                int(row["faulty_pixels_above_200"]) for row in rows
            ),
            "verified_interpretation": "All raw 255 pixels fail the later mitotic threshold after the reproduced unconditional uint8 multiplication.",
        },
        "corrected_branch_tests": {
            "normalized_float_input": normalized_probe.tolist(),
            "normalized_float_output": prepare_mask_uint8(normalized_probe).astype(int).tolist(),
            "uint8_input": uint8_probe.astype(int).tolist(),
            "uint8_output": prepare_mask_uint8(uint8_probe).astype(int).tolist(),
        },
        "evidence_status": {
            "fault_mechanism": "verified on actual data with the documented expression",
            "current_converted_masks": "verified by exact array comparison",
            "pre_fix_production_source": "not preserved or versioned",
            "specific_historical_training_run_impact": "unresolved",
        },
        "validations": {
            "all_as_gray_reads_are_uint8": all_dtypes == {"uint8"},
            "all_as_gray_values_are_0_128_255": all_values == {0, 128, 255},
            "faulty_mapping_is_0_128_1": [row["faulty_unconditional_output"] for row in mapping_rows] == [0, 128, 1],
            "normalized_branch_is_correct": prepare_mask_uint8(normalized_probe).astype(int).tolist() == [0, 128, 255],
            "uint8_branch_is_correct": prepare_mask_uint8(uint8_probe).astype(int).tolist() == [0, 128, 255],
            "all_current_instance_masks_match": instance_matches == total_masks,
            "all_current_class_masks_match": class_matches == total_masks,
            "all_faulty_255_pixels_fail_mitotic_threshold": sum(
                int(row["faulty_pixels_above_200"]) for row in rows
            ) == 0,
        },
    }

    comparison_path = output / "mask_comparison.csv"
    mapping_path = output / "value_mapping.csv"
    summary_path = output / "conversion_summary.json"
    manifest_path = output / "analysis_manifest.json"
    write_csv(comparison_path, rows, tuple(rows[0]))
    write_csv(mapping_path, mapping_rows, tuple(mapping_rows[0]))
    write_json(summary_path, summary)

    outputs = (comparison_path, mapping_path, summary_path)
    manifest = {
        "schema_version": "1.0",
        "analysis": "mask conversion fault audit",
        "command": [
            str(Path(sys.executable).resolve()),
            str(Path(__file__).resolve()),
            "--dataset-root",
            str(dataset_root),
            "--converter",
            str(converter),
            "--diagnostic",
            str(diagnostic),
            "--raw-check",
            str(raw_check),
            "--output",
            str(output),
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_image": skimage.__version__,
            "platform": platform.platform(),
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "historical_evidence": [
            {"path": str(path), "sha256": sha256_file(path)} for path in evidence_files
        ],
        "input_collections": {
            "raw_masks": {"count": len(raw_paths), "aggregate_sha256": aggregate_hash(raw_paths, dataset_root)},
            "stored_instance_masks": {"count": len(instance_paths), "aggregate_sha256": aggregate_hash(instance_paths, dataset_root)},
            "stored_class_masks": {"count": len(class_paths), "aggregate_sha256": aggregate_hash(class_paths, dataset_root)},
        },
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in outputs
        ],
        "interpretation_boundary": "The mechanism and current outputs are verified. The audit does not attribute the fault to a specific historical model-training run because the pre-fix production source was not preserved in version control.",
    }
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
