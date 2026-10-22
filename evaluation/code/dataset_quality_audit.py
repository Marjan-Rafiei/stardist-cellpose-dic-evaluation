"""Run the Step 14 dataset and data-quality audit without changing model inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import PIL
import scipy
from PIL import Image
from scipy import ndimage

from code.unified_evaluation import index_masks, sha256_file, write_json


ROLE_DIRS = {
    "image": "Images",
    "raw_mask": "Masks",
    "instance_mask": "Masks_converted",
    "class_mask": "Masks_classes",
}


def stable_membership_hash(members: Iterable[tuple[str, str]]) -> str:
    """Hash sorted logical identifiers and content hashes, independent of paths."""
    payload = "".join(
        f"{identifier}\t{digest}\n" for identifier, digest in sorted(members)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contiguous_instance_ids(mask: np.ndarray) -> bool:
    labels = np.unique(mask)
    labels = labels[labels > 0]
    return np.array_equal(labels, np.arange(1, len(labels) + 1))


def connected_instance_ids(mask: np.ndarray) -> bool:
    structure = np.ones((3, 3), dtype=np.uint8)
    for instance_id in np.unique(mask):
        if instance_id <= 0:
            continue
        if ndimage.label(mask == instance_id, structure=structure)[1] != 1:
            return False
    return True


def raw_class_mapping_valid(
    raw_mask: np.ndarray,
    class_mask: np.ndarray,
    mapping: dict[int, int],
) -> bool:
    if raw_mask.shape != class_mask.shape:
        return False
    expected = np.zeros(raw_mask.shape, dtype=class_mask.dtype)
    for raw_value, class_value in mapping.items():
        expected[raw_mask == raw_value] = class_value
    return np.array_equal(expected, class_mask)


def instance_class_mapping_valid(
    instance_mask: np.ndarray,
    class_mask: np.ndarray,
    allowed_classes: set[int],
) -> bool:
    if instance_mask.shape != class_mask.shape:
        return False
    if not np.array_equal(instance_mask > 0, class_mask > 0):
        return False
    for instance_id in np.unique(instance_mask):
        if instance_id <= 0:
            continue
        values = set(np.unique(class_mask[instance_mask == instance_id]).astype(int))
        if len(values) != 1 or not values <= allowed_classes:
            return False
    return True


def parse_latex_macros(path: Path) -> dict[str, str]:
    pattern = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")
    return dict(pattern.findall(path.read_text(encoding="utf-8")))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior-audit-dir", required=True)
    parser.add_argument("--thesis-macros", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset_root"]).resolve()
    prior_dir = Path(args.prior_audit_dir).resolve()
    macros_path = Path(args.thesis_macros).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = tuple(config["sequences"])
    expected_shape = tuple(config["expected_shape"])
    expected_dtypes = config["expected_dtypes"]
    allowed_raw = set(config["allowed_raw_values"])
    allowed_class = set(config["allowed_class_values"])
    class_mapping = {
        int(raw_value): int(class_value)
        for raw_value, class_value in config["raw_to_class_mapping"].items()
    }
    known_annotation_only = set(config["known_annotation_only_frames"])
    expected_frame_numbers = range(1, int(config["frames_per_sequence"]) + 1)
    issues: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    sequence_hash_rows: list[dict[str, Any]] = []
    all_hash_members: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sequence_combined_hashes: list[tuple[str, str]] = []
    global_values = {"raw_mask": set(), "class_mask": set()}
    check_counts: dict[str, int] = defaultdict(int)

    def record_failure(check: str, sequence: str, frame_id: str, detail: str) -> None:
        issues.append(
            {
                "severity": "high",
                "check": check,
                "sequence": sequence,
                "frame_id": frame_id,
                "detail": detail,
            }
        )

    for sequence in sequences:
        indexes = {
            role: index_masks(dataset_root / sequence / directory)
            for role, directory in ROLE_DIRS.items()
        }
        expected_ids = {
            f"{sequence}_{frame_number:04d}" for frame_number in expected_frame_numbers
        }
        expected_image_ids = expected_ids - {
            frame_id for frame_id in known_annotation_only if frame_id.startswith(sequence)
        }
        expected_ids_by_role = {
            "image": expected_image_ids,
            "raw_mask": expected_ids,
            "instance_mask": expected_ids,
            "class_mask": expected_ids,
        }
        role_hashes: list[tuple[str, str]] = []
        for role, index in indexes.items():
            observed_ids = set(index)
            check_counts["frame_set_checks"] += 1
            if observed_ids != expected_ids_by_role[role]:
                record_failure(
                    "frame_set",
                    sequence,
                    "",
                    f"{role}: missing={sorted(expected_ids_by_role[role] - observed_ids)}, "
                    f"extra={sorted(observed_ids - expected_ids_by_role[role])}",
                )
            members = []
            for frame_id, path in sorted(index.items()):
                digest = sha256_file(path)
                members.append((frame_id, digest))
                all_hash_members[role][digest].append((sequence, frame_id))
                array = np.asarray(Image.open(path))
                check_counts["file_shape_checks"] += 1
                if array.shape != expected_shape:
                    record_failure(
                        "shape",
                        sequence,
                        frame_id,
                        f"{role}: expected {expected_shape}, observed {array.shape}",
                    )
                check_counts["file_dtype_checks"] += 1
                if str(array.dtype) != expected_dtypes[role]:
                    record_failure(
                        "dtype",
                        sequence,
                        frame_id,
                        f"{role}: expected {expected_dtypes[role]}, observed {array.dtype}",
                    )
            role_digest = stable_membership_hash(members)
            role_hashes.append((role, role_digest))
            sequence_hash_rows.append(
                {
                    "sequence": sequence,
                    "role": role,
                    "file_count": len(index),
                    "membership_sha256": role_digest,
                }
            )

        combined_digest = stable_membership_hash(role_hashes)
        sequence_combined_hashes.append((sequence, combined_digest))
        sequence_hash_rows.append(
            {
                "sequence": sequence,
                "role": "all_roles",
                "file_count": sum(len(index) for index in indexes.values()),
                "membership_sha256": combined_digest,
            }
        )

        mask_frame_ids = sorted(
            set(indexes["raw_mask"])
            | set(indexes["instance_mask"])
            | set(indexes["class_mask"])
        )
        aligned_instances = 0
        annotated_instances = 0
        for frame_id in mask_frame_ids:
            if not all(frame_id in indexes[role] for role in ("raw_mask", "instance_mask", "class_mask")):
                record_failure("mask_alignment", sequence, frame_id, "one or more mask roles missing")
                continue
            raw_mask = np.asarray(Image.open(indexes["raw_mask"][frame_id]))
            instance_mask = np.asarray(Image.open(indexes["instance_mask"][frame_id]))
            class_mask = np.asarray(Image.open(indexes["class_mask"][frame_id]))
            raw_values = set(np.unique(raw_mask).astype(int))
            class_values = set(np.unique(class_mask).astype(int))
            global_values["raw_mask"].update(raw_values)
            global_values["class_mask"].update(class_values)
            validations = {
                "raw_values": bool(raw_values) and raw_values <= allowed_raw,
                "class_values": bool(class_values) and class_values <= allowed_class,
                "instance_ids_contiguous": contiguous_instance_ids(instance_mask),
                "instance_ids_connected": connected_instance_ids(instance_mask),
                "raw_class_mapping": raw_class_mapping_valid(raw_mask, class_mask, class_mapping),
                "instance_class_mapping": instance_class_mapping_valid(
                    instance_mask, class_mask, allowed_class - {0}
                ),
            }
            for check, passed in validations.items():
                check_counts[f"{check}_checks"] += 1
                if not passed:
                    record_failure(check, sequence, frame_id, "check failed")
            instance_count = int(len(np.unique(instance_mask)) - 1)
            annotated_instances += instance_count
            if frame_id in indexes["image"]:
                aligned_instances += instance_count

        annotation_only_observed = set(indexes["raw_mask"]) - set(indexes["image"])
        annotation_only_expected = {
            frame_id for frame_id in known_annotation_only if frame_id.startswith(sequence)
        }
        check_counts["annotation_only_set_checks"] += 1
        if annotation_only_observed != annotation_only_expected:
            record_failure(
                "annotation_only_set",
                sequence,
                "",
                f"expected={sorted(annotation_only_expected)}, observed={sorted(annotation_only_observed)}",
            )
        aligned_ids = set.intersection(*(set(index) for index in indexes.values()))
        sequence_rows.append(
            {
                "sequence": sequence,
                "role": config["sequence_roles"][sequence],
                "images": len(indexes["image"]),
                "raw_masks": len(indexes["raw_mask"]),
                "instance_masks": len(indexes["instance_mask"]),
                "class_masks": len(indexes["class_mask"]),
                "aligned_frames": len(aligned_ids),
                "annotation_only_masks": len(annotation_only_observed),
                "annotated_instances_all_masks": annotated_instances,
                "annotated_instances_aligned_frames": aligned_instances,
                "sequence_sha256": combined_digest,
            }
        )

    duplicate_rows = []
    for role, by_digest in sorted(all_hash_members.items()):
        for digest, members in sorted(by_digest.items()):
            if len(members) <= 1:
                continue
            duplicate_rows.append(
                {
                    "role": role,
                    "sha256": digest,
                    "member_count": len(members),
                    "sequence_count": len({sequence for sequence, _ in members}),
                    "members": ";".join(
                        f"{sequence}:{frame_id}" for sequence, frame_id in sorted(members)
                    ),
                }
            )

    prior_manifest_path = prior_dir / "dataset_manifest.json"
    prior_inventory_path = prior_dir / "dataset_inventory.csv"
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    prior_inventory_hash_matches = (
        sha256_file(prior_inventory_path) == prior_manifest["dataset_inventory_sha256"]
    )
    observed_total_aligned = int(sum(row["aligned_frames"] for row in sequence_rows))
    observed_training_aligned = int(
        sum(
            row["aligned_frames"]
            for row in sequence_rows
            if row["role"] == "classifier_training"
        )
    )
    prior_reconciliation = {
        "inventory_hash_matches_manifest": prior_inventory_hash_matches,
        "total_aligned_matches": observed_total_aligned == prior_manifest["total_aligned_images"],
        "counts_match": all(
            row["images"] == prior_manifest["counts"][row["sequence"]]["image"]
            and row["raw_masks"] == prior_manifest["counts"][row["sequence"]]["raw_mask"]
            and row["instance_masks"]
            == prior_manifest["counts"][row["sequence"]]["instance_mask"]
            and row["class_masks"] == prior_manifest["counts"][row["sequence"]]["class_mask"]
            and row["aligned_frames"]
            == prior_manifest["counts"][row["sequence"]]["aligned_core"]
            for row in sequence_rows
        ),
    }

    macros = parse_latex_macros(macros_path)
    expected_thesis_values = {
        "TotalAlignedFrames": str(observed_total_aligned),
        "TrainingAlignedFrames": str(observed_training_aligned),
        "ValidationFrames": str(
            next(row["aligned_frames"] for row in sequence_rows if row["sequence"] == "MI04")
        ),
        "TestFrames": str(
            next(row["aligned_frames"] for row in sequence_rows if row["sequence"] == "MI05")
        ),
        "MIOneImages": str(
            next(row["images"] for row in sequence_rows if row["sequence"] == "MI01")
        ),
        "MIOneAnnotationFiles": str(
            next(row["raw_masks"] for row in sequence_rows if row["sequence"] == "MI01")
        ),
    }
    thesis_reconciliation = {
        key: {
            "expected": expected,
            "observed_macro": macros.get(key),
            "matches": macros.get(key) == expected,
        }
        for key, expected in expected_thesis_values.items()
    }

    exact_value_unions = {
        "raw_mask": sorted(global_values["raw_mask"]),
        "class_mask": sorted(global_values["class_mask"]),
    }
    validations = {
        "no_quality_issues": not issues,
        "no_exact_duplicate_files_within_or_across_sequences": not duplicate_rows,
        "raw_value_union_exact": global_values["raw_mask"] == allowed_raw,
        "class_value_union_exact": global_values["class_mask"] == allowed_class,
        "known_annotation_only_exact": all(
            (
                set(index_masks(dataset_root / sequence / ROLE_DIRS["raw_mask"]))
                - set(index_masks(dataset_root / sequence / ROLE_DIRS["image"]))
            )
            == {frame for frame in known_annotation_only if frame.startswith(sequence)}
            for sequence in sequences
        ),
        "prior_audit_reconciled": all(prior_reconciliation.values()),
        "thesis_numbers_reconciled": all(
            item["matches"] for item in thesis_reconciliation.values()
        ),
    }
    dataset_digest = stable_membership_hash(sequence_combined_hashes)

    sequence_summary_path = output_dir / "sequence_summary.csv"
    sequence_hashes_path = output_dir / "sequence_hashes.csv"
    duplicate_groups_path = output_dir / "duplicate_groups.csv"
    quality_issues_path = output_dir / "quality_issues.csv"
    summary_path = output_dir / "quality_summary.json"
    pd.DataFrame(sequence_rows).to_csv(
        sequence_summary_path, index=False, lineterminator="\n"
    )
    pd.DataFrame(sequence_hash_rows).to_csv(
        sequence_hashes_path, index=False, lineterminator="\n"
    )
    pd.DataFrame(
        duplicate_rows,
        columns=["role", "sha256", "member_count", "sequence_count", "members"],
    ).to_csv(duplicate_groups_path, index=False, lineterminator="\n")
    pd.DataFrame(
        issues,
        columns=["severity", "check", "sequence", "frame_id", "detail"],
    ).to_csv(quality_issues_path, index=False, lineterminator="\n")
    summary = {
        "schema_version": "1.0",
        "status": "pass" if all(validations.values()) else "fail",
        "dataset_root": str(dataset_root),
        "unit_of_observation": {
            "image_level": "aligned frame",
            "object_level": "annotated or predicted instance nested within a frame",
            "split_level": "complete time-lapse sequence",
        },
        "totals": {
            "images": int(sum(row["images"] for row in sequence_rows)),
            "raw_masks": int(sum(row["raw_masks"] for row in sequence_rows)),
            "instance_masks": int(sum(row["instance_masks"] for row in sequence_rows)),
            "class_masks": int(sum(row["class_masks"] for row in sequence_rows)),
            "aligned_frames": observed_total_aligned,
            "training_aligned_frames": observed_training_aligned,
        },
        "representation": {
            "shape": list(expected_shape),
            "dtypes": expected_dtypes,
            "value_unions": exact_value_unions,
            "raw_to_class_mapping": {
                str(key): value for key, value in sorted(class_mapping.items())
            },
        },
        "known_annotation_only_frames": sorted(known_annotation_only),
        "sequence_roles": config["sequence_roles"],
        "dataset_membership_sha256": dataset_digest,
        "duplicate_groups": len(duplicate_rows),
        "quality_issue_count": len(issues),
        "check_counts": dict(sorted(check_counts.items())),
        "prior_audit_reconciliation": prior_reconciliation,
        "thesis_reconciliation": thesis_reconciliation,
        "validations": validations,
        "interpretation": {
            "temporal_dependence": "frames within a sequence are repeated observations of one field of view and are not independent",
            "random_frame_split": "prohibited because adjacent frames can place nearly repeated cells in development and evaluation subsets",
            "inclusion": "a frame is included in image-dependent analysis only when image, raw mask, instance mask, and class mask share the same frame identifier",
            "exclusion": "MI01_0001 and MI01_0062 are annotation-only and excluded from image-dependent analysis",
        },
    }
    write_json(summary_path, summary)

    output_paths = (
        sequence_summary_path,
        sequence_hashes_path,
        duplicate_groups_path,
        quality_issues_path,
        summary_path,
    )
    manifest = {
        "schema_version": "1.0",
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (prior_manifest_path, prior_inventory_path, macros_path)
        ],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
            "scipy": scipy.__version__,
        },
        "command": " ".join([str(Path(sys.executable).resolve()), *sys.argv]),
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in output_paths
        ],
        "dataset_membership_sha256": dataset_digest,
    }
    write_json(output_dir / "quality_manifest.json", manifest)
    if summary["status"] != "pass":
        raise SystemExit("Dataset quality audit failed; inspect quality_summary.json")


if __name__ == "__main__":
    main()
