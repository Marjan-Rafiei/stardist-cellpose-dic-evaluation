"""Create the locked dataset inventory and enforce the aligned-frame contract."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from code.unified_evaluation import index_masks, read_mask, sha256_file, write_json


ROLE_DIRS = {
    "image": "Images",
    "raw_mask": "Masks",
    "instance_mask": "Masks_converted",
    "class_mask": "Masks_classes",
}


def image_info(path: Path) -> tuple[str, str]:
    array = np.asarray(Image.open(path))
    return "x".join(str(item) for item in array.shape), str(array.dtype)


def instance_class_valid(instance_mask: np.ndarray, class_mask: np.ndarray) -> bool:
    if instance_mask.shape != class_mask.shape:
        return False
    if not np.array_equal(instance_mask > 0, class_mask > 0):
        return False
    for instance_id in np.unique(instance_mask):
        if instance_id <= 0:
            continue
        values = np.unique(class_mask[instance_mask == instance_id])
        if len(values) != 1 or int(values[0]) not in (1, 2):
            return False
    return True


def contiguous_labels(mask: np.ndarray) -> bool:
    labels = np.unique(mask)
    labels = labels[labels > 0]
    return np.array_equal(labels, np.arange(1, len(labels) + 1))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset_root"]).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = tuple(config["sequences"])
    known_annotation_only = set(config["known_annotation_only_frames"])
    allowed_raw = set(config["allowed_raw_values"])
    allowed_class = set(config["allowed_class_values"])
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    hashes_by_role: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[str, dict[str, int]] = {}

    for sequence in sequences:
        indexes = {
            role: index_masks(dataset_root / sequence / directory)
            for role, directory in ROLE_DIRS.items()
        }
        all_keys = sorted(set().union(*(set(index) for index in indexes.values())))
        counts[sequence] = {role: len(index) for role, index in indexes.items()}
        counts[sequence]["aligned_core"] = len(set.intersection(*(set(index) for index in indexes.values())))

        for frame_id in all_keys:
            row: dict[str, Any] = {"sequence": sequence, "frame_id": frame_id}
            arrays: dict[str, np.ndarray] = {}
            for role, index in indexes.items():
                path = index.get(frame_id)
                row[f"{role}_exists"] = int(path is not None)
                row[f"{role}_path"] = str(path) if path else ""
                if path is None:
                    row[f"{role}_sha256"] = ""
                    row[f"{role}_shape"] = ""
                    row[f"{role}_dtype"] = ""
                    continue
                digest = sha256_file(path)
                shape, dtype = image_info(path)
                row[f"{role}_sha256"] = digest
                row[f"{role}_shape"] = shape
                row[f"{role}_dtype"] = dtype
                hashes_by_role[role][digest].append((sequence, frame_id))
                if role != "image":
                    arrays[role] = read_mask(path)

            row["core_aligned"] = int(all(row[f"{role}_exists"] for role in ROLE_DIRS))
            existing_shapes = {
                row[f"{role}_shape"] for role in ROLE_DIRS if row[f"{role}_exists"]
            }
            row["core_shapes_match"] = int(row["core_aligned"] and len(existing_shapes) == 1)
            row["known_annotation_only"] = int(frame_id in known_annotation_only)
            raw_values = set(np.unique(arrays["raw_mask"]).astype(int)) if "raw_mask" in arrays else set()
            class_values = set(np.unique(arrays["class_mask"]).astype(int)) if "class_mask" in arrays else set()
            row["raw_values"] = ";".join(map(str, sorted(raw_values)))
            row["class_values"] = ";".join(map(str, sorted(class_values)))
            row["raw_values_valid"] = int(bool(raw_values) and raw_values <= allowed_raw)
            row["class_values_valid"] = int(bool(class_values) and class_values <= allowed_class)
            row["instance_labels_contiguous"] = int(
                "instance_mask" in arrays and contiguous_labels(arrays["instance_mask"])
            )
            row["instance_class_mapping_valid"] = int(
                "instance_mask" in arrays
                and "class_mask" in arrays
                and instance_class_valid(arrays["instance_mask"], arrays["class_mask"])
            )
            rows.append(row)

            missing_roles = [role for role in ROLE_DIRS if not row[f"{role}_exists"]]
            if missing_roles:
                expected = frame_id in known_annotation_only and missing_roles == ["image"]
                issues.append(
                    {
                        "severity": "known" if expected else "high",
                        "issue_type": "missing_role",
                        "sequence": sequence,
                        "frame_id": frame_id,
                        "detail": ",".join(missing_roles),
                    }
                )
            for check in (
                "raw_values_valid",
                "class_values_valid",
                "instance_labels_contiguous",
                "instance_class_mapping_valid",
                "core_shapes_match",
            ):
                if check == "core_shapes_match" and not row["core_aligned"]:
                    continue
                if not row[check]:
                    issues.append(
                        {
                            "severity": "high",
                            "issue_type": check,
                            "sequence": sequence,
                            "frame_id": frame_id,
                            "detail": "check failed",
                        }
                    )

    for role, digest_map in hashes_by_role.items():
        for digest, members in digest_map.items():
            member_sequences = {sequence for sequence, _ in members}
            if len(member_sequences) > 1:
                issues.append(
                    {
                        "severity": "high",
                        "issue_type": "cross_sequence_duplicate",
                        "sequence": ";".join(sorted(member_sequences)),
                        "frame_id": ";".join(frame for _, frame in members),
                        "detail": f"{role}:{digest}",
                    }
                )

    for sequence in sequences:
        expected_images = int(config["expected_image_counts"][sequence])
        expected_masks = int(config["expected_mask_counts"][sequence])
        observed = counts[sequence]
        expected_by_role = {
            "image": expected_images,
            "raw_mask": expected_masks,
            "instance_mask": expected_masks,
            "class_mask": expected_masks,
        }
        for role, expected in expected_by_role.items():
            if observed[role] != expected:
                issues.append(
                    {
                        "severity": "high",
                        "issue_type": "unexpected_count",
                        "sequence": sequence,
                        "frame_id": "",
                        "detail": f"{role}: expected {expected}, observed {observed[role]}",
                    }
                )

    inventory = pd.DataFrame(rows).sort_values(["sequence", "frame_id"], kind="stable")
    issue_columns = ["severity", "issue_type", "sequence", "frame_id", "detail"]
    issue_frame = pd.DataFrame.from_records(issues, columns=issue_columns).sort_values(
        issue_columns, kind="stable"
    )
    inventory.to_csv(output_dir / "dataset_inventory.csv", index=False, lineterminator="\n")
    issue_frame.to_csv(output_dir / "alignment_errors.csv", index=False, lineterminator="\n")

    blocking = issue_frame[issue_frame["severity"] == "high"]
    manifest = {
        "schema_version": "1.0",
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "dataset_root": str(dataset_root),
        "sequences": list(sequences),
        "counts": counts,
        "total_aligned_images": int(sum(counts[item]["aligned_core"] for item in sequences)),
        "known_annotation_only_frames": sorted(known_annotation_only),
        "blocking_issue_count": int(len(blocking)),
        "known_issue_count": int((issue_frame["severity"] == "known").sum()),
        "status": "pass_with_known_annotation_only_frames" if blocking.empty else "fail",
        "dataset_inventory_sha256": sha256_file(output_dir / "dataset_inventory.csv"),
        "alignment_errors_sha256": sha256_file(output_dir / "alignment_errors.csv"),
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    if not blocking.empty:
        raise SystemExit(f"Dataset audit failed with {len(blocking)} blocking issue(s)")


if __name__ == "__main__":
    main()
