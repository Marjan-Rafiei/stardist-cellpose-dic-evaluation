"""Generate deterministic source, model, project, and environment manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "backups" / "2026-09-18_initial"
PROVENANCE = ROOT / "provenance"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_manifest() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in BACKUP.rglob("*") if item.is_file()):
        rows.append(
            {
                "relative_path": path.relative_to(BACKUP).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(PROVENANCE / "preserved_source_manifest.csv", ["relative_path", "bytes", "sha256"], rows)


def model_comparison_rows(internship_root: Path) -> list[dict[str, object]]:
    active = internship_root / "models" / "mitosis_model"
    backup = internship_root / "models" / "mitosis_model_baseline_backup"
    if not active.is_dir() or not backup.is_dir():
        raise FileNotFoundError(f"Expected both StarDist model directories under {internship_root / 'models'}")
    rows: list[dict[str, object]] = []
    relative_paths = sorted(
        {path.relative_to(active) for path in active.rglob("*") if path.is_file()}
        | {path.relative_to(backup) for path in backup.rglob("*") if path.is_file()}
    )
    for relative in relative_paths:
        left = active / relative
        right = backup / relative
        left_hash = sha256(left) if left.exists() else "MISSING"
        right_hash = sha256(right) if right.exists() else "MISSING"
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "active_sha256": left_hash,
                "backup_sha256": right_hash,
                "byte_identical": str(left_hash == right_hash).lower(),
            }
        )
    return rows


def model_comparison(internship_root: Path) -> None:
    rows = model_comparison_rows(internship_root)
    write_csv(
        PROVENANCE / "model_pair_hash_comparison.csv",
        ["relative_path", "active_sha256", "backup_sha256", "byte_identical"],
        rows,
    )


def project_manifest() -> None:
    excluded = {"backups", ".git", "results"}
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in ROOT.rglob("*") if item.is_file()):
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] in excluded:
            continue
        if relative.parts[:2] == ("thesis", "build") or "__pycache__" in relative.parts:
            continue
        if relative.as_posix() == "provenance/project_file_manifest.csv":
            continue
        if path.name.endswith((".aux", ".log", ".bcf", ".run.xml", ".blg", ".bbl")):
            continue
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(PROVENANCE / "project_file_manifest.csv", ["relative_path", "bytes", "sha256"], rows)


def environment_inventory() -> None:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "scikit-image", "scikit-learn", "Pillow", "joblib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    payload = {
        "generated_by": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_repository": (ROOT / ".git").exists(),
        "tectonic": _command_version(["tectonic", "--version"]),
    }
    (PROVENANCE / "environment_inventory.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unavailable on PATH"
    return (result.stdout or result.stderr).strip()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internship-root",
        type=Path,
        default=ROOT.parent / "internship",
        help="Internship evidence directory (default: sibling 'internship' directory)",
    )
    args = parser.parse_args(argv)
    internship_root = args.internship_root.expanduser().resolve()
    if not internship_root.is_dir():
        parser.error(f"Internship directory does not exist: {internship_root}")
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    source_manifest()
    model_comparison(internship_root)
    environment_inventory()
    project_manifest()


if __name__ == "__main__":
    main()
