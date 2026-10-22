"""Plot the verified mask-conversion mechanism and dataset-wide audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from skimage import io

from code.mask_conversion_audit import faulty_unconditional_uint8, prepare_mask_uint8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    audit_dir = Path(args.audit_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = audit_dir / "conversion_summary.json"
    mapping_path = audit_dir / "value_mapping.csv"
    comparison_path = audit_dir / "mask_comparison.csv"
    analysis_manifest_path = audit_dir / "analysis_manifest.json"
    inputs = (summary_path, mapping_path, comparison_path, analysis_manifest_path)
    summary = read_json(summary_path)
    expected = {item["path"]: item["sha256"] for item in read_json(analysis_manifest_path)["outputs"]}
    for path in (comparison_path, mapping_path, summary_path):
        if expected.get(str(path)) != sha256_file(path):
            raise ValueError(f"Audit output hash mismatch: {path}")

    with mapping_path.open("r", encoding="utf-8", newline="") as stream:
        mapping = list(csv.DictReader(stream))
    frame = summary["example_frame"]
    sequence = frame.split("_")[1]
    raw_path = dataset_root / sequence / "Masks" / f"{frame}.png"
    raw = io.imread(raw_path, as_gray=True)
    faulty = faulty_unconditional_uint8(raw)
    corrected = prepare_mask_uint8(raw)

    display_cmap = ListedColormap(["#252A34", "#D1495B", "#2A9D8F", "#F4A261"])
    display_norm = BoundaryNorm([-0.5, 1.5, 64, 200, 255.5], display_cmap.N)
    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.titlesize": 13})
    fig = plt.figure(figsize=(10.6, 7.0), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.25, 1.0],
        left=0.04,
        right=0.98,
        bottom=0.07,
        top=0.90,
        wspace=0.28,
        hspace=0.50,
    )
    titles = (
        "Raw mask read by scikit-image\nvalues: 0, 128, 255",
        "Reproduced unconditional scaling\nvalues: 0, 1, 128",
        "Corrected conditional handling\nvalues: 0, 128, 255",
    )
    for column, (array, title) in enumerate(zip((raw, faulty, corrected), titles)):
        axis = fig.add_subplot(grid[0, 2 * column : 2 * column + 2])
        axis.imshow(array, cmap=display_cmap, norm=display_norm, interpolation="nearest")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#5C677D")
            spine.set_linewidth(0.8)

    table_axis = fig.add_subplot(grid[1, :4])
    table_axis.axis("off")
    table_axis.set_title("Verified value mapping", loc="left", fontweight="bold", pad=8)
    cell_text = [
        [row["input_uint8"], row["faulty_unconditional_output"], row["corrected_output"]]
        for row in mapping
    ]
    table = table_axis.table(
        cellText=cell_text,
        colLabels=["Input uint8", "Unconditional x255", "Correct branch"],
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0.0, 0.28, 0.98, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#DCE6F1")
            cell.set_text_props(weight="bold")
        elif row_index == 3:
            cell.set_facecolor("#F8D7DA")
    table_axis.text(
        0.0,
        0.08,
        "uint8(255 x 255) = 1. The original value 255 therefore no longer\n"
        "passes the >200 mitotic threshold.",
        transform=table_axis.transAxes,
        fontsize=9,
        color="#8B1E2D",
        weight="bold",
    )

    fact_axis = fig.add_subplot(grid[1, 4:])
    fact_axis.axis("off")
    fact_axis.set_title("Dataset-wide verification", loc="left", fontweight="bold", pad=8)
    facts = (
        f"{summary['total_masks']} raw masks audited",
        f"{summary['masks_with_value_255']} masks contain value 255",
        f"{summary['total_value_255_pixels']:,} value-255 pixels",
        f"{summary['corrected_instance_exact_matches']}/{summary['total_masks']} instance masks match",
        f"{summary['corrected_class_exact_matches']}/{summary['total_masks']} class masks match",
        "Normalized and uint8 branches pass",
    )
    for index, fact in enumerate(facts):
        fact_axis.text(
            0.0,
            0.90 - index * 0.13,
            f"{index + 1}. {fact}",
            transform=fact_axis.transAxes,
            va="top",
            fontsize=9,
        )
    fact_axis.text(
        0.0,
        0.02,
        "Scope boundary: the fault mechanism and current outputs are verified.\n"
        "Use in a specific historical training run remains unresolved.",
        transform=fact_axis.transAxes,
        va="bottom",
        fontsize=8.2,
        color="#4A4A4A",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F2F2F2", "edgecolor": "#B0B0B0"},
    )

    fig.suptitle("Mask-conversion fault audit", fontweight="bold")
    pdf_path = output / "mask_conversion_audit.pdf"
    png_path = output / "mask_conversion_audit.png"
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Creator": "scripts_figures/plot_mask_conversion_audit.py", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(fig)

    manifest = {
        "schema_version": "1.0",
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in (*inputs, raw_path)
        ],
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in (pdf_path, png_path)
        ],
        "notes": "Discrete colors expose the overflowed value 1; counts are read from the complete 345-mask audit.",
    }
    write_json(output / "figure_manifest.json", manifest)


if __name__ == "__main__":
    main()
