"""Illustrate the historical uint8 mask-scaling failure without modifying data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from code.unified_evaluation import sha256_file, write_json
from scripts_figures.provenance import write_figure_manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    input_path = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(Image.open(input_path))
    if raw.dtype != np.uint8:
        raise ValueError(f"Expected uint8 raw mask, found {raw.dtype}")
    old = (raw * np.uint8(255)).astype(np.uint8)
    fixed = raw.copy()
    values = {
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "raw_values": np.unique(raw).astype(int).tolist(),
        "historical_unconditional_scaling_values": np.unique(old).astype(int).tolist(),
        "fixed_values": np.unique(fixed).astype(int).tolist(),
        "explanation": "uint8 multiplication by 255 wraps 255 to 1; the fixed path does not rescale masks already in 0..255",
    }
    write_json(output / "mask_conversion_values.json", values)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for axis, array, title in zip(
        axes,
        (raw, old, fixed),
        ("Raw ALFI mask", "Historical unconditional scaling", "Correct conditional conversion input"),
    ):
        axis.imshow(array, cmap="viridis", vmin=0, vmax=255)
        axis.set_title(f"{title}\nvalues={np.unique(array).astype(int).tolist()}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(output / "mask_conversion_bug.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "mask_conversion_bug.pdf", bbox_inches="tight")
    plt.close(fig)
    write_figure_manifest(
        output,
        Path(__file__),
        (input_path,),
        (
            output / "mask_conversion_values.json",
            output / "mask_conversion_bug.png",
            output / "mask_conversion_bug.pdf",
        ),
        "Historical scaling is reproduced in memory only; no source mask is modified.",
    )


if __name__ == "__main__":
    main()
