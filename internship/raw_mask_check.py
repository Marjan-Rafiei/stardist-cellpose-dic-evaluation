"""
raw_mask_check.py — Diagnostic: inspect the RAW original mask files
(before any conversion), and report what pixel values actually exist.

This tells us if the raw masks even CONTAIN mitotic (white/255) pixels
at all, which would explain why 0 mitotic cells were found downstream.

Usage:
    python raw_mask_check.py
"""

import os
import numpy as np
from skimage import io
from config import BASE, SEQUENCES, MASKS_FOLDER

print("=" * 55)
print("  DIAGNOSTIC: Raw mask pixel value inspector")
print("=" * 55)

for seq in SEQUENCES:
    mask_folder = os.path.join(BASE, seq, MASKS_FOLDER)

    if not os.path.exists(mask_folder):
        print(f"\n  {seq}: FOLDER NOT FOUND at {mask_folder}")
        continue

    files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

    if not files:
        print(f"\n  {seq}: no .png files found in {mask_folder}")
        continue

    print(f"\n  {seq}: checking {len(files)} mask files...")

    # Check every file's unique raw pixel values (not just the first one)
    all_unique_vals = set()
    max_val_overall = 0

    for fname in files:
        path = os.path.join(mask_folder, fname)

        # Read WITHOUT as_gray, to see the raw untouched pixel data
        raw = io.imread(path)
        unique_vals = np.unique(raw)
        all_unique_vals.update(unique_vals.tolist())
        if raw.max() > max_val_overall:
            max_val_overall = raw.max()

    print(f"    Image shape (last file): {raw.shape}, dtype: {raw.dtype}")
    print(f"    All unique pixel values found across all {len(files)} files:")
    print(f"    {sorted(all_unique_vals)}")
    print(f"    Maximum pixel value found: {max_val_overall}")

print("\n" + "=" * 55)
print("  Diagnostic complete!")
print("=" * 55)
