"""
convert_step_check.py — Diagnostic: replicate convert_masks.py's EXACT
reading method (as_gray=True, then *255) on one real mask file, and
compare the result to the raw pixel values.

This tells us whether the as_gray conversion is silently destroying
the mitotic (255) pixel values before thresholding happens.

Usage:
    python convert_step_check.py
"""

import os
import numpy as np
from skimage import io
from config import BASE, SEQUENCES, MASKS_FOLDER

print("=" * 55)
print("  DIAGNOSTIC: convert_masks.py reading method check")
print("=" * 55)

seq = SEQUENCES[0]
mask_folder = os.path.join(BASE, seq, MASKS_FOLDER)
files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

# Find a file that actually has a 255 pixel in it, using the RAW read
chosen_file = None
for fname in files:
    raw = io.imread(os.path.join(mask_folder, fname))
    if raw.max() == 255:
        chosen_file = fname
        break

if chosen_file is None:
    print("  Could not find any file with a raw 255 pixel in", seq)
else:
    path = os.path.join(mask_folder, chosen_file)
    print(f"\n  Testing file: {seq}/{chosen_file}")

    # Method 1: RAW read (no as_gray)
    raw = io.imread(path)
    print(f"\n  [RAW read]        dtype={raw.dtype}  unique values={np.unique(raw)}")
    print(f"                     pixels == 255: {(raw == 255).sum()}")

    # Method 2: EXACTLY replicate convert_masks.py
    mask = io.imread(path, as_gray=True)
    print(f"\n  [as_gray=True]    dtype={mask.dtype}  min={mask.min():.6f}  max={mask.max():.6f}")
    mask_scaled = (mask * 255).astype(np.uint8)
    print(f"  [after *255,uint8] dtype={mask_scaled.dtype}  unique values={np.unique(mask_scaled)}")
    print(f"                     pixels > 200: {(mask_scaled > 200).sum()}")
    print(f"                     pixels == 255: {(mask_scaled == 255).sum()}")

print("\n" + "=" * 55)
print("  Diagnostic complete!")
print("=" * 55)
