"""
convert_masks.py — Step 1: Convert masks preserving interphase/mitotic classes.

FIXED VERSION: handles the case where io.imread(as_gray=True) returns
the image ALREADY in 0-255 range (instead of normalized 0-1), which
was silently corrupting mitotic (255) pixels via integer overflow.

Original masks:
  Black  (intensity=0)    → background
  Gray   (intensity~128)  → interphase cells
  White  (intensity=255)  → mitotic cells

Creates two output folders per sequence:
  Masks_converted/  → instance masks (each cell = unique integer) for StarDist
  Masks_classes/    → class map (1=interphase, 2=mitotic) for evaluation

Usage:
    python convert_masks.py
"""

import os
import numpy as np
from skimage import io, measure
from config import (BASE, SEQUENCES, MASKS_FOLDER,
                    MASKS_CONV_FOLDER, MASKS_CLS_FOLDER,
                    INTERPHASE_LABEL, MITOTIC_LABEL)

print("=" * 55)
print("  STEP 1: Converting masks (interphase + mitotic)")
print("  [FIXED: robust to as_gray scaling differences]")
print("=" * 55)

total = 0

for seq in SEQUENCES:
    mask_folder  = os.path.join(BASE, seq, MASKS_FOLDER)
    inst_folder  = os.path.join(BASE, seq, MASKS_CONV_FOLDER)
    cls_folder   = os.path.join(BASE, seq, MASKS_CLS_FOLDER)

    os.makedirs(inst_folder, exist_ok=True)
    os.makedirs(cls_folder,  exist_ok=True)

    files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

    for fname in files:
        # Load mask
        mask = io.imread(os.path.join(mask_folder, fname), as_gray=True)

        # FIX: only rescale to 0-255 if the values are actually normalized
        # (i.e. in the 0-1 range). If they're already 0-255, leave as is.
        # This avoids the integer-overflow bug that silently destroyed
        # mitotic (255) pixels on some skimage/imageio versions.
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # Separate interphase (gray) and mitotic (white) pixels
        interphase_px = (mask >= 50) & (mask <= 200)
        mitotic_px    = mask > 200

        # Label each class — each blob gets unique integer
        labeled_inter = measure.label(interphase_px, connectivity=2)
        labeled_mito  = measure.label(mitotic_px,    connectivity=2)

        # Offset mitotic IDs to avoid overlap with interphase IDs
        max_inter = labeled_inter.max()
        labeled_mito_offset = labeled_mito.copy()
        labeled_mito_offset[labeled_mito > 0] += max_inter

        # Combined instance mask — every cell has unique ID
        instance_mask = labeled_inter + labeled_mito_offset

        # Class mask — tells us which instance is which class
        class_mask = np.zeros_like(instance_mask, dtype=np.uint8)
        class_mask[interphase_px] = INTERPHASE_LABEL  # 1
        class_mask[mitotic_px]    = MITOTIC_LABEL     # 2

        # Save both masks
        io.imsave(os.path.join(inst_folder, fname), instance_mask.astype(np.int32), check_contrast=False)
        io.imsave(os.path.join(cls_folder,  fname), class_mask,                     check_contrast=False)

        n_inter = labeled_inter.max()
        n_mito  = labeled_mito.max()
        print(f"  {seq}/{fname} → {n_inter} interphase + {n_mito} mitotic")
        total += 1

print("=" * 55)
print(f"  Done! Converted {total} masks total.")
print("=" * 55)
