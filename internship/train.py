"""
train.py — Step 2: Train StarDist2D model.

Trains on TRAIN_SEQS, validates on VAL_SEQS.
Uses the instance masks from Masks_converted/.

Usage:
    python train.py
"""

import os
import numpy as np
from skimage import io
from stardist.models import StarDist2D, Config2D
from csbdeep.utils import normalize
from config import (BASE, TRAIN_SEQS, VAL_SEQS,
                    IMAGES_FOLDER, MASKS_CONV_FOLDER,
                    MODEL_NAME, MODELS_DIR,
                    N_RAYS, GRID, TRAIN_PATCH_SIZE,
                    TRAIN_BATCH_SIZE, TRAIN_EPOCHS,
                    TRAIN_STEPS, LEARNING_RATE)

def load_data(sequences):
    images, masks = [], []
    for seq in sequences:
        img_folder = os.path.join(BASE, seq, IMAGES_FOLDER)
        msk_folder = os.path.join(BASE, seq, MASKS_CONV_FOLDER)

        if not os.path.exists(msk_folder):
            print(f"  WARNING: Run convert_masks.py first! ({msk_folder} missing)")
            continue

        for fname in sorted(os.listdir(img_folder)):
            if not fname.endswith('.png'):
                continue
            mask_name = fname.replace('I_', 'M_')
            msk_path  = os.path.join(msk_folder, mask_name)
            if not os.path.exists(msk_path):
                continue

            img = io.imread(os.path.join(img_folder, fname), as_gray=True).astype(np.float32)
            msk = io.imread(msk_path).astype(np.int32)
            img = normalize(img, 1, 99.8, axis=(0, 1))

            images.append(img)
            masks.append(msk)

        print(f"  Loaded {seq}: {len(os.listdir(img_folder))} frames")
    return images, masks

print("=" * 55)
print("  STEP 2: Training StarDist2D model")
print("=" * 55)

print("\nLoading training data...")
X_train, Y_train = load_data(TRAIN_SEQS)

print("\nLoading validation data...")
X_val, Y_val = load_data(VAL_SEQS)

print(f"\n  Training images:   {len(X_train)}")
print(f"  Validation images: {len(X_val)}")

conf = Config2D(
    n_rays                = N_RAYS,
    grid                  = GRID,
    n_channel_in          = 1,
    train_patch_size      = TRAIN_PATCH_SIZE,
    train_batch_size      = TRAIN_BATCH_SIZE,
    train_epochs          = TRAIN_EPOCHS,
    train_steps_per_epoch = TRAIN_STEPS,
    train_learning_rate   = LEARNING_RATE,
)

os.makedirs(MODELS_DIR, exist_ok=True)
model = StarDist2D(conf, name=MODEL_NAME, basedir=MODELS_DIR)

print("\nTraining started...")
model.train(X_train, Y_train, validation_data=(X_val, Y_val))

print("\nOptimizing thresholds on validation set...")
model.optimize_thresholds(X_val, Y_val)

print("=" * 55)
print("  Training complete!")
print(f"  Model saved to: models/{MODEL_NAME}/")
print("=" * 55)
