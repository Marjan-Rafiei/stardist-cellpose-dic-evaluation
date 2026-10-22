"""
train_improved.py — Step 2 (IMPROVED): Train StarDist2D with fixes for
the mitotic-cell class imbalance problem.

Two improvements over the original train.py:
  1. Data augmentation (random flips/rotations + intensity jitter) —
     helps the model generalize better from limited data.
  2. Mitotic oversampling — training images that contain at least one
     mitotic cell are duplicated several times in the training set,
     so the model sees them more often (since mitotic cells are much
     rarer than interphase cells in the raw data).

Usage:
    python train_improved.py
"""

import os
import numpy as np
from skimage import io
from stardist.models import StarDist2D, Config2D
from csbdeep.utils import normalize
from config import (BASE, TRAIN_SEQS, VAL_SEQS,
                    IMAGES_FOLDER, MASKS_CONV_FOLDER, MASKS_CLS_FOLDER,
                    MODEL_NAME, MODELS_DIR,
                    N_RAYS, GRID, TRAIN_PATCH_SIZE,
                    TRAIN_BATCH_SIZE, TRAIN_EPOCHS,
                    TRAIN_STEPS, LEARNING_RATE,
                    MITOTIC_LABEL)

# How many extra times to duplicate images containing mitotic cells.
# 3 means: an image with mitotic cells appears 4x total in training
# (1 original + 3 duplicates), an image with only interphase appears once.
MITOTIC_OVERSAMPLE_FACTOR = 3


def load_data(sequences, oversample_mitotic=False):
    images, masks = [], []
    n_mitotic_frames = 0

    for seq in sequences:
        img_folder = os.path.join(BASE, seq, IMAGES_FOLDER)
        msk_folder = os.path.join(BASE, seq, MASKS_CONV_FOLDER)
        cls_folder = os.path.join(BASE, seq, MASKS_CLS_FOLDER)

        if not os.path.exists(msk_folder):
            print(f"  WARNING: Run convert_masks.py first! ({msk_folder} missing)")
            continue

        for fname in sorted(os.listdir(img_folder)):
            if not fname.endswith('.png'):
                continue
            mask_name = fname.replace('I_', 'M_')
            msk_path  = os.path.join(msk_folder, mask_name)
            cls_path  = os.path.join(cls_folder,  mask_name)
            if not os.path.exists(msk_path):
                continue

            img = io.imread(os.path.join(img_folder, fname), as_gray=True).astype(np.float32)
            msk = io.imread(msk_path).astype(np.int32)
            img = normalize(img, 1, 99.8, axis=(0, 1))

            has_mitotic = False
            if oversample_mitotic and os.path.exists(cls_path):
                cls_mask = io.imread(cls_path).astype(np.uint8)
                has_mitotic = np.any(cls_mask == MITOTIC_LABEL)

            images.append(img)
            masks.append(msk)

            if has_mitotic:
                n_mitotic_frames += 1
                # Duplicate this image+mask several extra times
                for _ in range(MITOTIC_OVERSAMPLE_FACTOR):
                    images.append(img)
                    masks.append(msk)

        print(f"  Loaded {seq}: {len(os.listdir(img_folder))} frames")

    if oversample_mitotic:
        print(f"  Frames with mitotic cells: {n_mitotic_frames} "
              f"(each duplicated {MITOTIC_OVERSAMPLE_FACTOR}x extra)")
        print(f"  Total training samples after oversampling: {len(images)}")

    return images, masks


def random_fliprot(img, mask):
    """Randomly flip and rotate an image+mask pair together (90-degree steps)."""
    axes = (0, 1)
    for ax in axes:
        if np.random.rand() > 0.5:
            img  = np.flip(img,  axis=ax)
            mask = np.flip(mask, axis=ax)
    n_rot = np.random.randint(0, 4)
    img  = np.rot90(img,  n_rot, axes=axes)
    mask = np.rot90(mask, n_rot, axes=axes)
    return img, mask


def random_intensity_change(img):
    """Slightly randomize brightness/contrast to make the model more robust."""
    img = img * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05)
    return img


def augmenter(x, y):
    """StarDist calls this on every training example, every epoch."""
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    return x, y


print("=" * 55)
print("  STEP 2 (IMPROVED): Training StarDist2D model")
print("  [with augmentation + mitotic oversampling]")
print("=" * 55)

print("\nLoading training data...")
X_train, Y_train = load_data(TRAIN_SEQS, oversample_mitotic=True)

print("\nLoading validation data...")
# NOTE: validation data is NOT oversampled — we want to evaluate on
# realistic, unmodified data.
X_val, Y_val = load_data(VAL_SEQS, oversample_mitotic=False)

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
model.train(X_train, Y_train, validation_data=(X_val, Y_val), augmenter=augmenter)

print("\nOptimizing thresholds on validation set...")
model.optimize_thresholds(X_val, Y_val)

print("=" * 55)
print("  Training complete!")
print(f"  Model saved to: models/{MODEL_NAME}/")
print("=" * 55)
