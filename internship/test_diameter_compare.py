import os
import glob
from cellpose import models
from skimage.io import imread
from skimage.measure import regionprops

from config import BASE, MASKS_CONV_FOLDER

# checking if manually setting diameter=38.5 actually helps vs
# letting cellpose guess it itself

img_dir = os.path.join(BASE, "MI01", "Images")
conv_dir = os.path.join(BASE, "MI01", MASKS_CONV_FOLDER)
img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:5]

model = models.Cellpose(model_type="cyto3", gpu=False)

print(f"{'image':<20} {'gt':<8} {'auto':<8} {'manual(38.5)':<12}")

for p in img_paths:
    fname = os.path.basename(p)
    frame_id = fname.split("_")[-1]
    gt_path = os.path.join(conv_dir, f"M_MI01_{frame_id}")

    img = imread(p)
    n_gt = 0
    if os.path.exists(gt_path):
        n_gt = len(regionprops(imread(gt_path)))

    masks_auto, _, _, _ = model.eval(img, diameter=None, channels=[0, 0])
    masks_manual, _, _, _ = model.eval(img, diameter=38.5, channels=[0, 0])

    print(f"{fname:<20} {n_gt:<8} {masks_auto.max():<8} {masks_manual.max():<12}")
