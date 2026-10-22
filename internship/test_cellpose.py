import os
import numpy as np
from cellpose import models
from skimage.io import imread

# quick sanity check before running this on the whole dataset
BASE = os.path.dirname(os.path.abspath(__file__))
test_img_path = os.path.join(BASE, "MI01", "Images", "I_MI01_0043.png")

img = imread(test_img_path)
print("image shape:", img.shape, img.dtype)

model = models.CellposeModel(gpu=False)

try:
    masks, flows, styles = model.eval(img)
except TypeError:
    # older cellpose versions want these explicitly
    masks, flows, styles = model.eval(img, diameter=None, channels=[0, 0])

print("cells found:", masks.max())
print("mask shape/dtype:", masks.shape, masks.dtype)
print(np.unique(masks)[:10])
