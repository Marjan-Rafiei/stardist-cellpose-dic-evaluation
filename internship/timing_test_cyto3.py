import os
import time
import glob
from cellpose import models
from skimage.io import imread

# same test as before but with the older cyto3 model instead of the
# default one - that one was way too slow on CPU

BASE = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE, "MI01", "Images")
img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:5]

print("testing on", len(img_paths), "images with cyto3")

t0 = time.time()
model = models.Cellpose(model_type="cyto3", gpu=False)
print("model load time:", round(time.time() - t0, 1), "s")

times = []
for p in img_paths:
    img = imread(p)
    t0 = time.time()
    masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0, 0])
    dt = time.time() - t0
    times.append(dt)
    print(os.path.basename(p), round(dt, 1), "s -", masks.max(), "cells")

avg = sum(times) / len(times)
n_total = 335

print()
print("avg per image:", round(avg, 1), "s")
print("estimated total for all sequences:", round(avg * n_total / 60, 1), "min")
