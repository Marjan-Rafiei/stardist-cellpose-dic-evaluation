import os
import time
import glob
from cellpose import models
from skimage.io import imread

# just want to know how long this actually takes per image before
# running it on everything

BASE = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE, "MI01", "Images")
img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:5]

print("testing on", len(img_paths), "images")

t0 = time.time()
model = models.CellposeModel(gpu=False)
print("model load time:", round(time.time() - t0, 1), "s")

times = []
for p in img_paths:
    img = imread(p)
    t0 = time.time()
    masks, flows, styles = model.eval(img)
    dt = time.time() - t0
    times.append(dt)
    print(os.path.basename(p), round(dt, 1), "s -", masks.max(), "cells")

avg = sum(times) / len(times)
n_total = 335  # ~67 images x 5 sequences

print()
print("avg per image:", round(avg, 1), "s")
print("estimated total for all sequences:", round(avg * n_total / 60, 1), "min")
