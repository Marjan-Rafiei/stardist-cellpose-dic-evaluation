import os
import re
import glob
import numpy as np
from skimage.io import imread
from skimage.measure import regionprops

from config import BASE, TRAIN_SEQS, MASKS_CONV_FOLDER

# cellpose's auto diameter guess seems way off (way fewer cells detected
# than there actually are), so let's just measure the real cell size from
# the ground truth masks and pass that in manually


def get_frame_id(filename):
    m = re.search(r"(\d+)\.png$", filename)
    return m.group(1) if m else None


diameters = []
for seq in TRAIN_SEQS:
    conv_dir = os.path.join(BASE, seq, MASKS_CONV_FOLDER)
    for p in glob.glob(os.path.join(conv_dir, "*.png")):
        mask = imread(p)
        for props in regionprops(mask):
            diameters.append(props.equivalent_diameter)

diameters = np.array(diameters)
print("n cells:", len(diameters), "from", TRAIN_SEQS)
print("mean diameter:", round(diameters.mean(), 1), "px")
print("median diameter:", round(np.median(diameters), 1), "px")
print("min/max:", round(diameters.min(), 1), round(diameters.max(), 1))
