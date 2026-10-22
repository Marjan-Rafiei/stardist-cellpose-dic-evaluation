import os
import re
import glob
import time
import numpy as np
from cellpose import models
from skimage.io import imread, imsave

from config import BASE, SEQUENCES, IMAGES_FOLDER

# runs cellpose on every image in every sequence and saves the predicted
# masks so we don't have to redo this every time. takes a while, just
# let it run

OUT_FOLDER = "Cellpose_masks"


def get_frame_id(filename):
    m = re.search(r"(\d+)\.png$", filename)
    return m.group(1) if m else None


def main():
    model = models.Cellpose(model_type="cyto3", gpu=False)

    t_start = time.time()
    done = 0

    for seq in SEQUENCES:
        img_dir = os.path.join(BASE, seq, IMAGES_FOLDER)
        out_dir = os.path.join(BASE, seq, OUT_FOLDER)
        os.makedirs(out_dir, exist_ok=True)

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        print(seq, "-", len(img_paths), "images")

        for i, p in enumerate(img_paths):
            frame_id = get_frame_id(os.path.basename(p))
            if frame_id is None:
                continue

            out_path = os.path.join(out_dir, f"C_{seq}_{frame_id}.png")
            if os.path.exists(out_path):
                # already ran this one, skip (so we can resume if it gets interrupted)
                continue

            img = imread(p)
            masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0, 0])
            imsave(out_path, masks.astype(np.uint16), check_contrast=False)
            done += 1

            if (i + 1) % 5 == 0 or (i + 1) == len(img_paths):
                mins = (time.time() - t_start) / 60
                print(f"  {i+1}/{len(img_paths)} done, {done} total so far, {mins:.1f} min elapsed")

        print(seq, "done\n")

    print("all done -", round((time.time() - t_start) / 60, 1), "min total")


if __name__ == "__main__":
    main()
