# Internship code: StarDist training and Cellpose predictions

This folder has the code I wrote during my internship at CNR (ICAR) for my Bachelor's thesis in Data Analytics at Università degli Studi della Campania "Luigi Vanvitelli":

*Numerical Evaluation of Cell Segmentation Methods: a Comparison between StarDist and Cellpose in DIC Microscopy*

Marjan Rafiei, 2026
Supervisor: Prof. Rosanna Campagna. Co-supervisor: Dr. Laura Antonelli (CNR, ICAR).

This code prepares the data and produces the segmentations: it converts the ALFI masks, trains the StarDist model and runs Cellpose. The final evaluation and all the results reported in the thesis are in the [evaluation](../evaluation) folder.

## What is in this folder

Main steps:

- config.py: paths, data split and StarDist settings. All the other scripts read from here.
- convert_masks.py: converts the original ALFI masks into instance masks (every cell has its own number) and class masks (1 = interphase, 2 = mitotic).
- train.py: trains StarDist 2D on MI01, MI02 and MI03 and uses MI04 for validation. At the end StarDist chooses its probability and NMS thresholds on MI04.
- run_cellpose_all.py: runs the pretrained Cellpose cyto3 model on every frame with automatic cell diameter.
- run_cellpose_all_v2.py: the same, but with the diameter fixed at 38.5 px.
- models/mitosis_model/: the trained StarDist model (weights_best.h5, config.json, thresholds.json). This is the model evaluated in the thesis.

Other scripts I used along the way:

- compute_diameter.py: average cell diameter in the MI01 to MI03 masks. It gave the 38.5 px used in the second Cellpose run.
- raw_mask_check.py and convert_step_check.py: I used these to find a bug in the first version of the mask conversion. The mask values were already between 0 and 255, but the code multiplied them by 255 again, so the mitotic value (255) overflowed and every mitotic cell was lost. convert_masks.py now rescales only when the values are really between 0 and 1.
- train_improved.py: a training run with augmentation and extra copies of frames with mitotic cells. It saves in the same model folder as train.py, so it is not used in the thesis comparison.
- test_cellpose.py, timing_test.py, timing_test_cyto3.py, test_diameter_compare.py: quick tests before the full Cellpose runs. The default Cellpose model was too slow on my laptop, so I used cyto3.

## Data

I used the ALFI dataset (Antonelli et al., Scientific Data, 2023), sequences MI01 to MI05. The images are not included. They can be downloaded here:
https://springernature.figshare.com/collections/ALFI_Cell_cycle_phenotype_annotations_of_label-free_time-lapse_imaging_data_from_cultured_human_cells/6436958/1

Each sequence goes next to the scripts:

```
MI01/Images/I_MI01_0001.png ...
MI01/Masks/M_MI01_0001.png ...
MI02/ ... MI05/
```

In the original masks, black is background, gray is an interphase cell and white is a mitotic cell.

| Role | Sequences |
|---|---|
| StarDist training | MI01, MI02, MI03 |
| Validation | MI04 |
| Test (used in the thesis) | MI05 |

## Installation

StarDist (TensorFlow) and Cellpose (PyTorch) need different packages, so I used two conda environments with Python 3.10. Everything runs on CPU.

```
conda create -n cellproject python=3.10
conda activate cellproject
pip install -r requirements_stardist.txt

conda create -n cellpose_env python=3.10
conda activate cellpose_env
pip install -r requirements_cellpose.txt
```

## How to run

From this folder:

```
conda activate cellproject
python convert_masks.py        # writes MIxx/Masks_converted and MIxx/Masks_classes
python train.py                # optional: the trained model is already in models/

conda activate cellpose_env
python run_cellpose_all.py     # writes MIxx/Cellpose_masks      (slow on CPU)
python run_cellpose_all_v2.py  # writes MIxx/Cellpose_masks_v2
```

Running train.py again replaces the model in models/mitosis_model, so the new model would give slightly different results from the ones in the thesis.

## Outputs used by the thesis code

| Output | Made by | Used in the thesis for |
|---|---|---|
| MIxx/Masks_converted, MIxx/Masks_classes | convert_masks.py | ground truth for all evaluations |
| models/mitosis_model | train.py | StarDist predictions on MI05 |
| MIxx/Cellpose_masks | run_cellpose_all.py | Cellpose, automatic diameter |
| MIxx/Cellpose_masks_v2 | run_cellpose_all_v2.py | Cellpose, fixed diameter 38.5 px |

The evaluation code reads these files from this folder (the path is set as `dataset_root` in `evaluation/code/configs/`).
