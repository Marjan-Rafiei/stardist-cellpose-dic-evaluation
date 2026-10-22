# Thesis code: evaluation of StarDist and Cellpose on MI05

This folder has the evaluation code for my Bachelor's thesis in Data Analytics at Università degli Studi della Campania "Luigi Vanvitelli":

*Numerical Evaluation of Cell Segmentation Methods: a Comparison between StarDist and Cellpose in DIC Microscopy*

Marjan Rafiei, 2026
Supervisor: Prof. Rosanna Campagna. Co-supervisor: Dr. Laura Antonelli (CNR, ICAR).

All the numbers, tables and figures in the thesis come from the code and the result files in this folder. The segmentations themselves (the trained StarDist model and the two Cellpose runs) come from the [internship](../internship) folder.

## What the code does

1. Checks the ALFI dataset (frames, masks, label values, duplicates).
2. Trains one Random Forest on cell shape features (MI01 to MI03) and checks it on MI04. The same classifier is then used for every method.
3. Runs the saved StarDist model on MI05.
4. Evaluates the three prediction sources on MI05 with the same rules: StarDist, Cellpose with automatic diameter and Cellpose with fixed diameter 38.5 px.
5. Follows the errors into cell counts and mitotic ratios, and checks how stable the results are (IoU threshold, moving-block bootstrap, feature importance).
6. Builds the figures, tables and numbers used in the LaTeX thesis.

A predicted cell matches an annotated cell when their IoU is at least 0.5 (one-to-one matching).

## Main results on MI05

| Method | Precision | Recall | Detection F1 | PQ | Matched IoU |
|---|---|---|---|---|---|
| StarDist | 0.808 | 0.670 | 0.733 | 0.482 | 0.658 |
| Cellpose automatic | 0.456 | 0.288 | 0.353 | 0.252 | 0.713 |
| Cellpose fixed | 0.440 | 0.282 | 0.344 | 0.246 | 0.714 |

| Method | Matched-only macro F1 | End-to-end macro F1 | Interphase F1 | Mitotic F1 |
|---|---|---|---|---|
| StarDist | 0.493 | 0.365 | 0.539 | 0.192 |
| Cellpose automatic | 0.662 | 0.391 | 0.091 | 0.690 |
| Cellpose fixed | 0.629 | 0.379 | 0.081 | 0.678 |

StarDist finds many more cells. Cellpose outlines the cells it finds a bit more closely, but it misses most of them. The full numbers are in results/verified/comparison_summary.csv.

## Folder structure

```
code/                  evaluation and analysis scripts
  configs/             dataset, classifier and Cellpose settings (JSON)
  tests/               unit tests
scripts_figures/       scripts that build the thesis figures, tables and LaTeX numbers
results/verified/      all result files used in the thesis
environment/           conda environments and package versions
```

Main scripts in code/:

| Script | What it does |
|---|---|
| audit_dataset.py, dataset_quality_audit.py | dataset checks |
| train_classifier.py | Random Forest on shape features (MI01 to MI03, checked on MI04) |
| run_stardist_inference.py | StarDist predictions on MI05 with the saved model |
| verify_predictions.py | checks the saved Cellpose masks |
| unified_evaluation.py | the common evaluator (matching, F1, PQ, IoU, classes, counts) |
| build_comparison_summary.py | collects the three evaluations in one table |
| block_bootstrap_metrics.py, temporal_bootstrap_audit.py | moving-block bootstrap on the MI05 frames |
| iou_sensitivity.py | results for other IoU thresholds |
| counts_ratio_validation.py, segmentation_analytics_error_chain.py | how errors change cell counts and mitotic ratios |
| classification_transition_analysis.py, morphology_domain_shift.py | how predicted shapes change the classifier output |
| random_forest_audit.py, feature_importance_audit.py, classifier_diagnostics.py | classifier checks |
| mask_conversion_audit.py | checks the mask conversion bug and the corrected masks |

Each result folder in results/verified/ has a manifest (JSON) with the inputs, settings, software versions and command used.

## Data and inputs

The code needs the ALFI data folder prepared with the [internship](../internship/README.md) scripts, containing:

- MI01 to MI05 images and the converted masks (Masks_converted, Masks_classes)
- MIxx/Cellpose_masks and MIxx/Cellpose_masks_v2
- models/mitosis_model

The path to this folder is set as `dataset_root` in the files in code/configs/ (currently `PATH/TO/ALFI_DATA`). Change it to the location on your computer before running.

## Installation

The evaluation runs in the StarDist environment:

```
conda env create -f environment/cellproject.environment.yml
conda activate cellproject
```

Package versions are listed in environment/ENVIRONMENT_INVENTORY.md (StarDist 0.9.2, TensorFlow 2.21.0, scikit-learn 1.7.2, Python 3.10).

## How to run

From this folder, with DATA set to the internship folder:

```
python -m code.audit_dataset --config code/configs/dataset_v1.json --output results/verified/dataset_audit
python -m code.train_classifier --dataset-root DATA --config code/configs/classifier_v1.json --output-dir results/verified/classifier_v1
python -m code.run_stardist_inference --dataset-root DATA --sequence MI05 --model-dir DATA/models/mitosis_model --output-dir results/verified/stardist_saved/predictions
python -m code.unified_evaluation --model-name stardist_saved --prediction-root results/verified/stardist_saved/predictions --ground-truth-root DATA/MI05/Masks_converted --ground-truth-class-root DATA/MI05/Masks_classes --sequence MI05 --iou-threshold 0.5 --classifier results/verified/classifier_v1/classifier.joblib --classifier-config code/configs/classifier_v1.json --output-dir results/verified/stardist_saved
python -m code.build_comparison_summary --results-root results/verified
```

The Cellpose evaluations use the same unified_evaluation command with --prediction-root DATA/MI05/Cellpose_masks (or Cellpose_masks_v2) and --model-name cellpose_auto (or cellpose_fixed). The exact command of every run is saved in its evaluation_manifest.json.

Unit tests:

```
python -m unittest discover -s code/tests -v
```

The LaTeX numbers and figures are rebuilt with scripts_figures/export_result_macros.py and scripts_figures/build_thesis_assets.py. They write into the thesis LaTeX folder, which is not part of this repository.

## Limitations

- All test results come from one sequence (MI05), whose frames are not independent.
- The StarDist model was not retrained (only its MI05 predictions were generated again), and Cellpose was not rerun. The saved model and masks were evaluated as they are.
- The classifier only uses shape features, so it cannot see intensity or texture.
