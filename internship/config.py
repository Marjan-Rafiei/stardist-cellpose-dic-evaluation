"""
config.py — Central configuration for the Cell Segmentation pipeline.
All other scripts import from here so paths work on any computer.
"""
import os

# Automatically finds the project folder on any computer!
BASE = os.path.dirname(os.path.abspath(__file__))

# Sequences
SEQUENCES   = ['MI01', 'MI02', 'MI03', 'MI04', 'MI05']
TRAIN_SEQS  = ['MI01', 'MI02', 'MI03']
VAL_SEQS    = ['MI04']
TEST_SEQS   = ['MI05']

# Folder names
IMAGES_FOLDER     = 'Images'
MASKS_FOLDER      = 'Masks'
MASKS_CONV_FOLDER = 'Masks_converted'   # instance masks for StarDist
MASKS_CLS_FOLDER  = 'Masks_classes'     # class labels (1=interphase, 2=mitotic)

# Class labels
INTERPHASE_LABEL = 1   # gray cells in original mask
MITOTIC_LABEL    = 2   # white cells in original mask

# Model
MODEL_NAME  = 'mitosis_model'
MODELS_DIR  = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')

# StarDist hyperparameters
N_RAYS           = 32
GRID             = (2, 2)
TRAIN_PATCH_SIZE = (128, 128)
TRAIN_BATCH_SIZE = 2
TRAIN_EPOCHS     = 100
TRAIN_STEPS      = 50
LEARNING_RATE    = 3e-4

print(f"[config] Project folder: {BASE}")
