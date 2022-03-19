from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

_C.DATASET = CN()
_C.DATASET.root_dataset = ''
_C.DATASET.train_json = ''
_C.DATASET.val_json = ''

# -----------------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------------

_C.AUG = CN()
_C.AUG.perform_augs = False
_C.AUG.perform_aug_proba = 0.5
_C.AUG.gaussian_blur_proba = 0.7
_C.AUG.color_jitter_proba  = 0.7 
_C.AUG.grid_distort_proba  = 0.7 
_C.AUG.guass_noise_proba   = 0.7 

