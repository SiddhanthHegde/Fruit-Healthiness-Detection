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

# -----------------------------------------------------------------------------
# Model design
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.model_arch = 'unet'
_C.MODEL.encoder_name = 'resnet18'
_C.MODEL.in_channels = 3
_C.MODEL.classes = 3
_C.MODEL.encoder_weights = 'imagenet'

# -----------------------------------------------------------------------------
# Train parameters
# -----------------------------------------------------------------------------

_C.TRAIN = CN()
_C.TRAIN.batch_size = 1
_C.TRAIN.shuffle = True
_C.TRAIN.num_workers = 0
_C.TRAIN.class_weights = []
_C.TRAIN.lr = 2e-2
_C.TRAIN.scheduler_patience = 2
_C.TRAIN.lr_reduce_factor = 0.01
_C.TRAIN.init_ckpt = ''
_C.TRAIN.wandb_iters = 100
_C.TRAIN.n_epochs = 10
_C.TRAIN.wandb_project_name = ''

# -----------------------------------------------------------------------------
# Val parameters
# -----------------------------------------------------------------------------

_C.VAL = CN()
_C.VAL.batch_size = 1
_C.VAL.shuffle = False
_C.num_workers = 0