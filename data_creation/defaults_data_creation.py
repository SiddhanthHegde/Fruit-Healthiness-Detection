from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# converting json to mask
# -----------------------------------------------------------------------------

_C.JSON2MASK = CN()
_C.JSON2MASK.convert = True             # Whether to convert json to mask or not
_C.JSON2MASK.root_dir = ''              # Root directory containing the jsons
_C.JSON2MASK.save_dir = ''              # Directory to save the masks in .png format
_C.JSON2MASK.num_processes = 16         # Number of processes for multiprocessing
_C.JSON2MASK.num_labels = 3             # Labels should be only 0, 1 and 2