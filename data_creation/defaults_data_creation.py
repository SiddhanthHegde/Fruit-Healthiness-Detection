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

# -----------------------------------------------------------------------------
# chipping data
# -----------------------------------------------------------------------------

_C.CHIPPING = CN()
_C.CHIPPING.perform_chipping = True     # Whether to preform batching (create image chips) or not.
_C.CHIPPING.dim = 512                   # dimension (in pixels) of the image chips to  be generated
_C.CHIPPING.stride = 256                # stride (in pixels) to be kept
_C.CHIPPING.padding = 0                 # Padding (in pixels) present in the PIXELSd images
_C.CHIPPING.num_processes = 1           # No. of processes for multiprocessing
_C.CHIPPING.rgb_dir = ''                # Path to the directory containing rgb images
_C.CHIPPING.gt_dir = ''                 # Path to the directory containing masks of images
_C.CHIPPING.out_dir = ''                # Path to the directory where you want to generate the chips

# -----------------------------------------------------------------------------
# Remove images having black portion above a certain threshold 
# -----------------------------------------------------------------------------

_C.REMOVE_IMAGES = CN()
_C.REMOVE_IMAGES.remove_images = True   # Whether to prune training data for rgb images having black portion above a certain threshold
_C.REMOVE_IMAGES.images_folder = ''     # Path to the directory containing the trainA and trainB folder
_C.REMOVE_IMAGES.percent_threshold = 1  # Percent thresold for removing images
_C.REMOVE_IMAGES.num_processes = 1      # No. of processes for multiprocessing

# -----------------------------------------------------------------------------
# compute pixel ratio and replace pixels
# -----------------------------------------------------------------------------

_C.PIXELS = CN()
_C.PIXELS.compute_pixel_ratio = True   # Whether to calculate pixels ratio (distribution) or not
_C.PIXELS.image_dir = ''               # Path to the directory containing the images
_C.PIXELS.class_values = []            # Class values present in gt 

# -----------------------------------------------------------------------------
# Make .json file 
# -----------------------------------------------------------------------------

_C.MAKE_JSON = CN()
_C.MAKE_JSON.make_json = True           # Whether to make .json file or not
_C.MAKE_JSON.in_dir = ''                # Path to the directory containing the rgb and gt images
_C.MAKE_JSON.out_dir = ''               # Path to the directory to generate the .json files
_C.MAKE_JSON.val_split_percent = 5      # Validation split
_C.MAKE_JSON.height = 2048              # Height of the images
_C.MAKE_JSON.width = 2048               # Width of the images