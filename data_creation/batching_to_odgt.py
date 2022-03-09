from multiprocessing import Pool
import os
import argparse
import numpy as np
import shutil
from tqdm import tqdm
from defaults_data_creation import _C as cfg
from data_utils import verify_gt_and_rgb
from data_utils import get_image_dimensions
from data_utils import create_slice_task_list
from data_utils import translate_rgb_gt

#-----------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Create image chips, compute pixel ratio, split data and make odgt files')
    parser.add_argument('--cfg', metavar = 'FILE', help = 'path to config file', type = str, default='configs/data_creation_config.yaml')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)

#-----------------------------------------------------------------------
    if cfg.CHIPPING.perform_chipping:
        print('Creating image chips now!!')

        dim = cfg.CHIPPING.dim
        stride = cfg.CHIPPING.stride
        padding = cfg.CHIPPING.padding
        num_processes = cfg.CHIPPING.num_processes
        rgb_dir = cfg.CHIPPING.rgb_dir
        gt_dir = cfg.CHIPPING.gt_dir
        out_dir = cfg.CHIPPING.out_dir

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        out_rgb_path = os.path.join(out_dir, 'rgb')
        out_gt_path = os.path.join(out_dir, 'gt')

        os.makedirs(out_dir,exist_ok=True)
        os.makedirs(out_rgb_path,exist_ok=True)
        os.makedirs(out_gt_path,exist_ok=True)
        verify_gt_and_rgb(rgb_dir,gt_dir)

        save_id = 0
        rgb_list = [x for x in os.listdir(rgb_dir) if x.endswith('.JPG')]

        for rgb in tqdm(rgb_list):
            rgb_path = os.path.join(rgb_dir, rgb)
            gt_path = os.path.join(gt_dir, rgb.replace('.JPG','.png'))
            height, width = get_image_dimensions(rgb_path)

            with open(os.path.join(out_dir, 'batching_numbers.txt'), 'a') as batching_file:
                batching_file.write('Image Path: {}, Start chip id: {}\n'.format(rgb_path, save_id))

            task_list, save_id = create_slice_task_list(
                rgb_path, gt_path, out_rgb_path, out_gt_path, save_id, height, width, stride, dim, padding
            )

            p = Pool(num_processes)
            p.starmap(translate_rgb_gt, task_list)
            p.close()
            p.join()




        
        

                


