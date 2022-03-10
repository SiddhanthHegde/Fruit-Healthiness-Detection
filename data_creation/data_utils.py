import os
import numpy as np
from osgeo import gdal
from skimage import io
from PIL import Image

def verify_gt_and_rgb(rgb_dir: str, gt_dir: str):
    rgb_names = [rgb for rgb in os.listdir(rgb_dir) if rgb.endswith('.JPG')]    
    gt_names = [gt for gt in os.listdir(gt_dir) if gt.endswith('.png')]    
    if len(rgb_names) == len(gt_names):        
        for rgb in rgb_names:
            gt_tif_name = rgb.replace('.JPG', '.png')
            if gt_tif_name not in gt_names:
                print('All RGBs do not have corresponding grount truth raster. Still moving on.')
    else:
        print('All RGBs do not have corresponding grount truth raster. Still moving on.')

def get_image_dimensions(rgb_path: str):    
    rgb_raster = io.imread(rgb_path)    
    height, width = rgb_raster.shape[0:2]    
    return height, width

def create_slice_task_list(rgb_path: str, gt_path: str, out_rgb_path: str, out_gt_path: str, 
                            save_id: int, height: int, width: int, stride: int, dim: int, padding=0):
    task_list = []
    for i in range(0, height + 1, stride):
        for j in range(0, width + 1, stride):
            task_list.append([rgb_path, gt_path, out_rgb_path, out_gt_path, save_id, i, j, dim, padding])
            save_id += 1

    return task_list, save_id

def translate_rgb_gt(rgb_input: str, gt_input: str, out_rgb_path: str, out_gt_path: str, save_id: int, i: int, j: int, dim: int, padding=0):      
    out_rgb_chip = os.path.join(out_rgb_path, '{}.png'.format(save_id))
    out_gt_chip = os.path.join(out_gt_path, '{}.png'.format(save_id))
                                       
    gdal.Translate(out_rgb_chip, rgb_input, srcWin=[i + (padding), j + (padding), dim, dim], format="PNG", outputType=gdal.GDT_Byte)
    gdal.Translate(out_gt_chip, gt_input, srcWin=[i + (padding), j + (padding), dim, dim], format="PNG", outputType=gdal.GDT_Byte)
    
    if os.path.exists(out_rgb_chip + '.aux.xml'):
        os.remove(out_rgb_chip + '.aux.xml')
    if os.path.exists(out_gt_chip + '.aux.xml'):
        os.remove(out_gt_chip + '.aux.xml')

def remove_images(image_name: str, rgb_dir: str, gt_dir: str, removed_rgb_dir: str, removed_gt_dir: str, percent_threshold: int):
    src_rgb = os.path.join(rgb_dir, image_name)
    src_gt = os.path.join(gt_dir, image_name)
    dest_rgb = os.path.join(removed_rgb_dir, image_name)
    dest_gt = os.path.join(removed_gt_dir, image_name)

    rgb_np = np.array(Image.open(src_rgb))
    gt_np = np.array(Image.open(src_gt))

    if 0 in rgb_np:
        num_pixels_0 = (rgb_np[:,:,0][rgb_np[:,:,0] == 0]).size +  (rgb_np[:,:,1][rgb_np[:,:,1] == 0]).size + (rgb_np[:,:,2][rgb_np[:,:,2] == 0]).size 
        percent_pixels_0 = (float(num_pixels_0)/rgb_np.size) * 100
        all_pixels_0_in_gt = np.sum(np.where(gt_np == 0)) == len(gt_np.reshape(-1))
        if percent_pixels_0 >= percent_threshold or all_pixels_0_in_gt:
            os.rename(src_rgb, dest_rgb)
            os.rename(src_gt, dest_gt)