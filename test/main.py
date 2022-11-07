import json
import os
import shutil
import time
from skimage import io
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from utils import chip_test_images, predict_and_save, stitch_and_save
from dataset import SegmentationTestDataset
from test_model import SegmentationTestModel


def main(rgb_path, mean_values, std_values, batch_size, ckpt_path, output_dir, final_pred_dir, device):
    
    height = 3888
    width = 3888
    stride = 1024
    window_size = 1024
    chips_dir = 'test/data/chips'
    if os.path.exists(chips_dir):
        shutil.rmtree(chips_dir)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if os.path.exists(final_pred_dir):
        shutil.rmtree(final_pred_dir)
    os.makedirs(chips_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_pred_dir, exist_ok=True)
    
    start_time = time.time()
    print('Chipping test images now!!!')
    json_path = chip_test_images(rgb_path, chips_dir, window_size, stride)
    print('Test images chipped')
    print('Predicting the outputs')

    test_dataset = SegmentationTestDataset(chips_dir, mean_values, std_values)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
    model = SegmentationTestModel().to(device)
    model = model.load_from_checkpoint(ckpt_path, map_location=device)
    predict_and_save(model, test_dataloader, output_dir, device)
    print('Test chips predicted')
    print('Stitching the predictions')
    stitch_and_save(json_path, height, width, stride, window_size, output_dir, final_pred_dir)
    end_time = time.time()

    n_images = len(os.listdir(rgb_path))
    total_time = end_time - start_time
    mins = int(total_time // 60)
    secs = total_time % 60
    total_time_p = total_time / n_images
    mins_p = int(total_time_p // 60)
    secs_p = total_time_p % 60
    print(f'Total time taken to predict {str(n_images)} images: {str(mins)} mins {secs:.3} secs')
    print(f'Approx time per image: {str(mins_p)} mins {secs_p:.3} secs')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rgb_path = 'test/data/input'
    mean_values = [0.5924760223571813, 0.6095723768544002, 0.4908530449121363]
    std_values = [0.13440405402135033, 0.12584898251577034, 0.2270173717238626]
    batch_size = 6
    output_dir = 'test/data/output'
    final_pred_dir = 'test/data/final_preds'
    ckpt_path = 'test/data/Guava_Final.ckpt'
    main(rgb_path, mean_values, std_values, batch_size, ckpt_path, output_dir, final_pred_dir, device)