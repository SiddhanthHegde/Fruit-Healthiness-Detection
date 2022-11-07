import os
import cv2
import numpy as np
from multiprocessing import Pool, Manager, Process
from tqdm import tqdm
import time


def RGB_Mean_StdDev(imgs_dir, save_txt_dir = None, ignore_value=0):
    count = 0
    r_mean_cumulative, g_mean_cumulative, b_mean_cumulative = 0, 0, 0
    r_stdDev_cumulative, g_stdDev_cumulative, b_stdDev_cumulative = 0, 0, 0

    for filename in tqdm(os.listdir(imgs_dir)):
        img = cv2.imread(os.path.join(imgs_dir, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            redChannel = np.reshape(img[:, :, 0], -1)
            r_mean_cumulative += redChannel[redChannel != ignore_value].mean()
            r_stdDev_cumulative += redChannel[redChannel != ignore_value].std()

            greenChannel = np.reshape(img[:, :, 1], -1)
            g_mean_cumulative += greenChannel[greenChannel != ignore_value].mean()
            g_stdDev_cumulative += greenChannel[greenChannel != ignore_value].std()

            blueChannel = np.reshape(img[:, :, 2], -1)
            b_mean_cumulative += blueChannel[blueChannel != ignore_value].mean()
            b_stdDev_cumulative += blueChannel[blueChannel != ignore_value].std()

            count += 1
            img = None
    r_mean, g_mean, b_mean = r_mean_cumulative / \
        count, g_mean_cumulative / count, b_mean_cumulative / count
    r_stdDev, g_stdDev, b_stdDev = r_stdDev_cumulative / count, g_stdDev_cumulative / count, \
        b_stdDev_cumulative / count

    r_mean_scaled, g_mean_scaled, b_mean_scaled = r_mean/255.0, g_mean/255.0, b_mean/255.0
    r_std_scaled, g_std_scaled, b_std_scaled = r_stdDev/255.0, g_stdDev/255.0, b_stdDev/255.0

    print(f'mean values: [{r_mean_scaled}, {g_mean_scaled}, {b_mean_scaled}]' )
    print(f'std values: [{r_std_scaled}, {g_std_scaled}, {b_std_scaled}]')
    
    if save_txt_dir is not None:
        save_txt_path = os.path.join(save_txt_dir, 'mean_std.txt')
        with open(save_txt_path, 'a') as txt_file:
            txt_file.write("r_mean {}\n".format(r_mean_scaled))
            txt_file.write("g_mean {}\n".format(g_mean_scaled))
            txt_file.write("b_mean {}\n".format(b_mean_scaled))
            txt_file.write("r_stdDev {}\n".format(r_std_scaled))
            txt_file.write("g_stdDev {}\n".format(g_std_scaled))
            txt_file.write("b_stdDev {}\n".format(b_std_scaled))
    
    return {
        'mean': [r_mean_scaled, g_mean_scaled, b_mean_scaled],
        'std': [r_std_scaled, g_std_scaled, b_std_scaled]
    }
    
 
if __name__ == '__main__':
    
    imgs_dir = "Batched_Data/training/images"
    time_start = time.time()
    RGB_Mean_StdDev(imgs_dir) 
      
    print('total time taken = {} seconds'.format(time.time() - time_start))