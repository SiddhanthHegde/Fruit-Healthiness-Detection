import os
import json
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from data_creation.get_mean_std import RGB_Mean_StdDev

class TrainDataset(Dataset):
    def __init__(self, cfg, json_file):
        self.cfg = cfg
        self.list_sample = [json.loads(x.rstrip())
                            for x in open(json_file, 'r')]
        imgs_dir = os.path.join(cfg.DATASET.root_dataset, 'training', 'images')
        self.mean_std_dict = RGB_Mean_StdDev(imgs_dir, cfg.DATASET.root_dataset)
        self.augments = self.get_augmentations(cfg)
   
    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, index):

        train_dict = self.list_sample[index]
        img_path = train_dict['img_path']
        annots_path = train_dict['annots_path']
        height = train_dict['height']
        width = train_dict['width']

        image = np.array(Image.open(img_path).convert('RGB'))
        label = np.array(Image.open(annots_path).convert('L'))

        image_name = os.path.basename(img_path)
        res = self.augments(image=image, mask=label)
        

        data_dict = {}
        data_dict['image'] = res['image']
        data_dict['label'] = res['mask']
        data_dict['height'] = height
        data_dict['width'] = width
        data_dict['image_name'] = image_name

        return data_dict

    def get_augmentations(self, cfg):
        mean_values = self.mean_std_dict['mean']
        stddev_values = self.mean_std_dict['std']
        
        if cfg.AUG.perform_augs:
            perform_aug_proba = cfg.AUG.perform_aug_proba
        else:
            perform_aug_proba = 0.0
        
        gaussian_blur_proba = cfg.AUG.gaussian_blur_proba
        color_jitter_proba = cfg.AUG.color_jitter_proba  
        grid_distort_proba = cfg.AUG.grid_distort_proba  
        guass_noise_proba = cfg.AUG.guass_noise_proba   
        augments = albu.Compose([
            albu.OneOf([
                albu.GaussianBlur(p=gaussian_blur_proba),
                albu.ColorJitter(p=color_jitter_proba),
                albu.GridDistortion(p=grid_distort_proba),
                albu.GaussNoise(p=guass_noise_proba),
            ],p=perform_aug_proba),
            albu.Normalize(mean=mean_values, std=stddev_values, p=1),
            ToTensorV2()   
        ])

        return augments

class ValDataset(Dataset):
    def __init__(self, cfg, json_file):
        self.cfg = cfg
        self.list_sample = [json.loads(x.rstrip())
                            for x in open(json_file, 'r')]
        
        imgs_dir = os.path.join(cfg.DATASET.root_dataset, 'training', 'images')
        mean_std_dict = RGB_Mean_StdDev(imgs_dir, cfg.DATASET.root_dataset)
        
        mean_values = mean_std_dict['mean']
        stddev_values = mean_std_dict['std']
        
        self.augments = albu.Compose([
            albu.Normalize(mean=mean_values, std=stddev_values),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, index):

        val_dict = self.list_sample[index]
        img_path = val_dict['img_path']
        annots_path = val_dict['annots_path']
        height = val_dict['height']
        width = val_dict['width']

        image = np.array(Image.open(img_path).convert('RGB'))
        label = np.array(Image.open(annots_path).convert('L'))

        image_name = os.path.basename(img_path)
        res = self.augments(image=image, mask=label)

        data_dict = {}
        data_dict['image'] = res['image']
        data_dict['label'] = res['mask']
        data_dict['height'] = height
        data_dict['width'] = width
        data_dict['image_name'] = image_name

        return data_dict