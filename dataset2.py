import os
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from data_creation.get_mean_std import RGB_Mean_StdDev

class SegmentationDataset(Dataset):
    def __init__(self, cfg, json_file):
        self.cfg = cfg
        self.list_sample = [json.loads(x.rstrip())
                            for x in open(json_file, 'r')]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.TRAIN.mean_values, std=cfg.TRAIN.std_values),
        ])
   
    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, index):

        train_dict = self.list_sample[index]
        img_path = train_dict['img_path']
        annots_path = train_dict['annots_path']
        height = train_dict['height']
        width = train_dict['width']

        image = Image.open(img_path).convert('RGB')
        label = Image.open(annots_path).convert('L')

        image_name = os.path.basename(img_path)
        res = self.normalise(image=image, mask=label)
        

        data_dict = {}
        data_dict['image'] = res['image']
        data_dict['label'] = res['mask']
        data_dict['height'] = height
        data_dict['width'] = width
        data_dict['image_name'] = image_name

        return data_dict

    def normalise(self, image, mask):
        
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(np.array(mask), dtype=torch.long)

        return {
            'image': image_tensor,
            'mask': label_tensor
        }