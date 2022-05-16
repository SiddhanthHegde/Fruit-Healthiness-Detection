import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class SegmentationTestDataset(Dataset):
    def __init__(self, img_dir, mean_values, std_values):
        self.list_sample = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values, std=std_values),
        ])
   
    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, index):

        image_name = self.list_sample[index]
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert('RGB')
        res = self.transform(image)
        height = res.shape[1]
        width = res.shape[2]
        

        data_dict = {}
        data_dict['image'] = res
        data_dict['height'] = height
        data_dict['width'] = width
        data_dict['image_name'] = image_name

        return data_dict