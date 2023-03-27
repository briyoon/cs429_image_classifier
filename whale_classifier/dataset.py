import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset

class HappyWhaleDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.annotation_file = annotation_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotation_file.iloc[idx, 0])
        image = read_image(img_path, mode=ImageReadMode.RGB)
        label = self.annotation_file.iloc[idx, 1]
        if self.transform:
            image = self.transform(image.float())
        if self.target_transform:
            label = self.target_transform(label)

        return image, label