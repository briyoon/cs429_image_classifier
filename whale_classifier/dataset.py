import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset

class HappyWhaleDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.onehot_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.onehot_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.onehot_labels.iloc[idx, 0])
        image = read_image(img_path, mode=ImageReadMode.RGB)
        label = self.onehot_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image.float())
        if self.target_transform:
            label = self.target_transform(label)
        return image, label