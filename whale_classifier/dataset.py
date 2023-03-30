import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset

class HappyWhaleTrainDataset(Dataset):
    def __init__(self, annotation_path, img_dir, classes_path, transform=None, target_transform=None, cache_images=False):
        annotation_file = pd.read_csv(annotation_path)

        # Create class list
        if not os.path.exists(classes_path):
            self.classes = [x for x in annotation_file["Id"].unique()]
            with open(classes_path, "w") as f:
                for x in self.classes:
                    f.write(f"{x}\n")
        else:
            with open(classes_path, "r") as f:
                self.classes = f.read().splitlines()

        # Create onehot labels
        for index, (id, class_label) in annotation_file.iterrows():
            onehot = np.zeros(len(self.classes), dtype=np.float32)
            onehot[self.classes.index(class_label)] = 1
            annotation_file.at[index, "Id"] = onehot

        self.onehot_labels = annotation_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.cache_images = cache_images
        self.image_cache = {}

    def __len__(self):
        return len(self.onehot_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.onehot_labels.iloc[idx, 0])

        if self.cache_images and img_path in self.image_cache:
            image = self.image_cache[img_path]
        else:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = to_tensor(image)
            if self.cache_images:
                self.image_cache[img_path] = image

        label = self.onehot_labels.iloc[idx, 1]

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_files = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]