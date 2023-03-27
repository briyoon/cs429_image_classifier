import os

import torch
import torchvision
import numpy as np
import pandas as pd

train_annotation_path = "data/whales/train.csv"
train_img_path = "data/whales/train/"
test_img_path = "data/whales/test/"

# Load dataset
print("loading dataset...")
annotation_file = pd.read_csv(train_annotation_path)
if not os.path.exists("data/class_list.txt"):
    classes = [x for x in annotation_file["Id"].unique()]
    with open("data/class_list.txt", "w") as f:
        for x in classes:
            f.write(f"{x}\n")
else:
    with open("data/class_list.txt", "r") as f:
        classes = f.read().splitlines()
print(f"[{len(classes)}] number of classes detected")

# Load image, perform normalize and transform, and create onehot labels
images = []
labels = []
for index, (id, class_label) in annotation_file.iterrows():
    # image
    image = torchvision.io.read_image(os.path.join(train_img_path, id), mode=torchvision.io.ImageReadMode.RGB).float()
    image = torchvision.transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))(image)
    image = torchvision.transforms.Resize((224, 224), antialias=True)(image)
    images.append(image)

    # label
    onehot = np.zeros(len(classes), dtype=np.float32)
    onehot[classes.index(class_label)] = 1
    labels.append(onehot)

print()