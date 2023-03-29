import os
import argparse
import configparser
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
# import torchvision.transforms as transforms
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

from torch.profiler import profile, record_function, ProfilerActivity

from whale_classifier.classifier import WhaleClassifier
from whale_classifier.dataset import HappyWhaleDataset

train_annotation_path = "data/whales/train.csv"
train_img_path = "data/whales/train/"
test_img_path = "data/whales/test/"
classes_path = "data/whales/class_list.txt"

### CONFIGURATION
TRAIN_SPLIT = 0.8

### HYPER PARAMETERS (change to config file) ###
EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# def parse_args():
#     parser = argparse.ArgumentParser(description="Train a whale classifier")
#     parser.add_argument("--config", type=str, default="config.ini", help="Path to the configuration file")
#     return parser.parse_args()

# def load_config(config_file):
#     config = configparser.ConfigParser()
#     config.read(config_file)
#     return config

def train(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for data in (progress_bar := tqdm(dataloader, desc=f"Epoch: {epoch + 1} / {EPOCHS}")):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.3f}")
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in (progress_bar := tqdm(dataloader, desc="Validating")):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.3f}")
    return running_loss / len(dataloader)

def inference():
    pass

def get_device() -> torch.device:
    """
    Determines the best available device (CUDA, MPS, or CPU) for PyTorch operations.

    This function checks for the availability of devices in the following order:
    1. CUDA
    2. MPS (if built)
    3. CPU

    If CUDA is available, it sets the device to 'cuda' and prints the CUDA device name.
    If CUDA is not available but MPS is built, it checks for MPS availability.
        - If MPS is available, it sets the device to 'mps' and prints that it's using the MPS device.
        - If MPS is built but not available, it defaults to the CPU and prints a message.
    If neither CUDA nor MPS is built, it defaults to the CPU.

    Returns:
        torch.device: The best available device for PyTorch operations ('cuda', 'mps', or 'cpu').
    """
    device = None

    # Check for CUDA
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_built():
        if torch.backends.mps.is_available():
            device = 'mps'
            print("Using MPS device")
        else:
            print("MPS device is built but not available. Using CPU instead.")
    else:
        device = 'cpu'
        print("Using CPU")

    return torch.device(device)

def main():
    device = get_device()

    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((300, 300), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
    ])

    dataset = HappyWhaleDataset(train_annotation_path, train_img_path, classes_path, transform)
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() - 1, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() - 1, pin_memory=True)

    # Create model, optimizer, and loss
    print("creating model...")
    # whale_classifier = torch.compile(WhaleClassifier(len(dataset.classes)).to(device))
    whale_classifier = WhaleClassifier(len(dataset.classes)).to(device)
    # optimizer = torch.optim.Adam(whale_classifier.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.SGD(whale_classifier.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    print("training model...")
    for epoch in range(EPOCHS):
        train_loss = train(whale_classifier, train_dataloader, criterion, optimizer, device, epoch)
        val_loss = validate(whale_classifier, val_dataloader, criterion, device)

    print('Finished Training')
    # Save model
    torch.save(whale_classifier.state_dict(), f"models/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth")

    # Run inference (for now just run on train set since we dont have test labels)

    # Plot results


if __name__ == "__main__":
    # args = parse_args()
    main()