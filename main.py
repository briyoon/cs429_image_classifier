import os
import csv
import argparse
import configparser
import json
from datetime import datetime
import importlib
import sys

import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt

from whale_classifier.dataset import HappyWhaleTrainDataset, TestDataset
from utils import ProgressBar, Config

train_annotation_path = "data/whales/train.csv"
train_img_path = "data/whales/train/"
classes_path = "data/whales/class_list.txt"
test_img_path = "data/whales/test/"

def parse_args():
    parser = argparse.ArgumentParser(description="Train a whale classifier")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to the configuration file")
    return parser.parse_args()

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

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train(model, dataloader, criterion, optimizer, device) -> float:
    """
    Trains the given model for one epoch using the provided dataloader, criterion, optimizer, and device.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        dataloader (DataLoader): The DataLoader object containing the training dataset.
        criterion (nn.Module): The loss function used to compute the training loss.
        optimizer (optim.Optimizer): The optimization algorithm used for model training.
        device (torch.device): The device to which tensors should be moved for computation (e.g., 'cuda' or 'cpu').

    Returns:
        float: The average training loss over the entire training dataset for one epoch.
    """
    model.train()
    running_loss = 0.0
    oneshot_acc_count = 0
    fiveshot_acc_count = 0
    for idx, data in enumerate(p_bar := ProgressBar(dataloader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 5_shot and 1_shot acc
        _, predicted_classes = torch.topk(outputs, 5, 1)
        for label, predicted_class in zip(labels, predicted_classes):
            label_index = torch.argmax(label)
            oneshot_acc_count += 1 if label_index == predicted_class[0] else 0
            fiveshot_acc_count += 1 if label_index in predicted_class else 0

        # loss
        running_loss += loss.item()
        p_bar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "oneshot_acc": f"{oneshot_acc_count / ((idx + 1) * dataloader.batch_size) * 100:.2f}",
            "fiveshot_acc": f"{fiveshot_acc_count / ((idx + 1) * dataloader.batch_size) * 100:.2f}"
        })
    num_images = len(dataloader) * dataloader.batch_size
    num_batches = len(dataloader)
    return running_loss / num_batches, oneshot_acc_count / num_images, fiveshot_acc_count / num_images

def validate(model, dataloader, classes, criterion, device) -> float:
    """
    Validates the performance of the given model using the provided dataloader, criterion, and device.

    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        dataloader (DataLoader): The DataLoader object containing the validation dataset.
        criterion (nn.Module): The loss function used to compute the validation loss.
        device (torch.device): The device to which tensors should be moved for computation (e.g., 'cuda' or 'cpu').

    Returns:
        float: The average validation loss over the entire validation dataset.
    """
    model.eval()
    num_images = len(dataloader) * dataloader.batch_size
    num_batches = len(dataloader)
    running_loss = 0.0
    oneshot_acc_count = 0
    fiveshot_acc_count = 0
    with torch.no_grad():
        for idx, data in enumerate(p_bar := ProgressBar(dataloader, desc="Validating")):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 5_shot and 1_shot acc
            _, predicted_classes = torch.topk(outputs, 5, 1)
            for label, predicted_class in zip(labels, predicted_classes):
                label_index = torch.argmax(label)
                oneshot_acc_count += 1 if label_index == predicted_class[0] else 0
                fiveshot_acc_count += 1 if label_index in predicted_class else 0

            running_loss += loss.item()
            p_bar.set_postfix({
                "val_loss": f"{loss.item():.3f}",
                "val_oneshot_acc": f"{oneshot_acc_count / ((idx + 1) * dataloader.batch_size) * 100:.2f}",
                "val_fiveshot_acc": f"{fiveshot_acc_count / ((idx + 1) * dataloader.batch_size) * 100:.2f}"
            })

    return running_loss / num_batches, oneshot_acc_count / num_images, fiveshot_acc_count / num_images

def test(model, dataloader, class_names, device):
    results = []
    model.eval()
    with torch.no_grad():
        for images, image_names in dataloader:
            images, image_names = images.to(device), image_names
            outputs = model(images)
            _, predicted_classes = torch.topk(outputs, 5, 1)
            for image_name, predicted_class in zip(image_names, predicted_classes):
                results.append((image_name, [class_names[x.item()] for x in predicted_class]))

    return results

def main(args, cf: Config):
    # Data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load dataset and create dataloaders
    dataset = HappyWhaleTrainDataset(train_annotation_path, train_img_path, classes_path, data_transforms["train"])
    train_size = int(cf.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = TestDataset(test_img_path, data_transforms["test"])

    train_dataloader = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True, num_workers=os.cpu_count() - 1, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cf.batch_size, shuffle=True, num_workers=os.cpu_count() - 1, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cf.batch_size, shuffle=False, num_workers=os.cpu_count() - 1)

    device = get_device()
    class_names = dataset.class_names

    # Initialize model
    model: nn.Module = cf.model

    model = model.to(device)

    criterion = cf.criterion

    # All parameters are being optimized
    optimizer = cf.optimizer

    # Train
    print("training model...")
    history = {"loss": [], "val_loss": [], "oneshot_acc": [], "fiveshot_acc": [], "oneshot_val_acc": [], "fiveshot_val_acc": []}
    for epoch in range(cf.epochs):
        print(f"Epoch {epoch + 1}/{cf.epochs}")
        loss, oneshot_acc, fiveshot_acc = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, oneshot_val_acc, fiveshot_val_acc = validate(model, val_dataloader, class_names, criterion, device)
        history["loss"].append(loss)
        history["oneshot_acc"].append(oneshot_acc)
        history["fiveshot_acc"].append(fiveshot_acc)
        history["val_loss"].append(val_loss)
        history["oneshot_val_acc"].append(oneshot_val_acc)
        history["fiveshot_val_acc"].append(fiveshot_val_acc)

    print('Finished Training')

    # Save model and model parameters
    filename = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(f"models/{filename}", exist_ok=True)
    torch.save(model.state_dict(), f"models/{filename}/{filename}.pth")

    # run inference
    results = test(model, test_dataloader, class_names, device)

    # save results to csv
    with open(f"submissions/{filename}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image", "Id"])
        for result in results:
            class_results = " ".join(result[1])
            writer.writerow([result[0], class_results])

    # Plot and save training results
    loss_fig, loss_ax = plt.subplots(1)
    acc_fig, acc_ax = plt.subplots(1)

    loss_ax.plot(history["loss"], label="Training Loss")
    loss_ax.plot(history["val_loss"], label="Validation Loss")
    acc_ax.plot(history["oneshot_acc"], label="1-shot Training Accuracy")
    acc_ax.plot(history["fiveshot_acc"], label="5-shot Training Accuracy")
    acc_ax.plot(history["oneshot_val_acc"], label="1-shot Validation Accuracy")
    acc_ax.plot(history["fiveshot_val_acc"], label="5-shot Validation Accuracy")

    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")

    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")

    loss_fig.legend()
    acc_fig.legend()

    loss_fig.savefig(f"models/{filename}/{filename}_loss.png", dpi=300)
    acc_fig.savefig(f"models/{filename}/{filename}_acc.png", dpi=300)

    # save model config
    with open(f"models/{filename}/{filename}_config.txt", "w") as f:
        f.write(str(cf))

if __name__ == "__main__":
    args = parse_args()

    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)

    main(args, config.CONFIG)