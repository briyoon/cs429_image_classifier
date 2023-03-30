import os
import csv

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from whale_classifier.dataset import TestDataset
from whale_classifier.cnn0 import WhaleClassifier

BATCH_SIZE = 32

test_img_path = "data/whales/test/"
model_path = "models/2023-03-29_03-36-08.pth"
classes_path = "data/whales/class_list.txt"

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
    # load testing images
    # Define image transform
    transform = transforms.Compose([
        transforms.Resize((300, 300), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
    ])

    with open(classes_path, "r") as f:
        classes = f.read().splitlines()

    dataset = TestDataset(test_img_path, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() - 1)

    # import pytorch model
    model = WhaleClassifier(len(classes)).to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    # run inference on test images
    results = []
    model.eval()
    with torch.no_grad():
        for images, image_names in dataloader:
            images, image_names = images.to(device), image_names
            outputs = model(images)
            _, predicted_classes = torch.topk(outputs, 5, 1)
            for image_name, predicted_class in zip(image_names, predicted_classes):
                results.append((image_name, [classes[x.item()] for x in predicted_class]))

    # save results to csv
    filename = model_path.split("/")[1].split(".")[0]
    with open(f"submissions/{filename}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image", "Id"])
        for result in results:
            class_results = " ".join(result[1])
            writer.writerow([result[0], class_results])

if __name__ == "__main__":
    main()