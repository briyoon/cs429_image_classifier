import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import DataLoader, random_split
from whale_classifier.dataset import HappyWhaleTrainDataset, TestDataset

from utils import progress_bar

train_annotation_path = "data/whales/train.csv"
train_img_path = "data/whales/train/"
classes_path = "data/whales/class_list.txt"
test_img_path = "data/whales/test/"

### CONFIGURATION (move to config file) ###
TRAIN_SPLIT = 0.8

### HYPER PARAMETERS (move to config file) ###
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.03
WEIGHT_DECAY = 0.03

# cudnn.benchmark = True
# plt.ion()   # interactive mode

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
    for idx, data in enumerate(p_bar := progress_bar(dataloader)):
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
        for idx, data in enumerate(p_bar := progress_bar(dataloader, desc="Validating")):
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

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in train_dataloader if phase == 'train' else val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

if __name__ == "__main__":

    # Data augmentation and normalization for training
    # Just normalization for validation
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

    data_dir = 'data/whales'
    # Load dataset and create dataloaders
    dataset = HappyWhaleTrainDataset(train_annotation_path, train_img_path, classes_path, data_transforms["train"])
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = TestDataset(test_img_path, data_transforms["test"])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() - 1, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() - 1, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() - 1)

    # dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    # dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'val', 'test']}

    dataset_sizes = {'train': train_size, 'val': val_size}

    device = get_device()
    class_names = dataset.classes

    # # Get a batch of training data
    # inputs, class_names = next(iter(train_dataloader))

    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[x for x in class_names])

    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Train
    print("training model...")
    history = {"loss": [], "val_loss": [], "oneshot_acc": [], "fiveshot_acc": [], "oneshot_val_acc": [], "fiveshot_val_acc": []}
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        loss, oneshot_acc, fiveshot_acc = train(model_ft, train_dataloader, criterion, optimizer_ft, device)
        val_loss, oneshot_val_acc, fiveshot_val_acc = validate(model_ft, val_dataloader, class_names, criterion, device)
        history["loss"].append(loss)
        history["oneshot_acc"].append(oneshot_acc)
        history["fiveshot_acc"].append(fiveshot_acc)
        history["val_loss"].append(val_loss)
        history["oneshot_val_acc"].append(oneshot_val_acc)
        history["fiveshot_val_acc"].append(fiveshot_val_acc)

    print()