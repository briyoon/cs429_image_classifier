import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from whale_classifier.classifier import WhaleClassifier
from whale_classifier.dataset import HappyWhaleDataset

train_annotation_path = "data/whales/train.csv"
train_img_path = "data/whales/train/"
test_img_path = "data/whales/test/"

### HYPER PARAMETERS (change to config file) ###
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# def parse_args():
#     argument_parser = argparse.ArgumentParser()

#     # load config
#     argument_parser.add_argument("--config", type=str, default="config.py")

#     return argument_parser.parse_args()

def main():
    ### Device checks ###
    if torch.cuda.is_available(): # NVIDIA GPU ACCELERATION
        device = torch.device("cuda:0")
        print("Running on NVIDIA GPU")
    elif torch.backends.mps.is_built(): # M1 GPU ACCELERATION
        device = torch.device("mps")
        print("Running on M1 GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    device = torch.device(device)

    # Define image transform
    transform = transforms.Compose([
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
        transforms.Resize((600, 600), antialias=True)
    ])

    # Load dataset
    annotation_file = pd.read_csv(train_annotation_path)
    if not (os.path.exists("data/class_list.txt") or (os.path.isfile())):
        classes = [x for x in annotation_file["Id"].unique()]
        with open("data/class_list.txt", "w") as f:
            for x in classes:
                f.write(f"{x}\n")
    else:
        with open("data/class_list.txt", "r") as f:
            classes = f.read().splitlines()
    print(f"[{len(classes)}] number of classes detected")

    # Create onehot labels
    for index, (id, class_label) in annotation_file.iterrows():
        onehot = np.zeros(len(classes), dtype=np.float32)
        onehot[classes.index(class_label)] = 1
        annotation_file.at[index, "Id"] = onehot
    # annotation_file.to_csv("data/train_onehot.csv")
        # f.write(f"{id},[{','.join([str(x) for x in onehot])}]\n")

    train_dataset = HappyWhaleDataset(annotation_file, train_img_path, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_dataset = HappyWhaleDataset(annotation_file, train_img_path, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Create model, optimizer, and loss
    whale_classifier = WhaleClassifier(len(classes))
    optimizer = torch.optim.Adam(whale_classifier.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    # whale_classifier.train(train_dataloader)
    for epoch in range(EPOCHS): # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = whale_classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(i)
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0

    print('Finished Training')
    # Save model (Saves entire model and not just weights, use whale_classifier.state_dict() to just save weights)
    torch.save(whale_classifier, ".pt")

    # Run inference (for now just run on train set since we dont have test labels)
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = whale_classifier(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    # Plot and save stats


    return

if __name__ == "__main__":
    # args = parse_args()



    main()