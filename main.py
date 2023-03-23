import argparse

import pandas as pd
from torch.utils.data import DataLoader

from whale_classifier.dataset import HappyWhaleDataset

train_annotation_path = "data/train.csv"
train_img_path = "data/train/"
test_img_path = "data/test/"

# def parse_args():
#     argument_parser = argparse.ArgumentParser()

#     # load config
#     argument_parser.add_argument("--config", type=str, default="config.py")

#     return argument_parser.parse_args()

def main():
    # Load dataset
    train_dataset = HappyWhaleDataset(train_annotation_path, train_img_path)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create model

    # Train

    # Test

    # Plot

    # Save

    return

if __name__ == "__main__":
    # args = parse_args()
    main()