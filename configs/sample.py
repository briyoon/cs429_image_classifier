import torch
from torch import nn, optim
from torchvision import models

from utils import Config

CLASS_NUMS = 4121

EPOCHS = 50

# [DATA]
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32

# [CRITERION]
CRITERION = nn.CrossEntropyLoss()

# [OPTIMIZER]
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.00001 # l2 regularization
OPTIMIZER = optim.Adam(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# [MODEL]
FEATURE_EXTRACT = True
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# if feature extraction, freeze all layers
if FEATURE_EXTRACT:
    for param in model.parameters():
        param.requires_grad = False

num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, CLASS_NUMS)
)

CONFIG = Config(
    epochs=EPOCHS,
    train_split=TRAIN_SPLIT,
    batch_size=BATCH_SIZE,
    criterion=CRITERION,
    optimizer=OPTIMIZER,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    feature_extract=FEATURE_EXTRACT,
    model=model
)




