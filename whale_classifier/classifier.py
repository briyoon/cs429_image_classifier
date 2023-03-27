import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class WhaleClassifier(nn.Module):
    def __init__(self, classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = F.relu(x)
        # x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        # x = F.relu(x)
        # x = self.pool(x)
        x =self.pool(F.relu(self.conv3(x)))
        # x = F.relu(x)
        # x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc2(x)
        return x