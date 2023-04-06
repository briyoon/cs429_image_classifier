from dataclasses import dataclass

from torch import optim
from torch import nn

@dataclass
class Config:
    epochs: int

    train_split: float
    batch_size: int

    criterion: nn

    optimizer: optim.Optimizer
    learning_rate: float
    weight_decay: float

    feature_extract: bool
    model: nn.Module

    def __str__(self) -> str:
        return \
            f"""
epochs: {self.epochs}
train_split: {self.train_split}
batch_size: {self.batch_size}
criterion: {self.criterion}
optimizer: {self.optimizer}
learning_rate: {self.learning_rate}
weight_decay: {self.weight_decay}
feature_extract: {self.feature_extract}
model: {self.model}
fc: {self.model.fc}
"""