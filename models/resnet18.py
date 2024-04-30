"""
CNN model
"""

import torch
from torch import nn


class ResNet18(torch.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
