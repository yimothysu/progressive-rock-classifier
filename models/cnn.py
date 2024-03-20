"""
CNN model
"""

import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("forward() method is not implemented in the model.")
