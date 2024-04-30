from enum import Enum

import torch
from torch import nn

import models


class ModelType(Enum):
    CNN = "CNN"
    RESNET_18 = "RESNET_18"


model_type_to_model = {
    ModelType.CNN: models.cnn.Conv_2d,
    ModelType.RESNET_18: models.resnet18.ResNet18,
}


class Model(nn.Module):
    def __init__(
        self,
        path: str | None = None,
        model_type: ModelType = ModelType.RESNET_18,
    ):
        super(Model, self).__init__()
        if path is not None:
            self.model = torch.load(path)
        else:
            self.model = model_type_to_model[model_type]()

    def forward(self, x):
        """Snippet"""
        return self.model(x)

    def predict(self, x):
        """Song"""
        snippet_preds = [torch.sigmoid(self.model(snippet)) for snippet in x]
        return sum(snippet_preds) / len(snippet_preds) > 0.5
    
    def prob(self, x):
        """Song -- like predict, but returns the probability of being prog rock"""
        snippet_preds = [torch.sigmoid(self.model(snippet)) for snippet in x]
        return sum(snippet_preds) / len(snippet_preds)
