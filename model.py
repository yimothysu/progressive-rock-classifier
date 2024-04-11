import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, path: str | None = None):
        super(Model, self).__init__()
        if path is not None:
            self.model = torch.load(path)
        else:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet18", pretrained=True
            )
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            model.fc = torch.nn.Linear(model.fc.in_features, 1)
            self.model = model

    def forward(self, x):
        """Snippet"""
        return self.model(x)

    def predict(self, x):
        """Song"""
        snippet_preds = [self.model(snippet) for snippet in x]
        return sum(snippet_preds) / len(snippet_preds) > 0.5
