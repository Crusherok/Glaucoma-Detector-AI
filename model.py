import torch.nn as nn
from torchvision import models


class GlaucomaResNet(nn.Module):
    """Wrapper around ResNet50 so saved weights with 'model.' prefix load cleanly."""

    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)


def get_model(model_type="resnet"):
    if model_type == "resnet":
        return GlaucomaResNet()
    raise ValueError("Unsupported model type. Use 'resnet'.")
