import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Classifier(nn.Module):
    def __init__(self, num_classes=1):
        super(Classifier, self).__init__()
        model = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)