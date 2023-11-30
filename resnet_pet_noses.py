import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet_Pet_Noses(nn.Module):

    def __init__(self):
        super(ResNet_Pet_Noses, self).__init__()
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features=in_features, out_features=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet18(x)
        x = self.sigmoid(x)
        return x
