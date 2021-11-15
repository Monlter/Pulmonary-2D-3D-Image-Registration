import torch.nn as nn
import torch
from tools.data_loader import normalData_reverse, stdData_reverse
import numpy as np
import os


class CNN_net(nn.Module):
    def __init__(self,in_channels):
        super(CNN_net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=8,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, 3, 1, 1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.mlp1 = nn.Linear(15*15*32, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.mlp2 = nn.Linear(1024, 3)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.dropout1(x)
        x = self.mlp2(x)
        return x

if __name__ == '__main__':
    pass


