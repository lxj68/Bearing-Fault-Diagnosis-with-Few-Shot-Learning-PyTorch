import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, depth_multiplier):
        super(SeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels * depth_multiplier, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels * depth_multiplier, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.convnet = nn.Sequential(
            SeparableConv1D(2, 14, kernel_size=64, stride=8, depth_multiplier=29),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            SeparableConv1D(14, 32, kernel_size=3, stride=1, depth_multiplier=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            SeparableConv1D(32, 64, kernel_size=2, stride=1, depth_multiplier=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            SeparableConv1D(64, 64, kernel_size=3, stride=1, depth_multiplier=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SeparableConv1D(64, 64, kernel_size=3, stride=1, depth_multiplier=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128, 100),
            nn.Sigmoid()
        )

        self.l1_layer = LambdaLayer(lambda tensors: torch.abs(tensors[0] - tensors[1]))
        self.dropout = nn.Dropout(p=0.5)
        self.prediction = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        encoded_l = self.convnet(x1)
        encoded_r = self.convnet(x2)

        L1_distance = self.l1_layer([encoded_l, encoded_r])
        D1_layer = self.dropout(L1_distance)
        prediction = self.prediction(D1_layer)
        return prediction

class SiameseNet2(nn.Module):
    def __init__(self):
        super(SiameseNet2, self).__init__()
        
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=64, stride=16, padding=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(192, 100),
            nn.Sigmoid()
        )
        
        self.l1_layer = LambdaLayer(lambda tensors: torch.abs(tensors[0] - tensors[1]))
        self.dropout = nn.Dropout(p=0.5)
        self.prediction = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        encoded_l = self.convnet(x1)
        encoded_r = self.convnet(x2)

        L1_distance = self.l1_layer([encoded_l, encoded_r])
        D1_layer = self.dropout(L1_distance)
        prediction = self.prediction(D1_layer)
        return prediction

class WDCNN(nn.Module):
    def __init__(self, nclasses=10):
        super(WDCNN, self).__init__()
        self.convnet = nn.Sequential(
            SeparableConv1D(in_channels=2, out_channels=14, kernel_size=64, stride=8, depth_multiplier=29),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            SeparableConv1D(in_channels=14, out_channels=32, kernel_size=3, stride=1, depth_multiplier=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            SeparableConv1D(in_channels=32, out_channels=64, kernel_size=2, stride=1, depth_multiplier=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            SeparableConv1D(in_channels=64, out_channels=64, kernel_size=3, stride=1, depth_multiplier=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SeparableConv1D(in_channels=64, out_channels=64, kernel_size=3, stride=1, depth_multiplier=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128, 100),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(100, nclasses),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.convnet(x)
        return x

class WDCNN2(nn.Module):
    def __init__(self, input_shape=(2048, 2), nclasses=10):
        super(WDCNN2, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[1], out_channels=16, kernel_size=64, stride=9, padding=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(384, 100),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(100, nclasses),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.convnet(x)
        return x
    
