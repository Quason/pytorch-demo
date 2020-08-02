import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            # suppose img size is 32*32*3
            # layer1
            nn.Conv2d(3, 6, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # layer2
            nn.Conv2d(6, 16, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # layer6
            nn.Dropout(),
            nn.Linear(6*6*16, 256),
            nn.ReLU(inplace=True),
            # layer7
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # layer8
            nn.Linear(256, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
