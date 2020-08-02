import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            # suppose img size is 32*32*3
            # layer1
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # layer2
            nn.Conv2d(64, 192, 3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # layer3
            nn.Conv2d(192, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # layer4
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            # layer5
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.classifier = nn.Sequential(
            # layer6
            nn.Dropout(),
            nn.Linear(2*2*256, 4096),
            nn.ReLU(inplace=True),
            # layer7
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # layer8
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
