import torch.nn as nn
import torch.nn.functional as F


class SimpleFlatter(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)

    def eval(self):
        pass


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.block4 = nn.Sequential(nn.Conv2d(128, 128, (1, 1)))
        self.last_relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.last_relu(x)

        x = self.avg_pool(x)
        x = x.view(-1, 128)

        return F.normalize(x)


class Classifier(nn.Module):
    def __init__(self, in_feature_size=128, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(in_feature_size, num_classes)

    def forward(self, x):
        return self.classifier(x)
