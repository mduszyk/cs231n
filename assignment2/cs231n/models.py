import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        # read in N, C, H, W
        N, C, H, W = x.size()
        # "flatten" the C * H * W values into a single vector per image
        return x.view(N, -1)


lenet_model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    Flatten(),

    nn.Linear(in_features=400, out_features=120),
    nn.ReLU(inplace=True),

    nn.Linear(in_features=120, out_features=84),
    nn.ReLU(inplace=True),

    nn.Linear(in_features=84, out_features=10)
)
LeNet = ("LeNet", lenet_model)
