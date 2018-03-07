import torch.nn as nn


class Flatten(nn.Module):

    def forward(self, x):
        # read in N, C, H, W
        N, C, H, W = x.size()
        # "flatten" the C * H * W values into a single vector per image
        return x.view(N, -1)


def lenet():
    return nn.Sequential(
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


def conv_relu_pool():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 16x16x32

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 8x8x64

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 4x4x128

        Flatten(),

        nn.Linear(in_features=2048, out_features=1000),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=1000, out_features=500),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=500, out_features=10)
    )


def conv_relu_pool_2():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 16x16x64

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 8x8x128

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 4x4x256

        Flatten(),

        nn.Linear(in_features=4096, out_features=2048),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=2048, out_features=1024),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=1024, out_features=10)
    )


def conv_relu_pool_3():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 16x16x64

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 8x8x128

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 4x4x256

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 2x2x512

        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 1x1x1024

        Flatten(),

        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=1024, out_features=10)
    )


def conv_relu_pool_4():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 16x16x128

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 8x8x256

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 4x4x512

        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 2x2x1024

        Flatten(),

        nn.Linear(in_features=4096, out_features=2048),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=2048, out_features=1024),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=1024, out_features=10)
    )


def conv_relu_pool_4_dr(dropout=0.5):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 16x16x128

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 8x8x256

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 4x4x512

        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 2x2x1024

        Flatten(),

        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=4096, out_features=2048),
        nn.ReLU(inplace=True),

        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=2048, out_features=1024),
        nn.ReLU(inplace=True),

        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=1024, out_features=10)
    )


def conv_relu_pool_4_dr_2(dropout=0.5):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 16x16x128

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 8x8x256

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 4x4x512

        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 2x2x1024

        Flatten(),

        nn.Linear(in_features=4096, out_features=2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout, inplace=True),

        nn.Linear(in_features=2048, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout, inplace=True),

        nn.Linear(in_features=1024, out_features=10)
    )


def conv_stride_3():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # 16x16x64

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # 8x8x128

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # 4x4x256

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # 2x2x512

        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        # 1x1x1024

        Flatten(),

        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=1024, out_features=10)
    )


def conv_conv_pool():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 16x16x64

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 8x8x128

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 4x4x256

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 2x2x512

        Flatten(),

        nn.Linear(in_features=2048, out_features=1024),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=1024, out_features=10)
    )


def conv_conv_pool_2():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 16x16x64

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 8x8x128

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 4x4x256

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 2x2x512

        Flatten(),

        nn.Linear(in_features=2048, out_features=1024),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=1024, out_features=10)
    )


def conv_conv_pool_2_dr(dropout=0.5):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 16x16x64

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 8x8x128

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 4x4x256

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # 2x2x512

        Flatten(),
        nn.Dropout(p=dropout, inplace=True),

        nn.Linear(in_features=2048, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout, inplace=True),

        nn.Linear(in_features=1024, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout, inplace=True),

        nn.Linear(in_features=1024, out_features=10)
    )


def conv_conv_pool_3():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2),
        # 16x16x64

        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2),
        # 8x8x128

        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2),
        # 4x4x256

        Flatten(),

        nn.Linear(in_features=8192, out_features=2048),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=2048, out_features=2048),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=2048, out_features=10)
    )
