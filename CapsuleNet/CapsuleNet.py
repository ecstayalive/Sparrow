"""
Degenerate capsule neural network.
Reference Papers: https://arxiv.org/abs/1710.09829
Reference Code： https://github.com/XifengGuo/CapsNet-Pytorch

Author: Bruce Hou, Email: ecstayalive@163.com
"""

import torch
import torch.fft
from torch import nn
from .CapsuleLayer import DenseCapsule, PrimaryCapsule
from .PreLayers import TFLayer


class CapsuleNet(nn.Module):
    """
    A Degenerate Capsule Neural Network.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """

    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        self.tfLayer = TFLayer()

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=3, stride=1, padding=0)

        self.maxPool = nn.MaxPool2d(3)
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primaryCaps = PrimaryCapsule(
            256, 256, 8, kernel_size=9, stride=2, padding=0
        )

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitCaps = DenseCapsule(
            in_num_caps=6 * 6 * 32,
            in_dim_caps=8,
            out_num_caps=classes,
            out_dim_caps=1,
            routings=routings,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.tfLayer(x)
        # print("经过傅里叶变换层后的x形状", x.size())
        x = self.relu(self.conv1(x))
        # print("经过巻积层1后的x形状", x.size())
        x = self.maxPool(x)
        # print("经过池化层后的x形状", x.size())
        x = self.primaryCaps(x)
        # print("经过初始胶囊层后的x形状", x.size())
        x = self.digitCaps(x)
        # print("经过胶囊层后的x形状", x.size())
        length = x.norm(dim=-1)
        return length


if __name__ == "__main__":
    pass
