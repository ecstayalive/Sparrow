import math

import torch
import torch.fft
import torch.nn as nn
from capsule_nn.functional import squash
from torch import Tensor


class TFLayer(nn.Module):
    """Time Frequency conversion layer"""

    def __init__(self, device=None, dtype=None) -> None:
        factor_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        # 傅里叶变换
        outputs = torch.abs(torch.fft.fft(input, dim=-1, norm="forward"))
        outputs = outputs.view(outputs.shape[0], 1, -1)
        return outputs


class PrimaryCapsule(nn.Module):
    """Apply Conv2D with `out_channels` and then reshape to get capsules

    Args:
        in_channels: input channels
        out_channels: output channels
        dim_caps: dimension of capsule
        kernel_size: kernel size

    Returns:
        output tensor, size=[batch, num_caps, dim_caps]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim_caps: int,
        kernel_size: int,
        stride=1,
        padding=0,
        device=None,
    ):
        super().__init__()
        self.dim_caps = dim_caps
        self.conv2d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            device=device,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv2d(input)
        output = output.view(input.shape[0], -1, self.dim_caps)
        output = squash(output)
        return output


class WaveletLayer(nn.Module):
    """ """

    # TODO(ecstayalive@163.com): How can i use the wave transform to make the neural network more robust?

    def __init__(self):
        super(WaveletLayer, self).__init__()
        # 第一层权重和偏置
        self.weight1 = nn.Parameter(0.01 * torch.randn(74, 2048))
        self.bias1 = nn.Parameter(0.01 * torch.randn(74, 1))
        # 偏移和尺度
        self.a = nn.Parameter(0.01 * torch.randn(74, 1))
        self.b = nn.Parameter(0.01 * torch.randn(74, 1))
        # 第二层权重和偏置
        self.weight2 = nn.Parameter(0.01 * torch.randn(2048, 74))
        self.bias2 = nn.Parameter(0.01 * torch.randn(2048, 1))

    # TODO(ecstayalive@163.com): How to built a neural network layer by using wave transform?
    def forward(self, x):
        y = (
            torch.matmul(self.weight1, x.view(x.size(0), 2048, -1)) + self.bias1
        )  # 矩阵相乘+偏置
        y = torch.div(y - self.b, self.a)  # 偏移因子和尺度因子
        y = self.wavelet_fn(y)  # 小波基函数

        # TODO(ecstayalive@163.com): 确定之后的运算和更新过程
        y = torch.matmul(self.weight2, y) + self.bias2  # 信号重构
        # print(y.size())
        # y = y.view(y.size()[0], 32, -1)
        y = y.unsqueeze(1)

        return y

    def wavelet_fn(self, inputs):
        """
        小波函数
        :param inputs 输入变量
        """
        # sourcery skip: inline-immediately-returned-variable
        outputs = torch.mul(
            torch.cos(torch.mul(inputs, 1.75)),
            torch.exp(torch.mul(-0.5, torch.mul(inputs, inputs))),
        )
        return outputs
