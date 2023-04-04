"""
Some Layers, really important to the process effect of the Network.

Author: Bruce Hou, Email: ecstayalive@163.com
"""
import torch
import torch.fft
from torch import nn


class TFLayer(nn.Module):
    """
    Time Frequency conversion
    """

    def __init__(self):
        super(TFLayer, self).__init__()

    def forward(self, x):
        # 傅里叶变换
        outputs = torch.abs(torch.fft.fft(x, dim=-1, norm="forward"))

        outputs = outputs.view(outputs.size()[0], 64, -1)
        outputs = outputs.unsqueeze(1)

        return outputs


class WaveletLayer(nn.Module):
    """
    小波全连接神经网络
    """

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
        y = self.WaveletFunction(y)  # 小波基函数

        # TODO(ecstayalive@163.com): 确定之后的运算和更新过程
        y = torch.matmul(self.weight2, y) + self.bias2  # 信号重构
        # print(y.size())
        # y = y.view(y.size()[0], 32, -1)
        y = y.unsqueeze(1)

        return y
    
    def WaveletFunction(self, inputs):
        """
        小波函数
        :param inputs 输入变量
        """
        outputs = torch.mul(
            torch.cos(torch.mul(inputs, 1.75)),
            torch.exp(torch.mul(-0.5, torch.mul(inputs, inputs))),
        )
        return outputs
