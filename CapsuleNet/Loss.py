"""
Loss function
Reference Papers: https://arxiv.org/abs/1710.09829
Reference Code： https://github.com/XifengGuo/CapsNet-Pytorch

Author: Bruce Hou, Email: ecstayalive@163.com
"""
import torch

def caps_loss(y_true, y_pred):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :return: Variable contains a scalar loss value.
    """

    L = (
        y_true * torch.clamp(0.9 - y_pred, min=0.0) ** 2
        + 0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.0) ** 2
    )
    L_margin = L.sum(dim=1).mean()

    return L_margin