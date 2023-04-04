"""
Test method for the DeCapsule Network.
Reference Papers: https://arxiv.org/abs/1710.09829
Reference Code： https://github.com/XifengGuo/CapsNet-Pytorch

Author: Bruce Hou, Email: ecstayalive@163.com
"""
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
import torch
from .Loss import caps_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for x, y in test_loader:
        # 由于nn.conv2d输入为4-d向量(batch_size, channels, width, height)
        x = x.unsqueeze(1)
        # one-hot编码
        y = torch.zeros(y.size(0), 8).scatter_(1, y.view(-1, 1), 1.0)

        with torch.no_grad():
            x, y = Variable(x.cuda()), Variable(y.cuda())

        y_pred = model(x)
        test_loss += caps_loss(y, y_pred).item() * x.size(0)  # sum up batch loss
        # print(y_pred.data)
        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]
        # print("y_pred:", y_pred.data)
        # print("y_true:", y_true)
        correct += y_pred.eq(y_true).cpu().sum()

    # print("correct:", correct)
    # print("length:", len(test_loader.dataset))

    test_loss = torch.true_divide(test_loss, len(test_loader.dataset))
    acc = torch.true_divide(correct, len(test_loader.dataset))

    print("acc:", acc)
    print("test loss:", test_loss)
    return test_loss.item(), acc.item()
