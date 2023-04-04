"""
Train method for the DeCapsule Network.
Reference Papers: https://arxiv.org/abs/1710.09829
Reference Code： https://github.com/XifengGuo/CapsNet-Pytorch

Author: Bruce Hou, Email: ecstayalive@163.com
"""
import torch
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from .Loss import caps_loss
from .Test import test


def train(model, train_loader, test_loader, count, epoch):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param count
    :param epoch
    :return: The trained model
    """
    print("Begin Training" + "-" * 70)
    from time import time
    import csv

    logfile = open("Result/log" + str(count) + ".csv", "w")
    logwriter = csv.DictWriter(
        logfile, fieldnames=["epoch", "loss", "val_loss", "val_acc"]
    )
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=0.001)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    best_val_acc = 0

    optimizer.step()  # 消除Warning

    for epoch in range(epoch):
        model.train()  # set to training mode
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(train_loader):  # batch training
            # 由于nn.conv2d输入为4-d向量(batch_size, channels, width, height)
            # x = x.unsqueeze(1)  # 在某一个方向上增加一维
            # print("初始x输入形状", x.shape)

            # change to one-hot coding（改变标签使其变成one-hot编码）
            y = torch.zeros(y.size(0), 8).scatter_(1, y.view(-1, 1), 1.0)

            x, y = (
                Variable(x.cuda()),
                Variable(y.cuda()),
            )  # convert input data to GPU Variable

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred = model(x)  # forward
            # print("输出的y", y_pred)
            loss = caps_loss(y, y_pred)  # compute loss

            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            training_loss += loss.item() * x.size(0)  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients
        # compute validation loss and acc
        val_loss, val_acc = test(model, test_loader)
        logwriter.writerow(
            dict(
                epoch=epoch,
                loss=training_loss / len(train_loader.dataset),
                val_loss=val_loss,
                val_acc=val_acc,
            )
        )
        print(
            "==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
            % (
                epoch,
                training_loss / len(train_loader.dataset),
                val_loss,
                val_acc,
                time() - ti,
            )
        )
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            torch.save(model.state_dict(), "Model/epoch%d.pkl" % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)
    logfile.close()
    torch.save(model.state_dict(), "Model/trained_model.pkl")
    print("Trained model saved to 'Model/trained_model.h5'")
    print("Total time = %ds" % (time() - t0))
    return model
    print("End Training" + "-" * 70)
