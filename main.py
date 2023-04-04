"""
程序调用入口

Author: Bruce Hou, Email: ecstayalive@163.com
"""

import torch
from CapsuleNet.CapsuleNet import *
from CapsuleNet.Test import test
from CapsuleNet.Train import train
from CapsuleNet.Util.Plot import PlotData
from CapsuleNet.Util.AddNoise import add_gaussian_noise


def Load(train_dataset, test_dataset, batch_size=10):
    """
    加载数据
    :param train_dataset:
    :param test_dataset:
    :param batch_size: batch size
    :return: train_loader, test_loader
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


class CreDataset(torch.utils.data.Dataset):
    """
    创建数据集
    """

    def __init__(self, data, label):
        import numpy as np

        self.data = torch.from_numpy(np.float32(data))
        self.label = torch.from_numpy(np.int64(label))

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, index):
        return self.data[index], self.label[index]


if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import KFold

    # load data
    data, label = np.load("Dataset/train.npy"), np.load("Dataset/label.npy")
    # data = add_gaussian_noise(data, 1)
    # Plot data
    plotdata = PlotData()
    # for example
    plotdata.plot_signal(data[0]) # plot an orginal signal
    plotdata.plot_signalByFFT(data[0])  # plot an signal after fft process
    plotdata.plot_spectrogram(data[0]) # plot a spectrogram of the signal
    plotdata.plot_inputImage(data[0])  # plot an input image

    print("train.shape:", data.shape)
    print("label.shape:", label.shape)

    # 四折交叉验证
    kf = KFold(n_splits=4)
    count = 0
    for train_index, test_index in kf.split(data):
        # 生成数据集
        train_data = data[train_index]
        train_label = label[train_index]
        test_data = data[test_index]
        test_label = label[test_index]
        # 建立数据集合
        train_dataset = CreDataset(train_data, train_label)
        test_dataset = CreDataset(test_data, test_label)
        train_loader, test_loader = Load(train_dataset, test_dataset, batch_size=10)

        # 创建模型
        model = CapsuleNet(input_size=[1, 4096], classes=8, routings=3)
        print(model)
        model.cuda()
        # 训练
        train(model, train_loader, test_loader, count, 30)
        count += 1

    
    # plot accuracy and loss
    plotdata.plot_log()