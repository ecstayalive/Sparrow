import numpy as np
import torch
from capsule_net.capsule_net import CapsuleNet
from capsule_net.utils import add_gaussian_noise, PlotData, trainer
from sklearn.model_selection import KFold


def load_dataset(train_dataset, test_dataset, batch_size=10):
    """加载数据
    Args:
        train_dataset:
        test_dataset:
        batch_size: batch size

    Returns:
        train_loader, test_loader
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


class VibrationDataset(torch.utils.data.Dataset):
    """创建数据集"""

    def __init__(self, data: np.ndarray, label: np.ndarray, classes: int) -> None:
        self.data = torch.unsqueeze(torch.tensor(data), 1)
        label = torch.tensor(label)
        self.label = torch.zeros(label.shape[0], classes).scatter_(
            -1, label.view(-1, 1), 1.0
        )

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, index):
        return self.data[index], self.label[index]


if __name__ == "__main__":
    ########################################################################
    # User-code
    data, label = np.load("dataset/train.npy"), np.load("dataset/label.npy")
    classes = 8
    # User-code
    ########################################################################
    # process the data and the label
    data = np.float32(data)
    # add noise to signal
    # data = add_gaussian_noise(data, 1.0)
    label = np.int64(label)
    signal_length = data.shape[1]
    # Plot data
    plot_data = PlotData()
    # plot an original signal
    plot_data.plot_signal(data[0])
    # plot an signal after fft process
    plot_data.plot_fft_signal(data[0])
    # plot a spectrogram of the signal
    plot_data.plot_spectrogram(data[0])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 四折交叉验证
    kf = KFold(n_splits=4)
    for id, (train_index, test_index) in enumerate(kf.split(data)):
        # 生成数据集
        train_data = data[train_index]
        train_label = label[train_index]
        test_data = data[test_index]
        test_label = label[test_index]
        # 建立数据集合
        train_dataset = VibrationDataset(train_data, train_label, classes)
        test_dataset = VibrationDataset(test_data, test_label, classes)
        train_loader, test_loader = load_dataset(
            train_dataset, test_dataset, batch_size=12
        )
        # 创建模型
        model = CapsuleNet(
            input_features=(1, signal_length),
            classes=classes,
            routings=3,
            device=device,
        )
        print(model)
        # 训练
        trainer(model, train_loader, test_loader, 15, id, device)
    # plot accuracy and loss
    plot_data.plot_log()
