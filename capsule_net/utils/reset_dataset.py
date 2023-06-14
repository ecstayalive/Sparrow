"""
This project is derived from the course work
and is an extension of the course work. Due
to the source of the dataset itself, the
dataset needs to be pre-processed before it
can be called.
"""


import itertools

import numpy as np
import scipy.io as scio


class ResetDataset:
    """
    加载数据并保存成npy格式
    """

    def __init__(self):
        pass

    def run(self):
        """
        调用接口
        :return
        """
        data = self.load()
        # 数据集大小
        train_data = np.empty((640, 4096))
        label = np.empty((640,))
        # 转换数据格式并保存
        train_data, label = self.transform_data(data, train_data, label)

    def load(self):
        """
        加载数据
        :return 加载的数据
        """
        return scio.loadmat("Dataset/lecture_data.mat")

    def transform_data(self, data, train, label):
        """
        改变格式，生成数据集并保存
        :param data 加载的mat数据
        :param train 需要的数据格式和形状
        :param label 数据对应的标签
        :return train, label
        """
        temp1 = np.empty((8, 4096, 80))

        temp1[0] = data["class0_train_normal"]
        temp1[1] = data["class1_train_inner"]
        temp1[2] = data["class2_train_outer"]
        temp1[3] = data["class3_train_roller"]
        temp1[4] = data["class4_train_crack"]
        temp1[5] = data["class5_train_pitting"]
        temp1[6] = data["class6_train_broken_tooth"]
        temp1[7] = data["class7_train_missing_tooth"]

        # 生成train和label数据集
        for i, j in itertools.product(range(8), range(80)):
            train[i * 80 + j, :] = temp1[i, :, j]
            label[i * 80 + j] = i

        # 打乱训练集和标签
        permutation = np.random.permutation(label.shape[0])
        print(permutation)
        train = train[permutation, :]
        label = label[permutation]

        # 保存数据
        np.save("Dataset/train.npy", train)
        np.save("Dataset/label.npy", label)

        return train, label
