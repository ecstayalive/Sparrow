"""
Plot all we need
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import pi
from colorsys import hls_to_rgb
import csv
import math
from scipy.fftpack import fft, ifft

class PlotData:
    def __init__(self):
        pass
    
    def plot_signal(self, signal, isShow=True, isSave=True, isPyqtGraph=False):
        """
        Plot the signal
        """
        # load data
        signal = np.array(signal)
        # plot
        fig = plt.figure(1)
        plt.plot(np.arange(len(signal)), signal)
        if isSave:
            fig.savefig('Docs/DataInfo/signal.png')
        if isShow:
            plt.show()
    
    def plot_signalByFFT(self, signal, isShow=True, isSave=True, isPyqtGraph=False):
        """
        Plot the data infomation after using fft to process the signal
        """
        # load data
        signal_by_fft = np.abs(fft(np.array(signal))) / len(signal)
        # plot
        fig = plt.figure(1)
        plt.plot(np.arange(len(signal_by_fft)), signal_by_fft)
        if isSave:
            fig.savefig('Docs/DataInfo/signal_by_fft.png')
        if isShow:
            plt.show()

    def plot_spectrogram(self, signal, isShow=True, isSave=True, isPyqtGraph=False):
        """
        Plot the spectrogram, using the short-time Fourier transform
        """
        # load data
        signal = np.array(signal)
        # plot
        fig = plt.figure(1)
        plt.specgram(signal, Fs=25600)
        if isSave:
            fig.savefig('Docs/DataInfo/spectrogram.png')
        if isShow:
            plt.show()
    
    def plot_inputImage(self, signal, isShow=True, isSave=True, isPyqtGraph=False):
        """
        Transform the signal to the shape of the input image and show it
        """
        # load data
        z = fft(np.array(signal) / len(signal))
        image = np.abs(np.reshape(z, newshape=(-1, 64)))
        # r = np.abs(z)
        # arg = np.angle(z)

        # plot
        fig = plt.figure(1)
        plt.imshow(image, cmap = plt.cm.jet)
        plt.colorbar()
        if isSave:
            fig.savefig('Docs/DataInfo/input_image.png')
        if isShow:
            plt.show()

    def plot_origin_signal(self, isShow=True, isSave=True, isPyqtGraph=False):
        """
        Plot the origin signal
        """
        # load data
        train_signal, label = (np.load("Dataset/train.npy"), np.load("Dataset/label.npy"))
        # plot
        fig = plt.figure(1, figsize=(48, 24))
        for i in range(0, 6):
            ax = plt.subplot(3, 2, i + 1)
            ax.set_title(str(label[i]))
            plt.plot(np.arange(4096), train_signal[i, :])
        if isSave:
            fig.savefig('Docs/DataInfo/origin_signal.png')
        if isShow:
            plt.show()
    
    

    def plot_log(self, filename="Result/log0.csv", isShow=True, isSave=True, isPyqtGraph=False):
        """
        Plot the log file
        To show the accuracy of the model
        """
        # load data
        keys = []
        values = []
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if keys == []:
                    for key, value in row.items():
                        keys.append(key)
                        values.append(float(value))
                    continue

                for _, value in row.items():
                    values.append(float(value))

            values = np.reshape(values, newshape=(-1, len(keys)))

        fig = plt.figure(figsize=(10, 12))
        fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
        fig.add_subplot(211)
        epoch_axis = 0
        for i, key in enumerate(keys):
            if key == "epoch":
                epoch_axis = i
                values[:, epoch_axis] += 1
                break
        for i, key in enumerate(keys):
            if key.find("loss") >= 0:  # loss
                print(values[:, i])
                plt.plot(values[:, epoch_axis], values[:, i], label=key)
        plt.legend()
        plt.title("Training loss")

        fig.add_subplot(212)
        for i, key in enumerate(keys):
            if key.find("acc") >= 0:  # acc
                plt.plot(values[:, epoch_axis], values[:, i], label=key)
        plt.legend()
        plt.grid()
        plt.title("Accuracy")
        if isSave:
            fig.savefig('Docs/DataInfo/accuracy_and_loss.png')
        if isShow:
            plt.show()


    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
        shape = generated_images.shape[1:3]
        image = np.zeros(
            (height * shape[0], width * shape[1]), dtype=generated_images.dtype
        )
        for index, img in enumerate(generated_images):
            i = int(index / width)
            j = index % width
            image[
                i * shape[0] : (i + 1) * shape[0], j * shape[1] : (j + 1) * shape[1]
            ] = img[:, :, 0]
        return image

if __name__ == "__main__":
    dataInfo = PlotData()
    dataInfo.plot_log("Result/log0.csv")

