import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft


class PlotData:
    """
    Plot all we need.
    """

    def __init__(self):
        pass

    def plot_signal(
        self, signal: np.ndarray, show_image: bool = True, save_image: bool = True
    ) -> None:
        """
        Plot the signal
        """
        # load data
        signal = np.array(signal)
        signal_length = len(signal)
        # plot
        fig = plt.figure(1)
        plt.plot(np.arange(signal_length), signal)
        if save_image:
            fig.savefig("docs/pics/signal.png")
        if show_image:
            plt.show()

    def plot_fft_signal(
        self, signal: np.ndarray, show_image: bool = True, save_image: bool = True
    ) -> None:
        """
        Plot the data information after using fft to process the signal
        """
        signal_length = len(signal)
        # load data
        signal_by_fft = np.abs(fft(np.array(signal))) / signal_length
        # plot
        fig = plt.figure(1)
        plt.plot(np.arange(signal_length), signal_by_fft)
        if save_image:
            fig.savefig("docs/pics/signal_by_fft.png")
        if show_image:
            plt.show()

    def plot_spectrogram(
        self,
        signal: np.ndarray,
        sample_frequency: int = 25600,
        show_image: bool = True,
        save_image: bool = True,
    ) -> None:
        """
        Plot the spectrogram, using the short-time Fourier transform
        """
        # load data
        signal = np.array(signal)
        # plot
        fig = plt.figure(1)
        plt.specgram(signal, Fs=sample_frequency)
        if save_image:
            fig.savefig("docs/pics/spectrogram.png")
        if show_image:
            plt.show()

    def plot_log(self, dir="log/", show_image=True, save_image=True):
        """
        Plot the log file
        To show the accuracy of the model
        """
        # load data
        file_name_list = os.listdir(dir)
        file_num = len(file_name_list)
        with open(f"{dir}{file_name_list[0]}", "rb") as f:
            log_file = pickle.load(f)

        for file_name in file_name_list[1:]:
            with open(f"{dir}{file_name}", "rb") as f:
                temp_log_file = pickle.load(f)
            for key in log_file.keys():
                log_file[key] = log_file[key] + temp_log_file[key]
        # get the average of the data
        for key in log_file.keys():
            log_file[key] /= file_num

        epochs = len(log_file["train_loss"])
        x = np.arange(1, epochs + 1)
        fig = plt.figure(figsize=(10, 12))
        fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
        fig.add_subplot(211)
        plt.plot(x, log_file["train_loss"], label="train_loss")
        plt.plot(x, log_file["test_loss"], label="test_loss")
        plt.legend()
        plt.title("Loss")

        fig.add_subplot(212)
        plt.plot(x, log_file["train_accuracy"], label="train_accuracy")
        plt.plot(x, log_file["test_accuracy"], label="test_accuracy")
        plt.grid()
        plt.legend()
        plt.title("Accuracy")
        if save_image:
            fig.savefig("docs/pics/accuracy_and_loss.png")
        if show_image:
            plt.show()
