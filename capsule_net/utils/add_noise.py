"""
add noise signal by need
"""
import numpy as np


def add_gaussian_noise(signal, snr):
    """Add noise to signal

    Args:
        signal: 原始信号
        snr: 添加噪声的信噪比

    Returns:
        含有噪声的信号
    """
    signal_len = signal.shape[1]
    num_signal = signal.shape[0]
    signal_power = np.sum(np.power(signal, 2), 1) / signal_len
    signal_db = 10 * np.log10(signal_power)
    # generate noise signal
    noise_db = signal_db - snr
    noise_power = 10 ** (noise_db / 10)
    noise_mean = np.zeros(noise_power.shape)

    signal_with_noise = signal.copy()
    for idx in range(num_signal):
        noise = np.random.normal(noise_mean[idx], noise_power[idx], size=(signal_len,))
        signal_with_noise[idx, :] += noise

    return signal_with_noise
