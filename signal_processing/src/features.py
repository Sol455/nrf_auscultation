import numpy as np
import matplotlib.pyplot as plt

def detect_peaks_npointaverage(signal, threshold_scale=1.5, min_peak_distance=100):
    locs = []
    peaks = []
    last_peak = -np.inf

    threshold = np.mean(signal) * threshold_scale

    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if len(locs) == 0 or (i - last_peak) >= min_peak_distance:
                locs.append(i)
                peaks.append(signal[i])
                last_peak = i

    return locs, peaks

def plot_peaks(signal, fs, locs, peaks, title="Peak Detection", xlim=None):
    t = np.arange(len(signal)) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, label="Signal", alpha=0.6)
    plt.plot(np.array(locs) / fs, peaks, 'rx', label="Detected Peaks", markersize=8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    if xlim:
        plt.xlim(xlim)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()