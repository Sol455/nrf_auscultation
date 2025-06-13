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

def extract_rms_around_peaks(signal, peak_locs, window_total, gd_offset=0):

    half_window = window_total // 2
    rms_values = []

    for loc in peak_locs:
        compensated_loc = loc - gd_offset
        compensated_loc = max(0, min(len(signal) - 1, compensated_loc))

        start = max(0, compensated_loc - half_window)
        end = min(len(signal), compensated_loc + half_window)
        window = signal[start:end]

        rms = np.sqrt(np.mean(window ** 2))
        rms_values.append(rms)

    return np.array(rms_values)

import matplotlib.pyplot as plt
import numpy as np

def plot_peaks(signal, fs, locs, peaks, rms_values=None, title="Peak Detection", xlim=None):
    t = np.arange(len(signal)) / fs
    peak_times = np.array(locs) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, label="Signal", alpha=0.6)
    plt.plot(peak_times, peaks, 'rx', label="Detected Peaks", markersize=8)

    if rms_values is not None:
        plt.plot(peak_times, rms_values, 'go', label="RMS Values", markersize=6)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    if xlim:
        plt.xlim(xlim)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
