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

def compute_rms_at_peak(signal, peak_loc, window_total):

    half_window = window_total // 2
    start = max(0, peak_loc - half_window)
    end = min(len(signal), peak_loc + half_window)
    window = signal[start:end]

    return np.sqrt(np.mean(window ** 2))

def calc_fft_db(signal, fs, plot=False, title=None):
    signal = np.asarray(signal)
    N = len(signal)
    window = np.hanning(N)
    windowed = signal * window

    X = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(N, d=1/fs)
    X_dB = 20 * np.log10(X + np.finfo(float).eps)

    # Spectral centroid based on magnitude
    centroid = np.sum(freqs * X) / (np.sum(X) + 1e-12)

    return freqs, X_dB, centroid

def extract_features_around_peaks(signal, peak_locs, window_total, fs, gd_offset=0, plot=False):
    rms_values = []
    centroids = []
    half_window = window_total // 2

    if plot:
        plt.figure(figsize=(10, 4))

    for i, loc in enumerate(peak_locs):
        adjusted_loc = loc - gd_offset
        adjusted_loc = max(0, min(len(signal) - 1, adjusted_loc))

        start = max(0, adjusted_loc - half_window)
        end = min(len(signal), adjusted_loc + half_window)
        windowed = signal[start:end]

        # RMS
        rms = np.sqrt(np.mean(windowed ** 2))
        rms_values.append(rms)

        # FFT + Centroid
        window = np.hanning(len(windowed))
        X = np.abs(np.fft.rfft(windowed * window))
        freqs = np.fft.rfftfreq(len(windowed), d=1/fs)
        centroid = np.sum(freqs * X) / (np.sum(X) + 1e-12)
        centroids.append(centroid)

        if plot:
            db = 20 * np.log10(X + np.finfo(float).eps)
            plt.plot(freqs, db, alpha=0.5, label=f"Peak {i+1}")
            plt.axvline(centroid, color='r', linestyle='--', linewidth=0.8)

    if plot:
        plt.xlim(20, 200)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("FFT around Peaks with Spectral Centroids")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return np.array(rms_values), np.array(centroids)
