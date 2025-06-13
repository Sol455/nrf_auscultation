import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfreqz

def design_bandpass_iir(fs, low_cutoff, high_cutoff, order=8):

    sos = butter(order, [low_cutoff, high_cutoff], btype='bandpass', fs=fs, output='sos')
    return sos

def design_lowpass_iir(fs, cutoff=20, order=3):
    sos = butter(order, cutoff, btype='low', fs=fs, output='sos')
    return sos

def plot_filter_response(sos, fs, title="Filter Frequency Response"):

    w, h = sosfreqz(sos, worN=2048, fs=fs)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10)) 

    plt.figure(figsize=(10, 4))
    plt.plot(w, magnitude_db, label="Magnitude Response")
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xscale('log')
    plt.xlim(10, fs / 2) 
    plt.ylim(-100, 5)
    plt.xticks([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 8000])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.legend()
