import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os

def load_wav(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    signal, samplerate = sf.read(filepath)
    
    # If stereo, convert to mono
    if signal.ndim == 2:
        signal = np.mean(signal, axis=1)
    
    return signal, samplerate

def plot_waveform(signal, fs, title="Audio Waveform", xlim=None):
    t = np.arange(len(signal)) / fs


    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, label="Waveform", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    if xlim:
        plt.xlim(xlim)
    plt.grid(True)
    plt.tight_layout()

