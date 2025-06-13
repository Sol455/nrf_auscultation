import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfilt

from src.utils import load_wav, plot_waveform
from src.filters import design_bandpass_iir, design_lowpass_iir, plot_filter_response, export_sos_to_cmsis_header
from src.features import detect_peaks_npointaverage, plot_peaks, extract_rms_around_peaks


wav_path = "data/MAY.wav"
signal, fs = load_wav(wav_path)

#Parameters
rms_window_size = 3200

#BandPass Filter
bp_low_cut = 30
bp_high_cut = 150
bp_order = 8
sos_bandpass = design_bandpass_iir(fs, bp_low_cut, bp_high_cut, bp_order)

lp_high_cut = 20
lp_order = 3
#Low-pass smoothing Filter
sos_lowpass = design_lowpass_iir(fs, lp_high_cut, lp_order)


#===================================DSP Chain==========================================

bandpass_audio = sosfilt(sos_bandpass, signal)

abs_signal = np.abs(bandpass_audio)

smoothed_envelope = sosfilt(sos_lowpass, abs_signal)

locs, peaks = detect_peaks_npointaverage(smoothed_envelope, 3, 4000)

rms_values = extract_rms_around_peaks(bandpass_audio, locs, rms_window_size, 0)
print(rms_values)


#======================================================================================

#Plots
#plot_filter_response(sos_bandpass, fs, title="8th-Order Bandpass Filter (30–150 Hz)")
#plot_filter_response(sos_lowpass, fs, title="3rd-Order Lowpass IIR Filter 20Hz Cutoff")
plot_waveform(smoothed_envelope, fs, "Smoothed Envelope")
#plot_waveform(bandpass_audio, fs, "BandPass Filtered Audio")
plot_peaks(bandpass_audio, fs, locs, peaks, rms_values, title="Peaks and rms values")

#Export Filter co-effiecients
export_sos_to_cmsis_header(sos_bandpass, "output/bandpass_coeffs", "bandpass_coeffs")
export_sos_to_cmsis_header(sos_lowpass, "output/lowpass_coeffs", "lowpass_coeffs")

plt.show()