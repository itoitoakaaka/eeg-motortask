import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import welch

# Paths
RAW_PATH = '/Users/itoakane/Research/Analyzer/rawdata/id0010001.eeg'
FILT_PATH = '/Users/itoakane/Research/Analyzer/id0010001_Filters_3-1000_50_US.eeg'

# Load parameters from VHDR (Parsed manually or assumed from previous cats)
# Raw: 6 channels (C3, CP1, CP3, CP5, P3, EOG), Multiplexed, float32, 2500Hz
# Filtered: 2 channels (C3, CP3)? Let's check VHDR again.
# id0010001_Filters_3-1000_50_US.vhdr said NumberOfChannels=2 (C3, CP3).
# So we compare C3 from Raw vs C3 from Filtered.

SR = 2500

def load_eeg(path, n_ch):
    data = np.fromfile(path, dtype='<f4')
    n_samples = len(data) // n_ch
    # Apply Resolution to Raw Data to match Filtered (uV)
    # Raw is float32 but values are ~300. Filtered is ~1.5.
    # VHDR says Resolution = 0.048828125
    # Let's apply this to raw.
    RESOLUTION = 0.048828125
    data = data * RESOLUTION
    return data.reshape((n_samples, n_ch)).T # (ch, time)

# Load Raw
raw_data = load_eeg(RAW_PATH, 6) # Raw has 6 chans
raw_c3 = raw_data[0] # Ch1 = C3

# Load Filtered
filt_data = load_eeg(FILT_PATH, 2) # Filtered has 2 chans
filt_c3 = filt_data[0] # Ch1 = C3

# Match lengths (Filtered might be shorter or same?)
min_len = min(len(raw_c3), len(filt_c3))
raw_c3 = raw_c3[:min_len]
filt_c3 = filt_c3[:min_len]

# FFT Analysis
print("Calculating PSD...")
f_raw, p_raw = welch(raw_c3, fs=SR, nperseg=SR*4)
f_filt, p_filt = welch(filt_c3, fs=SR, nperseg=SR*4)

# Spectrum Shape Analysis (Filtered Only)
print("--- Spectrum Shape Analysis (Filtered Data) ---")

# Normalize PSD to mean power in passband (e.g., 20-30Hz)
idx_pass_start = np.argmin(np.abs(f_filt - 20))
idx_pass_end = np.argmin(np.abs(f_filt - 30))
ref_power = np.mean(p_filt[idx_pass_start:idx_pass_end])
p_norm = p_filt / ref_power # 0dB at 20-30Hz

def check_freq_drop(f, p, target, label):
    idx = np.argmin(np.abs(f - target))
    val_db = 10 * np.log10(p[idx])
    print(f"{label} ({target}Hz): {val_db:.2f} dB")

check_freq_drop(f_filt, p_norm, 1.0, "HPF Check @ 1Hz")
check_freq_drop(f_filt, p_norm, 3.0, "HPF Cutoff @ 3Hz")
check_freq_drop(f_filt, p_norm, 45.0, "Pre-Notch @ 45Hz")
check_freq_drop(f_filt, p_norm, 50.0, "Notch Check @ 50Hz")
check_freq_drop(f_filt, p_norm, 55.0, "Post-Notch @ 55Hz")
check_freq_drop(f_filt, p_norm, 110.0, "Pre-RecLPF @ 110Hz")
check_freq_drop(f_filt, p_norm, 120.0, "RecLPF Check @ 120Hz")
check_freq_drop(f_filt, p_norm, 130.0, "Post-RecLPF @ 130Hz")
check_freq_drop(f_filt, p_norm, 900.0, "Pre-AnaLPF @ 900Hz")
check_freq_drop(f_filt, p_norm, 1000.0, "AnaLPF Check @ 1000Hz")

# 120Hz vs 1000Hz check (Filtered)
p_100 = p_norm[np.argmin(np.abs(f_filt - 100))]
p_200 = p_norm[np.argmin(np.abs(f_filt - 200))]
drop_120_db = 10 * np.log10(p_200 / p_100)
print(f"Filtered Drop from 100Hz to 200Hz: {drop_120_db:.2f} dB")

print("\n--- Spectrum Shape Analysis (Original Raw Data) ---")
# Normalize Raw PSD
idx_pass_start = np.argmin(np.abs(f_raw - 20))
idx_pass_end = np.argmin(np.abs(f_raw - 30))
ref_power_raw = np.mean(p_raw[idx_pass_start:idx_pass_end])
p_norm_raw = p_raw / ref_power_raw

def check_freq_drop_raw(f, p, target, label):
    idx = np.argmin(np.abs(f - target))
    val_db = 10 * np.log10(p[idx])
    print(f"{label} ({target}Hz): {val_db:.2f} dB")

check_freq_drop_raw(f_raw, p_norm_raw, 50.0, "Raw Notch Check @ 50Hz")
check_freq_drop_raw(f_raw, p_norm_raw, 100.0, "Raw Check @ 100Hz")
check_freq_drop_raw(f_raw, p_norm_raw, 120.0, "Raw LPF Cutoff @ 120Hz")
check_freq_drop_raw(f_raw, p_norm_raw, 200.0, "Raw Check @ 200Hz")

# Calculate Raw Drop from 100Hz to 200Hz
p_raw_100 = p_norm_raw[np.argmin(np.abs(f_raw - 100))]
p_raw_200 = p_norm_raw[np.argmin(np.abs(f_raw - 200))]
drop_raw_120 = 10 * np.log10(p_raw_200 / p_raw_100)
print(f"Raw Data Drop (100Hz -> 200Hz): {drop_raw_120:.2f} dB")
if drop_raw_120 < -6:
    print(">> EVIDENCE: Original Raw Data IS filtered at 120Hz (Significant Drop).")
else:
    print(">> EVIDENCE: Original Raw Data is BROAD BAND (No significant drop).")

