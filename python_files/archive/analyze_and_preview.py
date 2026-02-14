
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

ID = 'id0030001'
SAMPLING_RATE = 5000
LOW_CUT = 3.0
HIGH_CUT = 1000.0
NOTCH_FREQ = 50.0

# Hardcoded Target for pp30
tgt_pp = 0.9843 

print(f"--- Simulating Analyzer Analysis for {ID} (pp30) ---")

raw_path = f'Analyzer/{ID}.eeg'
if not os.path.exists(raw_path):
    print("File not found.")
    exit()

data = np.fromfile(raw_path, dtype=np.float32)
n_ch = 6
n_samples = data.shape[0] // n_ch
data = data.reshape(n_ch, n_samples)

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_notch(notch_freq, fs, order=2):
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    b, a = butter(order, [w0 - 0.005, w0 + 0.005], btype='bandstop')
    return b, a

b_bp, a_bp = butter_bandpass(LOW_CUT, HIGH_CUT, SAMPLING_RATE)
b_notch, a_notch = butter_notch(NOTCH_FREQ, SAMPLING_RATE)

filt_data = np.zeros_like(data)
for i in range(n_ch):
    temp = filtfilt(b_bp, a_bp, data[i])
    filt_data[i] = filtfilt(b_notch, a_notch, temp)

vmrk_path = f'Analyzer/{ID}.vmrk'
markers = []
with open(vmrk_path, 'r') as f:
    for line in f:
        if line.startswith('Mk'):
            parts = line.split(',')
            if 'B' in parts[1]: # Marker B
                markers.append(int(parts[2]))

print(f"   Found {len(markers)} 'B' (pp30) markers.")

start_samp = int(0.0 * SAMPLING_RATE)
end_samp = int(0.05 * SAMPLING_RATE)
n_pts = end_samp - start_samp

epochs = []
rejected = 0
for m in markers:
    s = m + start_samp
    e = m + end_samp
    if s < 0 or e > n_samples: continue
    
    epoch = filt_data[:, s:e]
    cp3 = epoch[2] 
    
    if np.max(np.abs(cp3)) > 70.0:
        rejected += 1
    else:
        epochs.append(cp3)

print(f"   Rejected: {rejected} epochs.")

avg = np.mean(epochs, axis=0)

t = np.linspace(0, 50, n_pts)
idx_n20 = np.where((t >= 15) & (t <= 23))[0]
idx_p25 = np.where((t >= 22) & (t <= 30))[0]

min_val = np.min(avg[idx_n20])
max_val = np.max(avg[idx_p25])
pp_val = max_val - min_val

print("4. Peak Detection")
print(f"   Target P-P: {tgt_pp:.4f} uV")
print(f"   Found P-P:  {pp_val:.4f} uV")
print(f"   Diff:       {pp_val - tgt_pp:.4f} uV")
