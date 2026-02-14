
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

ID = 'id0030002'
SAMPLING_RATE = 2500 # Fixed to Correct Rate
LOW_CUT = 3.0
HIGH_CUT = 1000.0
NOTCH_FREQ = 50.0

# Targets for id0030002
TGT_SP = 0.9400
TGT_PP30 = 0.7993

print(f"--- Simulating Analyzer Analysis for {ID} (V8.67) ---")

raw_path = f'/Users/itoakane/Research/Analyzer/{ID}.eeg'
if not os.path.exists(raw_path):
    print("File not found.")
    exit()

data = np.fromfile(raw_path, dtype=np.float32)
n_ch = 6
n_samples = data.shape[0] // n_ch
# V8.67 Fix: File is Multiplexed (samples, channels)
data = data.reshape(n_samples, n_ch).T

# Apply Resolution (0.048828125 uV)
RESOLUTION = 0.048828125
data = data * RESOLUTION

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

# Filter CP3 (Channel 2)
filt_cp3 = filtfilt(b_bp, a_bp, data[2])
filt_cp3 = filtfilt(b_notch, a_notch, filt_cp3)

vmrk_path = f'/Users/itoakane/Research/Analyzer/{ID}.vmrk'
markers = {'A': [], 'B': []}

with open(vmrk_path, 'r') as f:
    for line in f:
        if line.startswith('Mk'):
            parts = line.split(',')
            pos = int(parts[2])
            desc = parts[1].strip()
            if 'A' in desc: markers['A'].append(pos)
            if 'B' in desc: markers['B'].append(pos)

print(f"   Found {len(markers['A'])} 'A' (sp), {len(markers['B'])} 'B' (pp30).")

def measure(m_list, cond_name, tgt):
    epochs = []
    rejected = 0
    start_samp = int(0.0 * SAMPLING_RATE)
    end_samp = int(0.05 * SAMPLING_RATE)
    
    for m in m_list:
        # Rejection check (-50 to +100ms)
        chk_s = m + int(-0.05 * SAMPLING_RATE)
        chk_e = m + int(0.1 * SAMPLING_RATE)
        if chk_s < 0 or chk_e >= len(filt_cp3): continue
        if np.max(np.abs(filt_cp3[chk_s:chk_e])) > 70.0:
            rejected += 1
            continue
            
        s = m + start_samp
        e = m + end_samp
        if s < 0 or e >= len(filt_cp3): continue
        epochs.append(filt_cp3[s:e])
        
    if len(epochs) == 0:
        print(f"[{cond_name}] No Valid Epochs.")
        return
        
    avg = np.mean(epochs, axis=0)
    
    t = np.linspace(0, 50, len(avg))
    idx_n20 = np.where((t >= 15) & (t <= 23))[0]
    idx_p25 = np.where((t >= 22) & (t <= 30))[0]
    
    min_val = np.min(avg[idx_n20])
    max_val = np.max(avg[idx_p25])
    pp_val = max_val - min_val
    
    print(f"[{cond_name}] Rejected: {rejected} ({rejected/len(m_list)*100:.1f}%)")
    print(f"       Target: {tgt:.4f} uV")
    print(f"       Result: {pp_val:.4f} uV")
    print(f"       Diff:   {pp_val - tgt:.4f} uV")

measure(markers['A'], 'sp (A)', TGT_SP)
measure(markers['B'], 'pp30 (B)', TGT_PP30)
