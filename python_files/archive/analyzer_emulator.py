import numpy as np
import os
from scipy.signal import butter, lfilter

# Configuration - Matching Analyzer workflow in `analyzer_standard_workflow.md`
DATA_DIR = 'Analyzer'
SAMPLING_RATE = 2500
RESOLUTION = 0.048828125
CHANNEL_NAMES = ['C3', 'CP1', 'CP3', 'CP5', 'P3', 'EOG']
T_MIN = -0.020 # -20ms
T_MAX = 0.200  # 200ms
BASELINE_MIN = -0.020
BASELINE_MAX = 0.0

def load_eeg(fid):
    path = os.path.join(DATA_DIR, f"{fid}.eeg")
    data = np.fromfile(path, dtype='<f4')
    return data.reshape(-1, len(CHANNEL_NAMES)).T * RESOLUTION

def get_markers(fid):
    path = os.path.join(DATA_DIR, f"{fid}.vmrk")
    markers = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('Mk'):
                parts = line.split(',')
                if len(parts) > 2:
                    pos = int(parts[2])
                    label = parts[0].split('=')[1]
                    markers.append((pos, label))
    return markers

import pandas as pd

CSV_PATH = 'mne/SEP_processed/measurements.csv'

def analyze_subject(fid):
    print(f"\n--- Emulating Analyzer Pipeline for {fid} ---")
    data = load_eeg(fid)
    markers = get_markers(fid)
    df = pd.read_csv(CSV_PATH)
    row = df[df['file_id'] == fid]
    
    # 1. Filter (Standard Analyzer: 3-100Hz for SEP)
    b, a = butter(2, [3 / (0.5 * SAMPLING_RATE), 100 / (0.5 * SAMPLING_RATE)], btype='bandpass')
    filtered = lfilter(b, a, data[2, :]) # CP3 Only
    
    # 2. Segment, Baseline, and ARTIFACT REJECTION
    pre_samples = int(abs(T_MIN) * SAMPLING_RATE)
    post_samples = int(T_MAX * SAMPLING_RATE)
    
    epochs = { 'A1': [], 'B1': [], 'C1': [] }
    rejected_count = 0
    total_count = 0
    
    for pos, label in markers:
        if label in epochs:
            total_count += 1
            start = pos - pre_samples
            end = pos + post_samples
            if start >= 0 and end < len(filtered):
                seg = filtered[start:end]
                # Artifact Rejection (+/- 70uV)
                if np.max(np.abs(seg)) > 70:
                    rejected_count += 1
                    continue
                # Baseline correction (-20 to 0ms)
                seg = seg - np.mean(seg[:pre_samples])
                epochs[label].append(seg)
    
    print(f"Artifact Rejection: {rejected_count} / {total_count} epochs rejected ({(rejected_count/total_count)*100:.1f}%)")
    
    # 3. Average and Peak Detection
    times = np.linspace(T_MIN*1000, T_MAX*1000, pre_samples + post_samples)
    
    for cond, segs in epochs.items():
        if not segs: continue
        avg = np.mean(segs, axis=0)
        
        # Search Windows
        offset_ms = 0
        target_pp = 0
        if cond == 'A1': 
            offset_ms = 0
            target_pp = float(row['sp_pp_amp']) if not row.empty else 0
        elif cond == 'B1': 
            offset_ms = 30
            target_pp = float(row['pp30_pp_amp']) if not row.empty else 0
        elif cond == 'C1': 
            offset_ms = 100
            target_pp = float(row['pp100_pp_amp']) if not row.empty else 0
        
        n_win = (times >= offset_ms + 15) & (times <= offset_ms + 25)
        p_win = (times >= offset_ms + 22) & (times <= offset_ms + 35)
        
        v_n = np.min(avg[n_win]) # Standard N20 is Negative
        v_p = np.max(avg[p_win]) # Standard P25 is Positive
        p2p = v_p - v_n
        
        print(f"Condition {cond}:")
        print(f"  N20: {v_n:.4f} µV, P25: {v_p:.4f} µV")
        print(f"  P-P: {p2p:.4f} µV (Target: {target_pp:.4f} µV, Error: {p2p - target_pp:.4f})")
        if (v_n > 0 and v_p < 0):
            print("  WARNING: Polarity seems INVERTED")
        elif (v_n < 0 and v_p > 0):
            print("  Polarity is NORMAL (N=neg, P=pos)")

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)
    target_ids = df['file_id'].unique()
    for fid in target_ids:
        try: analyze_subject(fid)
        except Exception as e: print(f"Error {fid}: {e}")
