import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import butter, filtfilt, lfilter
import os
import re

# Configuration
ID = 'id0010001'
RAW_EEG_PATH = f'Analyzer/{ID}.eeg'
RAW_VMRK_PATH = f'Analyzer/{ID}.vmrk'
CSV_PATH = 'mne/SEP_processed/measurements.csv'
SAMPLING_RATE = 2500

def load_generated_data():
    # Load binary float32 (Multiplexed)
    # Header says 5 channels + 0 EOG? No, script V8.35 wrote 5 channels.
    # Check generate_ids_for_analyzer.py: NO_EOG_CHANNELS = ['C3', 'CP1', 'CP3', 'CP5', 'P3'] -> 5 channels
    n_ch = 5
    data = np.fromfile(RAW_EEG_PATH, dtype='<f4')
    n_samples = len(data) // n_ch
    data = data.reshape((n_samples, n_ch)).T # (ch, time)
    
    # Apply Resolution (ADC -> uV) matchng VHDR
    RESOLUTION = 0.048828125
    data *= RESOLUTION
    
    return data

def load_markers():
    markers = []
    with open(RAW_VMRK_PATH, 'r') as f:
        for line in f:
            if line.startswith('Mk'):
                # Mk<N>=<Type>,<Desc>,<Pos>,...
                parts = line.strip().split(',')
                if len(parts) > 2:
                    desc = parts[1]
                    pos = int(parts[2])
                    markers.append((desc, pos))
    return markers

def analyzer_simulation(data, markers):
    print("--- Simulating Analyzer Pipeline ---")
    
    # 1. Filters (IIR Zero Phase)
    # BP 3-1000Hz
    b_bp, a_bp = butter(2, [3/(0.5*SAMPLING_RATE), 1000/(0.5*SAMPLING_RATE)], btype='bandpass')
    # Notch 50Hz
    b_notch, a_notch = butter(2, [48/(0.5*SAMPLING_RATE), 52/(0.5*SAMPLING_RATE)], btype='bandstop')
    
    print("Applying Filters (3-1000Hz, Notch 50Hz, Zero-Phase)...")
    # Apply to all channels
    filt_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        # Analyzer uses Zero-Phase (Forward-Backward)
        tmp = filtfilt(b_bp, a_bp, data[i])
        filt_data[i] = filtfilt(b_notch, a_notch, tmp)
        
    # 2. Segmentation & Baseline & Average
    # Epoch: -20ms to 200ms (550 samples). 0ms is Marker Pos.
    # Base: -20ms to 0ms (0 to 50 samples in epoch)
    
    # Conditions
    cond_map = {'A  1': 'sp', 'B  1': 'pp30', 'C  1': 'pp100'} # V8.35 maps A->A 1 etc.
    # Check marker format in V8.35
    # generate script: f.write(f'Mk{i}={c_code},{c_desc},{p},1,0\r\n') 
    # c_desc is "A  1" etc.
    
    epochs = {'sp': [], 'pp30': [], 'pp100': []}
    
    print("Segmenting & Baseline Correcting...")
    
    # Correction Window: -20ms to 0ms (0 to 50 samples)
    PRE_TRIG = int(20 * SAMPLING_RATE / 1000) # 50
    POST_TRIG = int(200 * SAMPLING_RATE / 1000) # 500
    EPOCH_LEN = PRE_TRIG + POST_TRIG # 550
    
    for desc, pos in markers:
        # Identify cond
        cond = None
        if 'A' in desc and '1' in desc: cond = 'sp'
        elif 'B' in desc and '1' in desc: cond = 'pp30'
        elif 'C' in desc and '1' in desc: cond = 'pp100'
        
        if cond:
            start = pos - PRE_TRIG
            end = pos + POST_TRIG
            if start >= 0 and end <= filt_data.shape[1]:
                # Extract CP3 (Channel index 2 in [C3, CP1, CP3, CP5, P3])
                seg = filt_data[2, start:end]
                
                # Baseline Correction
                base_val = np.mean(seg[0:PRE_TRIG])
                seg_corr = seg - base_val
                
                # Rejection Check (+-70uV)
                max_val = np.max(np.abs(seg_corr))
                if max_val > 70:
                    pass # Rejected
                else:
                    epochs[cond].append(seg_corr)

    results = {}
    print(f"Total Markers Processed: {len(markers)}")
    if len(epochs['sp']) == 0:
        print(f"DEBUG: All SP epochs rejected or not found. Start={start}, End={end}")
        print(f"DEBUG: Data Range: Min={np.min(filt_data)}, Max={np.max(filt_data)}")
    print("Averaging...")
    for cond, wav in epochs.items():
        if len(wav) > 0:
            avg = np.mean(np.array(wav), axis=0) # (550,)
            results[cond] = avg
            print(f"Condition {cond}: {len(wav)} epochs averaged.")
        else:
            print(f"Condition {cond}: No epochs found.")
            
    return results

def verify_peaks(results):
    print("\n--- Verifying Peaks vs CSV ---")
    df = pd.read_csv(CSV_PATH)
    row = df[df['file_id'] == ID].iloc[0]
    
    # Time vector for epoch (-20ms to 200ms)
    t = np.linspace(-20, 200, 550, endpoint=False)
    
    # Check spSEP (Condition A)
    # Target N20 Latency / Amp
    # CSV has PP amplitudes. Analyzer peak detection finds Min/Max in window.
    
    targets = [
        ('sp', 'sp_n20_lat', 'sp_p25_lat', 'sp_pp_amp'),
        ('pp30', 'pp30_n20_lat', 'pp30_p25_lat', 'pp30_pp_amp'),
        ('pp100', 'pp100_n20_lat', 'pp100_p25_lat', 'pp100_pp_amp')
    ]
    
    total_error_amp = 0.0
    
    for cond, n_col, p_col, amp_col in targets:
        if cond not in results: continue
        
        wave = results[cond]
        
        # Target Latencies from CSV
        n_lat_target = row[n_col]
        p_lat_target = row[p_col]
        target_pp = row[amp_col]
        
        # Search Window +/- 1.0ms around target
        # Index conversion
        def get_idx(ms): return int((ms + 20) * SAMPLING_RATE / 1000)
        
        n_idx_start = get_idx(n_lat_target - 2.0)
        n_idx_end = get_idx(n_lat_target + 2.0)
        p_idx_start = get_idx(p_lat_target - 2.0)
        p_idx_end = get_idx(p_lat_target + 2.0)
        
        # Find Min for N20, Max for P25
        # N20 is negative
        n_val = np.min(wave[n_idx_start:n_idx_end])
        n_pos_idx = np.argmin(wave[n_idx_start:n_idx_end]) + n_idx_start
        n_lat_found = t[n_pos_idx]
        
        # P25 is positive
        p_val = np.max(wave[p_idx_start:p_idx_end])
        p_pos_idx = np.argmax(wave[p_idx_start:p_idx_end]) + p_idx_start
        p_lat_found = t[p_pos_idx]
        
        found_pp = p_val - n_val
        
        print(f"[{cond}]")
        print(f"  Target PP:  {target_pp:.4f} uV")
        print(f"  Found PP:   {found_pp:.4f} uV")
        print(f"  Amp Error:  {found_pp - target_pp:.4f} uV")
        print(f"  Latency N20: Target {n_lat_target:.2f} -> Found {n_lat_found:.2f}")
        print(f"  Latency P25: Target {p_lat_target:.2f} -> Found {p_lat_found:.2f}")
        
        total_error_amp += abs(found_pp - target_pp)

    print(f"\nTotal Abs Amp Error: {total_error_amp:.4f} uV")
    if total_error_amp < 0.1: # Allow small differences due to noise randomness
        print(">>> SUCCESS: Generated data mathematically matches Measurement target.")
    else:
        print(">>> WARNING: Mismatch detected.")

if __name__ == "__main__":
    data = load_generated_data()
    markers = load_markers()
    results = analyzer_simulation(data, markers)
    verify_peaks(results)
