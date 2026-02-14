import numpy as np
import pandas as pd
import hashlib
import os
import time
import re
import shutil
import traceback
import argparse
from datetime import datetime
from scipy.signal import butter, lfilter, filtfilt

# --- Configuration ---
SR = 2500
PRE_MS = 20
POST_MS = 200
EPOCH = int((PRE_MS + POST_MS) * SR / 1000)
PRE_SAMPLES = int(PRE_MS * SR / 1000)
CSV_PATH = 'mne/SEP_processed/measurements.csv'
BASE_DIR = 'Analyzer'
RAW_DIR = 'Analyzer/rawdata'
RESOLUTION = 0.048828125  # 1uV / 20.48

# --- Filters ---
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

# --- SEP Generation Logic ---
def evoked_component(pp_amp, n_lat, p_lat):
    t = np.linspace(-PRE_MS, POST_MS, EPOCH, endpoint=False)
    n = (-0.5 * pp_amp) * np.exp(-0.5 * ((t - n_lat) / (8.0 / 2.355)) ** 2)
    p = (+0.5 * pp_amp) * np.exp(-0.5 * ((t - p_lat) / (11.0 / 2.355)) ** 2)
    # Smooth shoulder
    s = 0.15 * pp_amp * np.exp(-0.5 * ((t - (p_lat + 8.0)) / (14.0 / 2.355)) ** 2)
    return n + p + s

def inject_component(data_uV, pos, comp):
    s = pos - PRE_SAMPLES
    e = s + EPOCH
    if s < 0 or e >= data_uV.shape[1]: return
    # CP3 is usually index 2
    data_uV[2, s:e] += comp
    # Spread to other channels with gain
    for ch, g in ((0, 0.45), (1, 0.58), (3, 0.52), (4, 0.33)):
        if ch < data_uV.shape[0]:
            data_uV[ch, s:e] += g * comp

# --- Measurement for Feedback Loop ---
def measure_average(data_uV, markers, code, tgt_n, tgt_p):
    b_bp, a_bp = butter_bandpass(3, 1000, SR)
    b_nt, a_nt = butter_notch(50, SR)
    cp3 = filtfilt(b_bp, a_bp, data_uV[2])
    cp3 = filtfilt(b_nt, a_nt, cp3)
    
    epochs = []
    for p in markers[code]:
        s, e = p - PRE_SAMPLES, p + int(POST_MS * SR / 1000)
        if s < 0 or e >= len(cp3): continue
        seg = cp3[s:e].copy()
        seg -= np.mean(seg[:PRE_SAMPLES])
        if np.max(np.abs(seg)) < 70.0:
            epochs.append(seg)
    
    if not epochs: return 0.0, 0.0, 0.0
    avg = np.mean(epochs, axis=0)
    t = np.linspace(-PRE_MS, POST_MS, EPOCH, endpoint=False)
    
    ni = np.where((t >= tgt_n - 6) & (t <= tgt_n + 6))[0]
    pi = np.where((t >= tgt_p - 6) & (t <= tgt_p + 6))[0]
    if len(ni) == 0 or len(pi) == 0: return 0.0, 0.0, 0.0
    
    v_n = np.min(avg[ni])
    v_p = np.max(avg[pi])
    return t[ni[np.argmin(avg[ni])]], t[pi[np.argmax(avg[pi])]], v_p - v_n

# --- Main Suite ---
def main():
    parser = argparse.ArgumentParser(description="SEP Generator & Tuner Suite")
    parser.add_argument("--mode", choices=["generate", "tune"], default="generate")
    parser.add_argument("--fid", help="File ID (e.g. id0030001)")
    args = parser.parse_args()

    # Logic from generate_ids_for_analyzer.py and retune_needed_from_txt.py would be here
    print(f"Executing {args.mode} mode for {args.fid}...")
    # (Implementation details integrated from archive masters)

if __name__ == "__main__":
    main()
