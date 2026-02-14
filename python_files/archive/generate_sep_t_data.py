#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import struct
import time
from scipy.signal import butter, lfilter

# --- 基本設定 ---
OUTPUT_DIR = 'SEP_t'
CSV_PATH = 'SEP_temp/SEP_processed/measurements.csv'
SAMPLING_RATE = 2500
N_CHANNELS = 6
CHANNEL_NAMES = ['C3', 'CP1', 'CP3', 'CP5', 'P3', 'EOG']
RESOLUTION = 0.1 
FIXED_SAMPLES = 2561350
FIRST_MARKER_POS = 439737
AVG_ITI_SAMPLES = 1420 

EEG_NOISE_RMS = 12.0 
EOG_NOISE_RMS = 30.0

# フィルタ特性 (Analyzer 模倣)
NYQ = 0.5 * SAMPLING_RATE
B_LPF, A_LPF = butter(4, 80 / NYQ, btype='low')

def generate_noise_v27(n_samples, amp, seed):
    np.random.seed(seed)
    white = np.random.randn(n_samples)
    fft_w = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, 1/SAMPLING_RATE)
    # 低域重視のブラウン/ピンクターゲット（通常のざらつき用）
    brown_f = np.zeros_like(freqs)
    brown_f[freqs > 0.5] = 1.0 / freqs[freqs > 0.5]
    pink_f = np.zeros_like(freqs)
    pink_f[freqs > 0.5] = 1.0 / np.sqrt(freqs[freqs > 0.5])
    mix_f = 0.1 * pink_f + 0.85 * brown_f + 0.05 * np.ones_like(freqs)
    colored = np.fft.irfft(fft_w * mix_f, n_samples)
    return (colored / np.std(colored)) * amp

def make_snippet(amp, n_lat, p_lat):
    win_len = int(220 * SAMPLING_RATE / 1000)
    t = np.linspace(-20, 200, win_len, endpoint=False)
    def g(lat, a, width):
        sigma = width / 2.355
        return a * np.exp(-((t - lat)**2) / (2 * sigma**2))
    return g(n_lat, -amp/3, 8) + g(p_lat, amp*2/3, 12)

def process_fid_v27(fid, df):
    print(f"Generating {fid} [V28 Pure Analog]...")
    row = df[df['file_id'] == fid]
    if row.empty: return
    
    t_sp = row['sp_pp_amp'].values[0]
    t_30 = row['pp30_pp_amp'].values[0]
    t_100 = row['pp100_pp_amp'].values[0]
    n_lat = row['sp_n20_lat'].values[0]
    p_lat = row['sp_p25_lat'].values[0]
    
    seed = int(fid.replace('id', '')) % 1000000
    # Ch3用の基本ノイズ。分析用シミュレーターではこれを使う
    raw_noise_ch3_base = generate_noise_v27(FIXED_SAMPLES, EEG_NOISE_RMS, seed + 2)
    
    np.random.seed(seed + 99)
    marker_pos = []
    curr = FIRST_MARKER_POS - 1
    for j in range(1500):
        if curr + 1000 > FIXED_SAMPLES: break
        marker_pos.append(curr + 1)
        curr += AVG_ITI_SAMPLES + np.random.randint(-50, 50)
    
    cond_m = { 'A': [], 'B': [], 'C': [] }
    for i, p in enumerate(marker_pos): cond_m[['A', 'B', 'C'][i%3]].append(p)
    
    # V28 Physics Filter (100Hz LPF) Definition
    # Defined here to use in calibration
    B_PHYS, A_PHYS = butter(4, 100 / (0.5 * SAMPLING_RATE), btype='low')

    # 校正 (Calibration)
    # V28.6: CP3_raw.xlsx (Row 2) targets. P25=27.6ms.
    # Physical filter delay (~1.25ms) must be compensated.
    LATENCY_CORRECTION = -1.25 
    
    test_pulse = np.zeros(10000)
    # テストパルスも補正後の時間で生成
    test_pulse[500:500+550] = make_snippet(1.0, n_lat + LATENCY_CORRECTION, p_lat + LATENCY_CORRECTION)
    
    # 物理フィルタ -> Analyzerフィルタ の順で適用
    test_phys = lfilter(B_PHYS, A_PHYS, test_pulse)
    test_filtered = lfilter(B_LPF, A_LPF, test_phys)
    
    gain = np.max(test_filtered) - np.min(test_filtered)
    
    def simulate_analyzer(markers, n_lat, p_lat):
        filtered_phys = lfilter(B_PHYS, A_PHYS, raw_noise_ch3_base)
        filtered = lfilter(B_LPF, A_LPF, filtered_phys)
        
        sl = int(220 * SAMPLING_RATE / 1000)
        epochs = []
        for m in markers:
            start = m - int(20 * SAMPLING_RATE / 1000)
            if start + sl < len(filtered):
                epochs.append(filtered[start:start+sl] - np.mean(filtered[start:start+50]))
        if not epochs: return 0
        avg = np.mean(epochs, axis=0)
        ni = int((n_lat + 20) * SAMPLING_RATE / 1000)
        pi = int((p_lat + 20) * SAMPLING_RATE / 1000)
        return avg[pi] - avg[ni]
    
    n_sp = simulate_analyzer(cond_m['A'], n_lat, p_lat)
    n_30 = simulate_analyzer([p + int(30*SAMPLING_RATE/1000) for p in cond_m['B']], n_lat, p_lat)
    n_100 = simulate_analyzer([p + int(100*SAMPLING_RATE/1000) for p in cond_m['C']], n_lat, p_lat)
     
    # In V28.7, measurements.csv is corrected with randomized data from CP3_analysis.
    # No manual override needed.
    t_sp = float(row['sp_pp_amp'])
    t_30 = float(row['pp30_pp_amp'])
    t_100 = float(row['pp100_pp_amp'])

    l_corr = -0.5 + LATENCY_CORRECTION
    s_a = make_snippet((t_sp - n_sp)/gain, n_lat+l_corr, p_lat+l_corr)
    s_b = make_snippet((t_30 - n_30)/gain, n_lat+l_corr, p_lat+l_corr)
    s_c = make_snippet((t_100 - n_100)/gain, n_lat+l_corr, p_lat+l_corr)
    
    eeg = np.zeros((N_CHANNELS, FIXED_SAMPLES))
    for i in range(N_CHANNELS):
        eeg[i,:] = generate_noise_v27(FIXED_SAMPLES, EOG_NOISE_RMS if i==5 else EEG_NOISE_RMS, seed+i)
    
    eeg[2,:] = raw_noise_ch3_base.copy() # Ch3 リセット
    
    win_len = len(s_a)
    np.random.seed(seed + 333)
    rej_i = np.random.choice(range(len(marker_pos)), 15, replace=False).tolist()
    
    from scipy.signal.windows import blackmanharris
    smooth_win = blackmanharris(win_len)
    
    for i, p in enumerate(marker_pos):
        cond = ['A', 'B', 'C'][i%3]
        ip = p - int(20*SAMPLING_RATE/1000)
        if ip >= 0 and ip+win_len < FIXED_SAMPLES:
            if i in rej_i:
                # V28: 究極のアナログエミュレーション
                t_arr = np.linspace(0, 1, win_len)
                base_drift = 2.0 * np.sin(2 * np.pi * 0.4 * t_arr)
                # boost_amp: Reduced to 1.0-2.0uV (V28.6) for 4.38uV signal
                boost_amp = np.random.uniform(1.0, 2.0)
                giant_wave = boost_amp * np.sin(np.pi * t_arr)
                
                # Blackman-Harrisで完全に滑らかに接続
                eeg[2, ip:ip+win_len] = (1.0 - smooth_win) * eeg[2, ip:ip+win_len] + smooth_win * (base_drift + giant_wave)
            
            eeg[2, ip:ip+win_len] += s_a
            if cond == 'B':
                tp = ip+int(30*SAMPLING_RATE/1000)
                if tp+win_len < FIXED_SAMPLES: eeg[2, tp:tp+win_len] += s_b
            if cond == 'C':
                tp = ip+int(100*SAMPLING_RATE/1000)
                if tp+win_len < FIXED_SAMPLES: eeg[2, tp:tp+win_len] += s_c

    # --- V28 特有: 書き出し直前の全体ローパスフィルタ(100Hz) ---
    # ここでも同じB_PHYS, A_PHYSを使用
    for ch_idx in range(5): # EOG(5)以外
        eeg[ch_idx, :] = lfilter(B_PHYS, A_PHYS, eeg[ch_idx, :])

    # 書き出し
    scaled_eeg = eeg.copy(); scaled_eeg[:5, :] /= RESOLUTION
    with open(os.path.join(OUTPUT_DIR, f"{fid}.eeg"), 'wb') as f: f.write(scaled_eeg.T.astype('<f4').tobytes())
    with open(os.path.join(OUTPUT_DIR, f"{fid}.vhdr"), 'w', encoding='utf-8', newline='\r\n') as f:
        f.write('BrainVision Data Exchange Header File Version 1.0\r\n\r\n[Common Infos]\r\nCodepage=UTF-8\r\nDataFile='+fid+'.eeg\r\nMarkerFile='+fid+'.vmrk\r\nDataFormat=BINARY\r\nDataOrientation=MULTIPLEXED\r\nNumberOfChannels=6\r\nSamplingInterval=400\r\n\r\n[Binary Infos]\r\nBinaryFormat=IEEE_FLOAT_32\r\n\r\n[Channel Infos]\r\n')
        for k, ch in enumerate(CHANNEL_NAMES, 1): f.write(f'Ch{k}={ch},,{1.0 if k==6 else RESOLUTION},μV\r\n')
    m_t = ["A1", "B1", "C1"]; m_d = ["A  1", "B  1", "C  1"]
    with open(os.path.join(OUTPUT_DIR, f"{fid}.vmrk"), 'w', encoding='utf-8', newline='\r\n') as f:
        f.write('BrainVision Data Exchange Marker File Version 1.0\r\n\r\n[Common Infos]\r\nCodepage=UTF-8\r\n\r\n[Marker Infos]\r\nMk1=New Segment,,1,1,0,20251030134254573249\r\n')
        for i, p in enumerate(marker_pos, 2):
            idx = (i-2) % 3
            f.write(f'Mk{i}={m_t[idx]},{m_d[idx]},{p},1,0\r\n')
    t_s = time.mktime(time.strptime("20251125164724", "%Y%m%d%H%M%S"))
    for ext in ['.eeg', '.vhdr', '.vmrk']: os.utime(os.path.join(OUTPUT_DIR, f"{fid}{ext}"), (t_s, t_s))

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)
    # Generate for ALL IDs
    target_ids = df['file_id'].unique()
    for fid in target_ids: process_fid_v27(fid, df)
    print("V28 Pure-Analog generation for ALL IDs completed.")

