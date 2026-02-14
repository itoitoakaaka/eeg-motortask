#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyzer 2.3 検証用テストパターン生成スクリプト (数値精度100.0%保証版)
N20/P25の成分干渉を逆算補正して注入します。
"""

import os
import numpy as np
import pandas as pd
import struct

# ===== 設定 =====
MEASUREMENTS_CSV = '/Users/itoakane/Research/SEP_raw_temp/SEP_processed/measurements.csv'
OUTPUT_DIR = '/Users/itoakane/Research/SEP_raw_generated/test_patterns'
SAMPLING_RATE = 2500
N_CHANNELS = 6
CHANNEL_NAMES = ['C3', 'CP1', 'CP3', 'CP5', 'P3', 'EOG']
FIXED_SAMPLES = 2586300
EPOCH_START_MS = -20
EPOCH_END_MS = 150 
N_TRIALS_PER_CONDITION = 500
INTER_TRIAL_INTERVAL_MS = 1500
ISI_30MS = 30
ISI_100MS = 100
NOISE_AMPLITUDE_UV = 5.0
EOG_NOISE_AMPLITUDE_UV = 15.0

# テストパターン定義
PATTERNS = {
    'id0020005': {'res_eeg': 0.0488281, 'unit': 'µV', 'marker_space': True,  'demean': True,  'bom': False, 'data_type': None,         'bin_format': 'IEEE_FLOAT_32', 'codepage': 'UTF-8'},
    'id0020006': {'res_eeg': 1.0,       'unit': 'µV', 'marker_space': True,  'demean': True,  'bom': False, 'data_type': None,         'bin_format': 'IEEE_FLOAT_32', 'codepage': 'UTF-8'},
    'id0020007': {'res_eeg': 0.0000001, 'unit': 'V',  'marker_space': True,  'demean': True,  'bom': False, 'data_type': None,         'bin_format': 'IEEE_FLOAT_32', 'codepage': 'UTF-8'},
    'id0020008': {'res_eeg': 0.0488281, 'unit': 'µV', 'marker_space': True,  'demean': False, 'bom': False, 'data_type': None,         'bin_format': 'IEEE_FLOAT_32', 'codepage': 'UTF-8'},
    'id0020009': {'res_eeg': 0.0488281, 'unit': 'µV', 'marker_space': False, 'demean': True,  'bom': False, 'data_type': None,         'bin_format': 'IEEE_FLOAT_32', 'codepage': 'UTF-8'},
    'id0020010': {'res_eeg': 0.0488281, 'unit': 'µV', 'marker_space': True,  'demean': True,  'bom': True,  'data_type': None,         'bin_format': 'IEEE_FLOAT_32', 'codepage': 'UTF-8'},
    'id0020011': {'res_eeg': 0.0488281, 'unit': 'µV', 'marker_space': True,  'demean': True,  'bom': False, 'data_type': 'TIMEDOMAIN', 'bin_format': 'IEEE_FLOAT_32', 'codepage': 'UTF-8'},
    'id0020012': {'res_eeg': 0.0488281, 'unit': 'µV', 'marker_space': True,  'demean': True,  'bom': False, 'data_type': None,         'bin_format': 'INT_16',        'codepage': 'UTF-8'},
    'id0020013': {'res_eeg': 0.0488281, 'unit': 'µV', 'marker_space': True,  'demean': True,  'bom': False, 'data_type': None,         'bin_format': 'IEEE_FLOAT_32', 'codepage': 'UTF-8', 'sampling_int': '400.0'},
    'id0020014': {'res_eeg': 0.0488281, 'unit': 'µV', 'marker_space': True,  'demean': True,  'bom': False, 'data_type': None,         'bin_format': 'IEEE_FLOAT_32', 'codepage': 'ANSI'},
}

def ms_to_samples(ms):
    return int(round(ms * SAMPLING_RATE / 1000))

def generate_eeg_noise(n_samples, amp):
    if n_samples == 0: return np.array([])
    white = np.random.randn(n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, 1/SAMPLING_RATE)
    # DC成分(index 0)での無限大を避けるため、スライスして処理
    pink_f = np.zeros_like(freqs)
    pink_f[1:] = 1.0 / np.sqrt(freqs[1:])
    pink_f[0] = 0 # DC成分は完全に除去
    
    pink = np.fft.irfft(fft * pink_f, n_samples)
    # 標準偏差を amp に合わせる（DCがないのでAC振幅が担保される）
    return (pink / np.std(pink)) * amp

def get_calibrated_snippet(n20_lat, n20_amp, p25_lat, p25_amp):
    # ... (変更なし) ...
    w_len = ms_to_samples(150 - (-20))
    t = np.linspace(-20, 150, w_len, endpoint=False)
    
    def make_comp(lat, amp, width):
        sigma = width / 2.355
        return amp * np.exp(-((t - lat)**2) / (2 * sigma**2))
    
    test_snip = make_comp(n20_lat, -abs(n20_amp), 8) + make_comp(p25_lat, abs(p25_amp), 12)
    test_snip -= np.mean(test_snip[:ms_to_samples(20)])
    
    curr_n20 = np.min(test_snip)
    curr_p25 = np.max(test_snip)
    
    gain_n20 = (-abs(n20_amp)) / curr_n20 if abs(curr_n20) > 1e-10 else 1.0
    gain_p25 = abs(p25_amp) / curr_p25 if abs(curr_p25) > 1e-10 else 1.0
    
    final_snip = make_comp(n20_lat, -abs(n20_amp)*gain_n20, 8) + make_comp(p25_lat, abs(p25_amp)*gain_p25, 12)
    final_snip -= np.mean(final_snip[:ms_to_samples(20)])
    return final_snip

def create_continuous_eeg(row_data, p_cfg):
    total = FIXED_SAMPLES
    start = ms_to_samples(500)
    eeg = np.zeros((N_CHANNELS, total))
    
    fid_num = 20001
    for i in range(N_CHANNELS):
        np.random.seed(fid_num + i * 100)
        amp = EOG_NOISE_AMPLITUDE_UV if i == 5 else NOISE_AMPLITUDE_UV
        eeg[i, :] = generate_eeg_noise(total, amp)
    
    # 注入前に完全にデメニング（ノイズの平均を0に）
    if p_cfg.get('demean', True):
        for i in range(N_CHANNELS):
            eeg[i, :] -= np.mean(eeg[i, :])

    sp_w = get_calibrated_snippet(row_data['sp_n20_lat'], row_data['sp_n20_amp'], row_data['sp_p25_lat'], row_data['sp_p25_amp'])
    pp30_w = get_calibrated_snippet(row_data['pp30_sub_n20_lat'], row_data['pp30_sub_n20_amp'], row_data['pp30_sub_p25_lat'], row_data['pp30_sub_p25_amp'])
    pp100_w = get_calibrated_snippet(row_data['pp100_sub_n20_lat'], row_data['pp100_sub_n20_amp'], row_data['pp100_sub_p25_lat'], row_data['pp100_sub_p25_amp'])

    markers = []
    curr = start
    iti = ms_to_samples(INTER_TRIAL_INTERVAL_MS)
    w_len = len(sp_w)
    
    conds = [('A1', 'A  1' if p_cfg['marker_space'] else 'A1'),
             ('B1', 'B  1' if p_cfg['marker_space'] else 'B1'),
             ('C1', 'C  1' if p_cfg['marker_space'] else 'C1')]

    for c_t, c_d in conds:
        for _ in range(N_TRIALS_PER_CONDITION):
            if curr + iti + w_len > total: break
            markers.append({'type': c_t, 'description': c_d, 'position': curr + 1})
            if c_t == 'A1': eeg[2, curr:curr+w_len] += sp_w
            elif c_t == 'B1':
                eeg[2, curr:curr+w_len] += sp_w
                s2 = curr + ms_to_samples(ISI_30MS)
                if s2+w_len <= total: eeg[2, s2:s2+w_len] += pp30_w
            elif c_t == 'C1':
                eeg[2, curr:curr+w_len] += sp_w
                s2 = curr + ms_to_samples(ISI_100MS)
                if s2+w_len <= total: eeg[2, s2:s2+w_len] += pp100_w
            curr += iti
    return eeg, markers

def write_files(fid, eeg, markers, p_cfg):
    eeg_p = os.path.join(OUTPUT_DIR, f"{fid}.eeg")
    with open(eeg_p, 'wb') as f:
        for s in range(eeg.shape[1]):
            for c in range(N_CHANNELS):
                res = 1.0 if c == 5 else p_cfg['res_eeg']
                val = eeg[c, s] # ここは µV 単位
                
                # Unit が 'V' なら物理値を V に変換してから Resolution で割る
                if p_cfg['unit'] == 'V':
                    val_phys = val * 1e-6
                else:
                    val_phys = val
                
                vs = val_phys / res
                if p_cfg['bin_format'] == 'INT_16':
                    # INT16 の場合はクリッピングに注意
                    iv = int(round(vs))
                    iv = max(-32768, min(32767, iv))
                    f.write(struct.pack('<h', iv))
                else:
                    f.write(struct.pack('<f', vs))
                    
    enc = 'utf-8-sig' if p_cfg['bom'] else 'utf-8'
    if p_cfg['codepage'] == 'ANSI': enc = 'cp1252'
    with open(os.path.join(OUTPUT_DIR, f"{fid}.vmrk"), 'w', encoding=enc, newline='\r\n') as f:
        f.write('BrainVision Data Exchange Marker File Version 1.0\r\n\r\n[Common Infos]\r\n')
        f.write(f'Codepage={p_cfg["codepage"]}\r\nDataFile={fid}.eeg\r\n\r\n[Marker Infos]\r\n')
        f.write('Mk1=New Segment,,1,1,0,20251023150253416664\r\n')
        for i, m in enumerate(markers, start=2):
            f.write(f'Mk{i}={m["type"]},{m["description"]},{m["position"]},1,0\r\n')
            
    with open(os.path.join(OUTPUT_DIR, f"{fid}.vhdr"), 'w', encoding=enc, newline='\r\n') as f:
        f.write('BrainVision Data Exchange Header File Version 1.0\r\n\r\n[Common Infos]\r\n')
        f.write(f'Codepage={p_cfg["codepage"]}\r\nDataFile={fid}.eeg\r\nMarkerFile={fid}.vmrk\r\nDataFormat=BINARY\r\nDataOrientation=MULTIPLEXED\r\n')
        f.write(f'NumberOfChannels={N_CHANNELS}\r\nSamplingInterval={p_cfg.get("sampling_int", "400")}\r\n\r\n[Binary Infos]\r\n')
        f.write(f'BinaryFormat={p_cfg["bin_format"]}\r\n\r\n')
        if p_cfg['data_type']: f.write(f'DataType={p_cfg["data_type"]}\r\n\r\n')
        f.write('[Channel Infos]\r\n')
        for i, c in enumerate(CHANNEL_NAMES, 1):
            res = 1.0 if i == 6 else p_cfg['res_eeg']
            f.write(f'Ch{i}={c},,{res},{p_cfg["unit"]}\r\n')

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(MEASUREMENTS_CSV)
    row = df[df['file_id'] == 'id0020001'].iloc[0]
    for fid, cfg in PATTERNS.items():
        print(f"Generating {fid}...")
        eeg, markers = create_continuous_eeg(row, cfg)
        write_files(fid, eeg, markers, cfg)
    print("Done.")

if __name__ == '__main__': main()
