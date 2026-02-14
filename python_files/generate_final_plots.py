
import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import sys

# Settings
RAW_DIR = '/Users/itoakane/Research/SEP_rawdata_new'
EXCEL_PATH = '/Users/itoakane/Research/task/task.xlsx'  
OUTPUT_DIR = '/Users/itoakane/Research/SEP_rawdata_new'
TIME_WINDOW_PLOT = (-20, 100) 
PLOT_X_LIM = (-10, 100) 
Y_LIM = (-5, 5)

def load_conditions():
    df = pd.read_excel(EXCEL_PATH)
    file_cond_map = {}
    return file_cond_map

def process_all():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    cond_map = load_conditions()
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.vhdr")))
    
    # Store: [Condition][Type] -> list of evoked (or array)
    data_store = {
        'Land': {'spSEP': [], 'PPI30': [], 'PPI100': []},
        'Water': {'spSEP': [], 'PPI30': [], 'PPI100': []}
    }
    
    # Re-read Excel with logic
    df_excel = pd.read_excel(EXCEL_PATH)
    # Excel uses 'ID' for Subject Identifier (e.g. test002-1)
    # We need to extract the base Subject (e.g. 002) to group.
    
    def extract_subject_num(text_id):
        # test002-1 -> 002
        s = str(text_id)
        s = s.replace('test', '')
        s = s.split('-')[0]
        return s.zfill(3)

    df_excel['SubjectNum'] = df_excel['ID'].apply(extract_subject_num)
    df_excel['counts'] = df_excel.groupby('SubjectNum').cumcount()
    
    trial_cond_map = {}
    for _, row in df_excel.iterrows():
        sid = f"id{row['SubjectNum']}" # id002
        key = f"{sid}_{row['counts']}"
        trial_cond_map[key] = row['Condition']
        
    subj_counters = {}
    
    for fpath in files:
        basename = os.path.basename(fpath)
        file_id = os.path.splitext(basename)[0]
        subj_id = file_id[:5]
        
        if subj_id not in subj_counters: subj_counters[subj_id] = 0
        current_idx = subj_counters[subj_id]
        subj_counters[subj_id] += 1
        
        key = f"{subj_id}_{current_idx}"
        cond = trial_cond_map.get(key, None)
        
        if not cond:
            # Fallback for old mapping style if index mismatch
            # Try plain subject match if only 1 exists? No, safer to skip.
            print(f"Skipping {file_id}: Condition not found in Excel (Key: {key})")
            continue
            
        print(f"File: {file_id} | Subj: {subj_id} | Idx: {current_idx} | Cond: {cond}")
            
        # Load and Epoch
        try:
            raw = mne.io.read_raw_brainvision(fpath, preload=True, verbose='ERROR')
            d = raw.get_data()[0] * 1e6
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue
            
        events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
        if len(events) == 0: 
            print(f"No events in {file_id}")
            continue
        
        # Map events
        code_map = {}
        for k, v in event_id.items():
            k_upper = str(k).upper()
            if '10001' in k_upper or 'A1' in k_upper or 'S1' in k_upper.replace("S10","X") or k=='1' or 'S 1' in k_upper or k_upper.endswith('S 1') or k_upper.endswith('S  1'):
                code_map['sp'] = v
            elif '10002' in k_upper or 'B1' in k_upper or 'S2' in k_upper or k=='2' or 'S 2' in k_upper or k_upper.endswith('S 2') or k_upper.endswith('S  2'):
                code_map['pp30'] = v
            elif '10003' in k_upper or 'C1' in k_upper or 'S3' in k_upper or k=='3' or 'S 3' in k_upper or k_upper.endswith('S 3') or k_upper.endswith('S  3'):
                code_map['pp100'] = v
        
        tmin, tmax = -0.05, 0.15
        epochs = mne.Epochs(raw, events, event_id=None, tmin=tmin, tmax=tmax, baseline=(tmin,0),
                            picks=[raw.ch_names[0]], verbose='ERROR', event_repeated='merge') 
                            
        evoked_sp = epochs[str(code_map.get('sp'))].average() if code_map.get('sp') in events[:,2] else None
        evoked_pp30 = epochs[str(code_map.get('pp30'))].average() if code_map.get('pp30') in events[:,2] else None
        evoked_pp100 = epochs[str(code_map.get('pp100'))].average() if code_map.get('pp100') in events[:,2] else None
        
        if evoked_sp:
            data_sp = evoked_sp.data[0] * 1e6
            times = evoked_sp.times * 1000
            data_store[cond]['spSEP'].append(data_sp)
            
            # Subtraction
            if evoked_pp30:
                sub_30 = mne.combine_evoked([evoked_pp30, evoked_sp], weights=[1, -1])
                data_store[cond]['PPI30'].append(sub_30.data[0] * 1e6)
                
            if evoked_pp100:
                sub_100 = mne.combine_evoked([evoked_pp100, evoked_sp], weights=[1, -1])
                data_store[cond]['PPI100'].append(sub_100.data[0] * 1e6)
                
    if 'times' not in locals(): 
        print("No valid data found.")
        return

    common_times = times 
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    colors = {'Land': 'orange', 'Water': 'skyblue', 'Underwater': 'skyblue'}
    labels = {'Land': 'Land', 'Water': 'Water', 'Underwater': 'Water'}
    titles = ['spSEP', 'ppSEP 30', 'ppSEP 100']
    keys = ['spSEP', 'PPI30', 'PPI100']
    
    for i, key in enumerate(keys):
        ax = axes[i]
        
        for cond in ['Land', 'Water']:
            arrays = data_store[cond][key]
            if not arrays: continue
            
            stack = np.vstack(arrays)
            mean_wave = np.mean(stack, axis=0)
            sd_wave = np.std(stack, axis=0)
            n_trials = len(arrays)
            
            ax.plot(common_times, mean_wave, color=colors[cond], label=f"{labels[cond]} (n={n_trials})", linewidth=2)
            ax.fill_between(common_times, mean_wave - sd_wave, mean_wave + sd_wave, color=colors[cond], alpha=0.3)
            
            if key == 'spSEP' and cond == 'Land':
                 mask_n20 = (common_times >= 18) & (common_times <= 22)
                 if np.any(mask_n20):
                     target = mean_wave[mask_n20]
                     t_subset = common_times[mask_n20]
                     idx_min = np.argmin(target)
                     n20_t = t_subset[idx_min]
                     n20_v = target[idx_min]
                     ax.plot(n20_t, n20_v, 'v', color='black', zorder=10)
                     ax.text(n20_t, n20_v - 0.7, "N20", ha='center', va='bottom', fontweight='bold')
                     
                 mask_p25 = (common_times >= 26) & (common_times <= 32)
                 if np.any(mask_p25):
                     target = mean_wave[mask_p25]
                     t_subset = common_times[mask_p25]
                     idx_max = np.argmax(target)
                     p25_t = t_subset[idx_max]
                     p25_v = target[idx_max]
                     ax.plot(p25_t, p25_v, '^', color='black', zorder=10)
                     ax.text(p25_t, p25_v + 0.5, "P25", ha='center', va='top', fontweight='bold')
                     
        ax.set_title(titles[i], fontweight='bold')
        ax.set_xlim(PLOT_X_LIM)
        ax.set_ylim(Y_LIM[1], Y_LIM[0])
        ax.axhline(0, color='black', linestyle='--')
        ax.axvline(0, color='black', linestyle='--')
        ax.invert_yaxis()
        ax.set_xlabel("Time (ms)")
        if i == 0: ax.set_ylabel("Amplitude (uV)", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        if i == 2: ax.legend(loc='upper right', fontsize=12)
        
    fig.text(0.5, 0.05, 'Fig. 12 Grand Averaged SEP Waveforms', ha='center', fontweight='bold', fontsize=14)
    plt.subplots_adjust(bottom=0.15)
    
    out_path = os.path.join(OUTPUT_DIR, "SEP_Poster_Comparison_3panel_New.png")
    plt.savefig(out_path, dpi=300)
    print(f"Generated {out_path}")
    plt.close()

if __name__ == "__main__":
    process_all()
