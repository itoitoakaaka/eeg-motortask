import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# --- Configuration ---
MEASUREMENT_PATH = '/Users/itoakane/Research/SEP_processed/measurement2_swapped.xlsx'
TASK_DIR = '/Users/itoakane/Research/task'
OUTPUT_DIR = '/Users/itoakane/Research/SEP_processed/Figures_0217'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Neuro ID -> Behavior ID suffix mapping for Land
# Hypothesis: Neuro id010 -> Behav test015, Neuro id011 -> Behav test010, Neuro id015 -> Behav test011
swaps_neuro_to_behav = {
    'id010': '015',
    'id011': '010',
    'id015': '011'
}

def get_base_id(full_id):
    if '-' in full_id:
        return full_id.split('-')[0]
    return full_id

# 1. Get target subjects
df_measure = pd.read_excel(MEASUREMENT_PATH)
df_measure['sid'] = df_measure['file_id'].astype(str).str[:5]
target_sids = set(df_measure[df_measure['sid'].str.contains('id0') & 
                             ~df_measure['sid'].str.contains('id001') & 
                             ~df_measure['sid'].str.contains('id014')]['sid'].unique())

# 2. Process MAE
results = []
for folder in ['land', 'water']:
    cond_dir = os.path.join(TASK_DIR, folder)
    # Search for all +50%.CSV files in the folder
    files_50 = glob.glob(os.path.join(cond_dir, '*+50%.CSV'))
    
    for f_50 in files_50:
        filename = os.path.basename(f_50)
        behav_id_full = filename.split('+')[0]
        behav_id_base = get_base_id(behav_id_full)
        behav_num = behav_id_base.replace('test', '')
        
        # Determine which Neuro SID this Behavior data belongs to
        # By default, Behav testXXX belongs to Neuro idXXX
        neuro_sid = f"id{behav_num}"
        
        # APPLY THE SWAP for Land
        if folder == 'land':
            # Reverse mapping: which neuro_sid should use this behav_id?
            # behav 015 -> neuro 010, behav 010 -> neuro 011, behav 011 -> neuro 015
            reverse_swaps = {'015': 'id010', '010': 'id011', '011': 'id015'}
            if behav_num in reverse_swaps:
                neuro_sid = reverse_swaps[behav_num]
        
        if neuro_sid not in target_sids:
            continue
            
        f_3mean = f_50.replace('+50%.CSV', '+3MEAN.CSV')
        f_tsk2 = f_50.replace('+50%.CSV', '+tsk2.CSV')
        
        if not os.path.exists(f_3mean) or not os.path.exists(f_tsk2):
            continue
            
        try:
            max_mean = pd.read_csv(f_3mean, header=6).iloc[0, 5]
            mae_vals = {}
            df_tsk = pd.read_csv(f_tsk2, header=6)
            for p in [10, 30, 50, 70, 90]:
                sub = df_tsk[df_tsk['Stim [%]'] == p]
                if not sub.empty:
                    mae_vals[f'{p}%MAE'] = ((sub['deg/sec [deg/sec]'] - max_mean*p/100).abs() / max_mean).mean() * 100
            
            if mae_vals:
                results.append({
                    'Neuro_SID': neuro_sid, 'Behav_ID': behav_id_full, 'Condition': folder.capitalize(),
                    **mae_vals
                })
        except Exception as e:
            print(f"Error processing {behav_id_full}: {e}")

df_res = pd.DataFrame(results)

# 3. Create Plot
percentages = [10, 30, 50, 70, 90]
mae_cols = [f'{p}%MAE' for p in percentages]
if not df_res.empty:
    means = df_res.groupby('Condition')[mae_cols].mean()
    stds = df_res.groupby('Condition')[mae_cols].std()
    counts = df_res.groupby('Condition')[mae_cols].count()
    ses = stds / np.sqrt(counts)

    plt.figure(figsize=(7, 7.5))
    L_COLOR, W_COLOR = 'darkorange', 'skyblue'
    
    if 'Land' in means.index:
        plt.errorbar(percentages, means.loc['Land'], yerr=ses.loc['Land'], 
                     label='Land', color=L_COLOR, marker='o', capsize=5, linewidth=3, markersize=8)
    if 'Water' in means.index:
        plt.errorbar(percentages, means.loc['Water'], yerr=ses.loc['Water'], 
                     label='Water', color=W_COLOR, marker='o', capsize=5, linewidth=3, markersize=8)

    plt.xlabel('Stimulus Speed (%)', fontsize=14, fontweight='bold')
    plt.ylabel('MAE (%)', fontsize=14, fontweight='bold')
    plt.title('MAE by Speed (Figures_0217_v2)', fontweight='bold', fontsize=16, pad=20)
    plt.xticks(percentages, [f'{p}%' for p in percentages], fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=12, frameon=True)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_box_aspect(1)

    out_path = os.path.join(OUTPUT_DIR, 'All_MAE.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path} with {len(df_res)} data points.")
else:
    print("No data collected for MAE plot.")
