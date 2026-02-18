import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# --- Configuration ---
MEASUREMENT_PATH = '/Users/itoakane/Research/SEP_processed/measurement2 2.xlsx'
TASK_DIR = '/Users/itoakane/Research/task'
OUTPUT_DIR = '/Users/itoakane/Research/SEP_processed/Figures_0216'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_base_id(full_id):
    if '-' in full_id:
        return full_id.split('-')[0]
    return full_id

# 1. Get subjects from measurement2 2.xlsx to ensure consistency
df_measure = pd.read_excel(MEASUREMENT_PATH)
df_measure['sid'] = df_measure['file_id'].astype(str).str[:5]
target_sids = set(df_measure[df_measure['sid'].str.contains('id0') & ~df_measure['sid'].str.contains('id001')]['sid'].unique())

# 2. Process MAE from task folders
results = []
for folder in ['land', 'water']:
    cond_dir = os.path.join(TASK_DIR, folder)
    files_50 = glob.glob(os.path.join(cond_dir, '*+50%.CSV'))
    for f_50 in files_50:
        filename = os.path.basename(f_50)
        id_val = filename.split('+')[0]
        # Keep id002-id013 format to match sid
        base_id = get_base_id(id_val)
        sid = base_id.replace('test', 'id')
        
        if sid not in target_sids:
            continue
            
        f_3mean = f_50.replace('+50%.CSV', '+3MEAN.CSV')
        f_tsk2 = f_50.replace('+50%.CSV', '+tsk2.CSV')
        try:
            max_mean = pd.read_csv(f_3mean, header=6).iloc[0, 5]
            mae_vals = {}
            df_tsk = pd.read_csv(f_tsk2, header=6)
            for p in [10, 30, 50, 70, 90]:
                sub = df_tsk[df_tsk['Stim [%]'] == p]
                mae_vals[f'{p}%MAE'] = ((sub['deg/sec [deg/sec]'] - max_mean*p/100).abs() / max_mean).mean() * 100
            
            results.append({
                'ID': id_val, 'sid': sid, 'Condition': folder.capitalize(),
                **mae_vals
            })
        except Exception as e:
            print(f"Error {id_val}: {e}")

df_res = pd.DataFrame(results)

# 3. Create Plot (Logic from task.py but refined)
percentages = [10, 30, 50, 70, 90]
mae_cols = [f'{p}%MAE' for p in percentages]
means = df_res.groupby('Condition')[mae_cols].mean()
stds = df_res.groupby('Condition')[mae_cols].std()
counts = df_res.groupby('Condition')[mae_cols].count()
ses = stds / np.sqrt(counts)

# PROJECT COLORS (consistent with update_figures_v5_0216.py)
L_COLOR = 'darkorange'
W_COLOR = 'skyblue'

plt.figure(figsize=(7, 7.5))
if 'Land' in means.index:
    plt.errorbar(percentages, means.loc['Land'], yerr=ses.loc['Land'], 
                 label='Land', color=L_COLOR, marker='o', capsize=5, linewidth=3, markersize=8)
if 'Water' in means.index:
    plt.errorbar(percentages, means.loc['Water'], yerr=ses.loc['Water'], 
                 label='Water', color=W_COLOR, marker='o', capsize=5, linewidth=3, markersize=8)

plt.xlabel('Stimulus Speed (%)', fontsize=14, fontweight='bold')
plt.ylabel('MAE (%)', fontsize=14, fontweight='bold')
plt.title('Mean Absolute Error (MAE) by Speed', fontweight='bold', fontsize=16, pad=20)
plt.xticks(percentages, [f'{p}%' for p in percentages], fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(fontsize=12, frameon=True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().set_box_aspect(1)

# Save without fig number
output_path = os.path.join(OUTPUT_DIR, 'All_MAE.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Generated {output_path} with {len(df_res)} data points.")
print(f"Land: {len(df_res[df_res['Condition']=='Land'])} participants")
print(f"Water: {len(df_res[df_res['Condition']=='Water'])} participants")
