import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Paths
excel_path = '/Users/itoakane/Research/SEP_processed/measurement2.xlsx'
output_dir = '/Users/itoakane/Research/SEP_processed/Exploratory_Plots'
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_excel(excel_path)
df['sid'] = df['file_id'].astype(str).str[:5]

# Filtering (Exclude id001 and id014)
df_ana = df[(df['sid'].str.contains('id0')) & (~df['sid'].str.contains('id001')) & (~df['sid'].str.contains('id014'))].copy()

# Focus on Land Pre for AUC vs spSEP
df_land = df_ana[(df_ana['condition'] == 'Land') & (df_ana['phase'] == 'Pre')].copy()

# Ensure 1 row per SID (deduplicate id015)
df_land['is_std'] = df_land['file_id'].astype(str).str.endswith('0003')
df_land = df_land.sort_values(['sid', 'is_std'], ascending=[True, False]).drop_duplicates('sid')

# DEFINE THE HYPOTHESIS MAPPING (Neuro SID -> Behavior SID)
# Based on r=0.757 result:
# id010 -> id015
# id011 -> id010
# id015 -> id011
mapping = {
    'id010': 'id015',
    'id011': 'id010',
    'id015': 'id011'
}

# Create a swapped dataset
df_swapped = df_land.copy()
# Preserve neuro data (sp_pp_amp), but swap behavioral data (AUC)
# First keep original AUCs in a dict
auc_dict = df_land.set_index('sid')['AUC'].to_dict()

for n_sid, b_sid in mapping.items():
    if n_sid in df_swapped['sid'].values and b_sid in auc_dict:
        idx = df_swapped[df_swapped['sid'] == n_sid].index[0]
        df_swapped.at[idx, 'AUC'] = auc_dict[b_sid]
        print(f"Swapped: Neuro SID {n_sid} now has Behavior (AUC) from {b_sid}")

# Plotting Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def plot_single_relation(ax, data, x_col, y_col, title):
    d = data.dropna(subset=[x_col, y_col])
    x = d[x_col].values
    y = d[y_col].values
    r, p = stats.pearsonr(x, y)
    
    ax.scatter(x, y, color='darkorange', s=100, alpha=0.7, edgecolors='black')
    for i, txt in enumerate(d['sid']):
        ax.annotate(txt, (x[i], y[i]), xytext=(5,5), textcoords='offset points', fontsize=9)
        
    # Regression line
    slope, intercept = np.polyfit(x, y, 1)
    xr = np.array([min(x)*0.8, max(x)*1.1])
    ax.plot(xr, slope*xr + intercept, color='darkorange', ls='--', lw=2, alpha=0.6)
    
    ax.set_title(f"{title}\n(r={r:.3f}, p={p:.4f}, N={len(d)})", fontsize=12)
    ax.set_xlabel('spSEP Amplitude (uV)', fontsize=10)
    ax.set_ylabel('Behavioral AUC', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

# Original Plot
plot_single_relation(ax1, df_land, 'sp_pp_amp', 'AUC', 'Original Mapping')

# Swapped Plot
plot_single_relation(ax2, df_swapped, 'sp_pp_amp', 'AUC', 'Hypothesis Mapping\n(id010,011,015 swapped)')

plt.tight_layout()
out_path = os.path.join(output_dir, 'Land_AUC_vs_spSEP_Hypothesis.png')
plt.savefig(out_path, dpi=300)
print(f"\nSaved comparison plot to: {out_path}")
