import pandas as pd
import numpy as np
from scipy import stats

# Paths
excel_path = '/Users/itoakane/Research/SEP_processed/measurement2.xlsx'

# Load
df = pd.read_excel(excel_path)
df['sid'] = df['file_id'].astype(str).str[:5]
df_ana = df[(df['sid'].str.contains('id0')) & (~df['sid'].str.contains('id001')) & (~df['sid'].str.contains('id014'))].copy()

# Land Pre for AUC/spSEP comparisons
df_land_pre = df_ana[(df_ana['condition'] == 'Land') & (df_ana['phase'] == 'Pre')].copy()
df_land_pre['is_std'] = df_land_pre['file_id'].astype(str).str.endswith('0003')
df_land_pre = df_land_pre.sort_values(['sid', 'is_std'], ascending=[True, False]).drop_duplicates('sid')

# DEFINE THE SWAP
# Neuro SID -> Behavior SID
mapping = {
    'id010': 'id015',
    'id011': 'id010',
    'id015': 'id011'
}

def check_impact(data, n_metrics, b_metrics):
    auc_dict = data.set_index('sid')['AUC'].to_dict()
    mae_dict = data.set_index('sid')['MAE_All'].to_dict()
    
    # Swapped data
    d_swapped = data.copy()
    for n_sid, b_sid in mapping.items():
        if n_sid in d_swapped['sid'].values:
            idx = d_swapped[d_swapped['sid'] == n_sid].index[0]
            if b_sid in auc_dict: d_swapped.at[idx, 'AUC'] = auc_dict[b_sid]
            if b_sid in mae_dict: d_swapped.at[idx, 'MAE_All'] = mae_dict[b_sid]

    print(f"{'Metric Pair':35s} | {'Original R (p)':20s} | {'Swapped R (p)':20s}")
    print("-" * 80)
    
    current_results = []
    for n in n_metrics:
        for b in b_metrics:
            # Original
            d0 = data.dropna(subset=[n, b])
            r0, p0 = stats.pearsonr(d0[n], d0[b])
            
            # Swapped
            d1 = d_swapped.dropna(subset=[n, b])
            r1, p1 = stats.pearsonr(d1[n], d1[b])
            
            status = "UP! ↑" if abs(r1) > abs(r0) + 0.1 else ""
            print(f"{b + ' vs ' + n:35s} | {r0:6.3f} ({p0:.4f}) | {r1:6.3f} ({p1:.4f}) {status}")

n_list = ['sp_pp_amp', 'ppi30', 'ppi100']
b_list = ['AUC', 'MAE_All']

print("--- Impact Assessment of ID Swap (id010/011/015) in Land ---")
check_impact(df_land_pre, n_list, b_list)
