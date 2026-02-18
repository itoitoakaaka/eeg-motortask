import pandas as pd
from scipy import stats
import numpy as np

# Load the original measurement2.xlsx
df = pd.read_excel('/Users/itoakane/Research/SEP_processed/measurement2.xlsx')
df['sid'] = df['file_id'].astype(str).str[:5]

# Filtering Land Pre-phase
df_land = df[(df['sid'].str.contains('id0')) & 
              (~df['sid'].str.contains('id001')) & 
              (~df['sid'].str.contains('id014')) & 
              (df['condition'] == 'Land') & 
              (df['phase'] == 'Pre')].copy()

# Deduplicate
df_land['is_std'] = df_land['file_id'].astype(str).str.endswith('0003')
df_land = df_land.sort_values(['sid', 'is_std'], ascending=[True, False]).drop_duplicates('sid')

n_metrics = ['sp_pp_amp', 'ppi30', 'ppi100']
b_metrics = ['AUC', 'MAE_All']

# SWAP HYPOTHESIS
# Neuro SID mapping to Behavioral SID
mapping = {
    'id010': 'id015',
    'id011': 'id010',
    'id015': 'id011'
}

def get_corrs(data, is_swapped=False):
    d_work = data.copy()
    if is_swapped:
        # Get behavior dict
        auc_dict = d_work.set_index('sid')['AUC'].to_dict()
        mae_dict = d_work.set_index('sid')['MAE_All'].to_dict()
        
        for neuro_sid, behav_sid in mapping.items():
            if neuro_sid in d_work['sid'].values:
                idx = d_work[d_work['sid'] == neuro_sid].index[0]
                d_work.at[idx, 'AUC'] = auc_dict.get(behav_sid, np.nan)
                d_work.at[idx, 'MAE_All'] = mae_dict.get(behav_sid, np.nan)
    
    res = {}
    for b in b_metrics:
        for n in n_metrics:
            subset = d_work.dropna(subset=[b, n])
            if len(subset) > 2:
                r, p = stats.pearsonr(subset[b], subset[n])
                res[(b, n)] = (r, p, len(subset))
            else:
                res[(b, n)] = (np.nan, np.nan, len(subset))
    return res

orig_res = get_corrs(df_land, is_swapped=False)
swap_res = get_corrs(df_land, is_swapped=True)

print("=== Land Correlation Comparison: Original vs Swapped (10, 11, 15) ===")
print(f"{'Metric Pair':25s} | {'Original R (p)':20s} | {'Swapped R (p)':20s} | Impact")
print("-" * 90)

for pair in orig_res.keys():
    b_col, n_col = pair
    r0, p0, n0 = orig_res[pair]
    r1, p1, n1 = swap_res[pair]
    
    impact = ""
    if not np.isnan(r1):
        if abs(r1) > abs(r0) + 0.1: impact = "Improve ↑"
        elif abs(r1) < abs(r0) - 0.1: impact = "Worsen ↓"
    
    pair_str = f"{b_col} vs {n_col.replace('sp_pp_amp', 'spSEP')}"
    s0 = f"{r0:6.3f} ({p0:.3f})" if not np.isnan(r0) else "N/A"
    s1 = f"{r1:6.3f} ({p1:.3f})" if not np.isnan(r1) else "N/A"
    
    print(f"{pair_str:25s} | {s0:20s} | {s1:20s} | {impact}")
