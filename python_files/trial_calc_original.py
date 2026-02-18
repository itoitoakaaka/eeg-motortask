import pandas as pd
from scipy import stats
import numpy as np

# Load the original measurement2.xlsx
df = pd.read_excel('/Users/itoakane/Research/SEP_processed/measurement2.xlsx')
df['sid'] = df['file_id'].astype(str).str[:5]

# Filtering
df_ana = df[(df['sid'].str.contains('id0')) & 
            (~df['sid'].str.contains('id001')) & 
            (~df['sid'].str.contains('id014'))].copy()

n_metrics = ['sp_pp_amp', 'ppi30', 'ppi100']
b_metrics = ['AUC', 'MAE_All']

print("=== Trial Calculation: Original measurement2.xlsx ===")

for cond in ['Land', 'Water']:
    # Get Pre phase for cross-sectional correlation
    sub = df_ana[(df_ana['phase']=='Pre') & (df_ana['condition']==cond)].copy()
    
    # Deduplicate id015 for Land
    sub['is_std'] = sub['file_id'].astype(str).str.endswith('0003')
    sub = sub.sort_values(['sid', 'is_std'], ascending=[True, False]).drop_duplicates('sid')
    
    print(f"\n--- {cond} (N={len(sub.dropna(subset=['AUC', 'sp_pp_amp']))}) ---")
    for b in b_metrics:
        for n in n_metrics:
            d = sub.dropna(subset=[b, n])
            if len(d) > 2:
                r, p = stats.pearsonr(d[b], d[n])
                print(f"{b:8s} vs {n:10s} | r={r:6.3f} | p={p:.4f}")
            else:
                print(f"{b:8s} vs {n:10s} | Insufficient data")
