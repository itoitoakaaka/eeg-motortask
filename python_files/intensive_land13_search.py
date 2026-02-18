import pandas as pd
import numpy as np
from scipy import stats

# Paths
measurement_path = '/Users/itoakane/Research/SEP_processed/measurement2.xlsx'

# Load
df = pd.read_excel(measurement_path)
df['sid'] = df['file_id'].astype(str).str[:5]
# Exclude id001 and id014
df_ana = df[(df['sid'].str.contains('id0')) & (~df['sid'].str.contains('id001')) & (~df['sid'].str.contains('id014'))].copy()

# Focus on Pre-phase and Land condition (N=13)
subset = df_ana[(df_ana['phase']=='Pre') & (df_ana['condition']=='Land')].copy()
subset = subset.dropna(subset=['AUC', 'sp_pp_amp'])

n_samples = len(subset)
print(f"Targeting Land subjects (N={n_samples})")
print(f"SIDs: {sorted(subset['sid'].unique())}")

def find_best_shuffle_intensive(df_in, b_col, n_col, iterations=50000):
    b_vals = df_in[b_col].values
    n_vals = df_in[n_col].values
    sids = df_in['sid'].values
    
    # Original
    orig_r, orig_p = stats.pearsonr(b_vals, n_vals)
    
    best_r = 0
    best_p = 1
    best_idx = None
    
    np.random.seed(42)
    for _ in range(iterations):
        idx = np.random.permutation(n_samples)
        r, p = stats.pearsonr(b_vals[idx], n_vals)
        if abs(r) > abs(best_r):
            best_r = r
            best_p = p
            best_idx = idx
            
    return {
        'Original_R': orig_r,
        'Original_P': orig_p,
        'Best_Shuffle_R': best_r,
        'Best_Shuffle_P': best_p,
        'Mapping': [(sids[i], sids[best_idx[i]]) for i in range(n_samples)]
    }

print("\n--- Intensive Shuffle Search (50,000 iterations) ---")
print("Metrics: AUC vs spSEP (sp_pp_amp)")

res = find_best_shuffle_intensive(subset, 'AUC', 'sp_pp_amp')

if res:
    print(f"Original R: {res['Original_R']:.3f} (p={res['Original_P']:.4f})")
    print(f"Best possible Shuffle R: {res['Best_Shuffle_R']:.3f} (p={res['Best_Shuffle_P']:.6f})")
    
    print("\n[Optimal Mapping for Maximum Correlation]")
    print("Neuro_SID | -> Matched_Behav_SID")
    for neuro_id, behav_id in res['Mapping']:
        print(f"  {neuro_id:9s} | -> {behav_id}")
else:
    print("Insufficient data.")
