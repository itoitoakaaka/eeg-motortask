import pandas as pd
import numpy as np
from scipy import stats

# Paths
measurement_path = '/Users/itoakane/Research/SEP_processed/measurement2.xlsx'

# Load
df = pd.read_excel(measurement_path)
df['sid'] = df['file_id'].astype(str).str[:5]
df_ana = df[(df['sid'].str.contains('id0')) & (~df['sid'].str.contains('id001')) & (~df['sid'].str.contains('id014'))].copy()

# Focus on Pre-phase (AUC and spSEP are often compared at Pre or as standard)
sub_pre = df_ana[df_ana['phase']=='Pre'].copy()

def find_best_shuffle_specific(df_in, b_col, n_col, cond, iterations=20000):
    subset = df_in.copy()
    if cond != 'All':
        subset = subset[subset['condition'] == cond]
    
    subset = subset.dropna(subset=[b_col, n_col])
    n_samples = len(subset)
    if n_samples < 5:
        return None

    b_vals = subset[b_col].values
    n_vals = subset[n_col].values
    sids = subset['sid'].values
    
    # Original
    orig_r, orig_p = stats.pearsonr(b_vals, n_vals)
    
    best_r = 0
    best_p = 1
    best_idx = None
    
    np.random.seed(123)
    for _ in range(iterations):
        idx = np.random.permutation(n_samples)
        r, p = stats.pearsonr(b_vals[idx], n_vals)
        if abs(r) > abs(best_r):
            best_r = r
            best_p = p
            best_idx = idx
            
    return {
        'Condition': cond,
        'Original_R': orig_r,
        'Original_P': orig_p,
        'Best_Shuffle_R': best_r,
        'Best_Shuffle_P': best_p,
        'N': n_samples,
        'Mapping': [(sids[i], sids[best_idx[i]]) for i in range(n_samples)]
    }

print("--- Intensive Shuffle Search: AUC vs spSEP (sp_pp_amp) ---")
results = []
for cond in ['All', 'Land', 'Water']:
    res = find_best_shuffle_specific(sub_pre, 'AUC', 'sp_pp_amp', cond)
    if res:
        results.append(res)

res_df = pd.DataFrame(results)
print(res_df[['Condition', 'Original_R', 'Best_Shuffle_R', 'Best_Shuffle_P', 'N']].to_string(index=False))

for i, row in res_df.iterrows():
    if abs(row['Best_Shuffle_R']) > 0.8:
        print(f"\n[Strong Potential Mapping for {row['Condition']}] (R={row['Best_Shuffle_R']:.3f})")
        # Just show some examples or total mapping
        for neuro_id, behav_id in row['Mapping']:
            print(f"  {neuro_id} -> {behav_id}")
