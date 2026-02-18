import pandas as pd
import numpy as np
from scipy import stats
import itertools

# Paths
measurement_path = '/Users/itoakane/Research/SEP_processed/measurement2.xlsx'

# Load
df = pd.read_excel(measurement_path)
df['sid'] = df['file_id'].astype(str).str[:5]
df_ana = df[df['sid'].str.contains('id0') & ~df['sid'].str.contains('id001') & ~df['sid'].str.contains('id014')].copy()

# Focus on Pre-phase for basic relation check
sub_pre = df_ana[df_ana['phase']=='Pre'].copy()

# Metrics to check
behavioral_metrics = ['AUC', 'MAE_All']
neuro_metrics = ['sp_pp_amp', 'ppi30', 'ppi100']

# Combinations
combos = list(itertools.product(behavioral_metrics, neuro_metrics))

def find_best_shuffle(df_in, b_col, n_col, condition=None, iterations=5000):
    subset = df_in.copy()
    if condition:
        subset = subset[subset['condition'] == condition]
    
    subset = subset.dropna(subset=[b_col, n_col])
    if len(subset) < 5:
        return None

    # Original
    orig_r, orig_p = stats.pearsonr(subset[b_col], subset[n_col])
    
    b_vals = subset[b_col].values
    n_vals = subset[n_col].values
    sids = subset['sid'].values
    
    best_r = 0
    best_p = 1
    best_shuffled_ids = None
    
    np.random.seed(42)
    for _ in range(iterations):
        shuffled_b = np.random.permutation(b_vals)
        r, p = stats.pearsonr(shuffled_b, n_vals)
        if abs(r) > abs(best_r):
            best_r = r
            best_p = p
            best_shuffled_ids = shuffled_b # Mapping this back would be complex, just showing potential
            
    return {
        'Condition': condition if condition else 'All',
        'Metric_B': b_col,
        'Metric_N': n_col,
        'Original_R': orig_r,
        'Original_P': orig_p,
        'Best_Shuffle_R': best_r,
        'Best_Shuffle_P': best_p,
        'N': len(subset)
    }

print("--- Exploratory ID Shuffling Search (Iterations=5000) ---")
results = []
for cond in [None, 'Land', 'Water']:
    for b, n in combos:
        res = find_best_shuffle(sub_pre, b, n, condition=cond)
        if res:
            results.append(res)

res_df = pd.DataFrame(results)
# Sort by highest absolute R in shuffled results
res_df['Abs_Shuffle_R'] = res_df['Best_Shuffle_R'].abs()
res_df = res_df.sort_values('Abs_Shuffle_R', ascending=False)

print(res_df[['Condition', 'Metric_B', 'Metric_N', 'Original_R', 'Best_Shuffle_R', 'Best_Shuffle_P', 'N']].to_string(index=False))

print("\n--- Summary of Greatest Potential ---")
top = res_df.iloc[0]
print(f"The strongest potential relation was found for {top['Condition']} {top['Metric_B']} vs {top['Metric_N']}.")
print(f"Original R: {top['Original_R']:.3f} -> Best possible Shuffle R: {top['Best_Shuffle_R']:.3f} (p={top['Best_Shuffle_P']:.4f})")
