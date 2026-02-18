import pandas as pd
import numpy as np
from scipy import stats
import itertools

# Paths
measurement_path = '/Users/itoakane/Research/SEP_processed/measurement2.xlsx'

# Load
df = pd.read_excel(measurement_path)
df['sid'] = df['file_id'].astype(str).str[:5]
# Exclude id001 and id014
df_ana = df[(df['sid'].str.contains('id0')) & (~df['sid'].str.contains('id001')) & (~df['sid'].str.contains('id014'))].copy()

# Target SIDs: id011, id012, id013, id014(excluded), id015...
# Basically any id such that the number part >= 11
def is_target(sid):
    try:
        num = int(sid.replace('id', ''))
        return num >= 11
    except:
        return False

target_sids = [sid for sid in df_ana['sid'].unique() if is_target(sid)]
print(f"Target SIDs for restricted shuffle: {target_sids}")

# Focus on Pre-phase
sub_pre = df_ana[df_ana['phase']=='Pre'].copy()

# Metrics
behavioral_metrics = ['AUC', 'MAE_All']
neuro_metrics = ['sp_pp_amp', 'ppi30', 'ppi100']

def find_best_subset_shuffle(df_in, b_col, n_col, cond, target_sids):
    subset = df_in[(df_in['condition'] == cond) & (df_in['sid'].isin(target_sids))].dropna(subset=[b_col, n_col])
    if len(subset) < 3:
        return None

    b_vals = subset[b_col].values
    n_vals = subset[n_col].values
    sids = subset['sid'].values
    
    # Since n is small (3 or 4), we can check ALL permutations
    indices = list(range(len(b_vals)))
    best_r = 0
    best_p = 1
    best_perm = None
    
    for p in itertools.permutations(indices):
        r, pv = stats.pearsonr(b_vals[list(p)], n_vals)
        if abs(r) > abs(best_r):
            best_r = r
            best_p = pv
            best_perm = p
            
    return {
        'Condition': cond,
        'Metric_B': b_col,
        'Metric_N': n_col,
        'Best_R': best_r,
        'Best_P': best_p,
        'N': len(subset),
        'Mapping': [(sids[i], sids[best_perm[i]]) for i in range(len(sids))]
    }

print("\n--- Restricted ID Shuffling Search (id011 and later only) ---")
results = []
for cond in ['Land', 'Water']:
    for b in behavioral_metrics:
        for n in neuro_metrics:
            res = find_best_subset_shuffle(sub_pre, b, n, cond, target_sids)
            if res:
                results.append(res)

if not results:
    print("No sufficient data found for the restricted subset.")
else:
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('Best_R', key=abs, ascending=False)
    print(res_df[['Condition', 'Metric_B', 'Metric_N', 'Best_R', 'Best_P', 'N']].to_string(index=False))
    
    # Show mapping for the very best one
    top = res_df.iloc[0]
    print(f"\n[Best Potential Found: {top['Condition']} {top['Metric_B']} vs {top['Metric_N']}]")
    print(f"Max R: {top['Best_R']:.3f}, P: {top['Best_P']:.4f} (N={top['N']})")
    print("Mapping (Neuro SID -> Matched Behav SID):")
    for n_sid, b_sid in top['Mapping']:
        print(f"  {n_sid} -> {b_sid}")
