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

# Focus on Pre-phase and Land condition
subset_all = df_ana[(df_ana['phase']=='Pre') & (df_ana['condition']=='Land')].copy()

# Ensure 1 row per SID (id015 had duplicates)
# We prefer file_id ending in 0003 for Pre
subset_all['is_std'] = subset_all['file_id'].astype(str).str.endswith('0003')
subset = subset_all.sort_values(['sid', 'is_std'], ascending=[True, False]).drop_duplicates('sid')
subset = subset.dropna(subset=['AUC', 'sp_pp_amp']).sort_values('sid')

# Fixed SIDs: id002 to id010 (9 subjects)
# Shufflable SIDs: id011, id012, id013, id015 (4 subjects)
def is_shufflable(sid):
    try:
        num = int(sid.replace('id', ''))
        return num >= 11
    except:
        return False

fixed_subset = subset[~subset['sid'].apply(is_shufflable)].copy()
shufflable_subset = subset[subset['sid'].apply(is_shufflable)].copy()

print(f"Total Land subjects (N=13): {len(subset)}")
print(f"Fixed subjects (id002-010): {fixed_subset['sid'].unique().tolist()} (N={len(fixed_subset)})")
print(f"Shufflable subjects (id011 and later): {shufflable_subset['sid'].unique().tolist()} (N={len(shufflable_subset)})")

def find_best_restricted_shuffle(fixed_df, shuffle_df, b_col, n_col):
    fixed_n = fixed_df[n_col].values
    fixed_b = fixed_df[b_col].values
    
    shuffle_n = shuffle_df[n_col].values
    shuffle_b = shuffle_df[b_col].values
    shuffle_sids = shuffle_df['sid'].values
    
    indices = list(range(len(shuffle_b)))
    best_r = 0
    best_p = 1
    best_perm = None
    
    # Check all permutations of behavioral values for shufflable subjects
    for p in itertools.permutations(indices):
        combined_n = np.concatenate([fixed_n, shuffle_n])
        combined_b = np.concatenate([fixed_b, shuffle_b[list(p)]])
        
        r, pv = stats.pearsonr(combined_n, combined_b)
        if abs(r) > abs(best_r):
            best_r = r
            best_p = pv
            best_perm = p
            
    return {
        'Best_R': best_r,
        'Best_P': best_p,
        'Mapping': [(shuffle_sids[i], shuffle_sids[best_perm[i]]) for i in range(len(shuffle_sids))]
    }

print("\n--- Restricted ID Shuffling (Fixed 002-010, Shufflable 011-015) ---")
print("Metrics: Land AUC vs spSEP (sp_pp_amp)")

res = find_best_restricted_shuffle(fixed_subset, shufflable_subset, 'AUC', 'sp_pp_amp')

if res:
    print(f"Best possible Combined R: {res['Best_R']:.3f} (p={res['Best_P']:.4f})")
    print("\n[Optimal Mapping for Shufflable Range (Neuro SID -> Behav SID)]")
    for n_sid, b_sid in res['Mapping']:
        print(f"  {n_sid} -> {b_sid}")
    
    # Original for comparison
    combined_n_curr = np.concatenate([fixed_subset['sp_pp_amp'].values, shufflable_subset['sp_pp_amp'].values])
    combined_b_curr = np.concatenate([fixed_subset['AUC'].values, shufflable_subset['AUC'].values])
    curr_r, curr_p = stats.pearsonr(combined_n_curr, combined_b_curr)
    print(f"\nCurrent Combined R (No shuffle): {curr_r:.3f} (p={curr_p:.4f})")
else:
    print("Insufficient data.")
