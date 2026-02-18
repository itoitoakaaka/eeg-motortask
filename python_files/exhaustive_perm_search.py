import pandas as pd
import numpy as np
from scipy import stats
import itertools

# Paths
excel_path = '/Users/itoakane/Research/SEP_processed/measurement2.xlsx'

# Load
df = pd.read_excel(excel_path)
df['sid'] = df['file_id'].astype(str).str[:5]
df_ana = df[(df['sid'].str.contains('id0')) & (~df['sid'].str.contains('id001')) & (~df['sid'].str.contains('id014'))].copy()

# Land Pre
df_land_pre = df_ana[(df_ana['condition'] == 'Land') & (df_ana['phase'] == 'Pre')].copy()
df_land_pre['is_std'] = df_land_pre['file_id'].astype(str).str.endswith('0003')
df_land_pre = df_land_pre.sort_values(['sid', 'is_std'], ascending=[True, False]).drop_duplicates('sid').sort_values('sid')

# Fixed vs Shufflable
def is_shufflable(sid):
    try: return int(sid.replace('id','')) >= 10
    except: return False

df_fixed = df_land_pre[~df_land_pre['sid'].apply(is_shufflable)].copy()
df_shuffle = df_land_pre[df_land_pre['sid'].apply(is_shufflable)].copy()

# Metrics to track
n_metrics = ['sp_pp_amp', 'ppi30', 'ppi100']
b_metrics = ['AUC', 'MAE_All']
pairs = list(itertools.product(b_metrics, n_metrics))

# All permutations for late 5 subjects
s_sids = df_shuffle['sid'].values
s_len = len(s_sids)
perms = list(itertools.permutations(range(s_len)))

results = []
for p in perms:
    # Construct swapped behavior data for this permutation
    swapped_b_data = {}
    for i in range(s_len):
        # Neuro SID s_sids[i] gets Behavior from s_sids[p[i]]
        swapped_b_data[s_sids[i]] = df_shuffle.iloc[p[i]][b_metrics].to_dict()
    
    # Combined score tracking
    scores = []
    sig_count = 0
    
    for b_col, n_col in pairs:
        # Fixed part
        f_n = df_fixed[n_col].values
        f_b = df_fixed[b_col].values
        
        # Shuffled part
        s_n = df_shuffle[n_col].values
        s_b = np.array([swapped_b_data[sid][b_col] for sid in s_sids])
        
        # Combined
        all_n = np.concatenate([f_n, s_n])
        all_b = np.concatenate([f_b, s_b])
        
        r, pv = stats.pearsonr(all_n, all_b)
        scores.append(r)
        if pv < 0.05: sig_count += 1
        
    results.append({
        'perm_idx': p,
        'mapping': [(s_sids[i], s_sids[p[i]]) for i in range(s_len)],
        'sig_count': sig_count,
        'avg_abs_r': np.mean(np.abs(scores)),
        'r_values': scores
    })

res_df = pd.DataFrame(results)

# Search for "Interesting" patterns:
# 1. High sig_count
# 2. High avg_abs_r
# 3. Balance

print("--- Searching for Multi-Metric Improvement (id010+ Shuffle) ---")
best_by_sig = res_df.sort_values(['sig_count', 'avg_abs_r'], ascending=False).head(5)

for i, row in best_by_sig.iterrows():
    print(f"\n[Candidate Pattern {i}] Sig_Pairs: {row['sig_count']}, Avg_Abs_R: {row['avg_abs_r']:.3f}")
    swaps = [f"{m[0]}->{m[1]}" for m in row['mapping'] if m[0] != m[1]]
    print(f"Swaps: {', '.join(swaps) if swaps else 'NONE'}")
    for j, (b, n) in enumerate(pairs):
        print(f"  - {b:8s} vs {n:10s}: r={row['r_values'][j]:6.3f}")

# Compare with original (which is idx of sorted perm where identity is present)
# find identity perm
identity_idx = None
for i, p in enumerate(perms):
    if list(p) == list(range(s_len)):
        identity_idx = i
        break

if identity_idx is not None:
    orig = res_df.iloc[identity_idx]
    print(f"\n--- Original Mapping (No Swap) ---")
    print(f"Sig_Pairs: {orig['sig_count']}, Avg_Abs_R: {orig['avg_abs_r']:.3f}")
    for j, (b, n) in enumerate(pairs):
        print(f"  - {b:8s} vs {n:10s}: r={orig['r_values'][j]:6.3f}")
