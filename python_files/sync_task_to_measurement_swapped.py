import pandas as pd
import numpy as np

# Paths
measurement_path = '/Users/itoakane/Research/SEP_processed/measurement2.xlsx'
task_path = '/Users/itoakane/Research/task/task.xlsx'
swapped_output_path = '/Users/itoakane/Research/SEP_processed/measurement2_swapped.xlsx'

# Load
df_m = pd.read_excel(measurement_path)
df_t = pd.read_excel(task_path)

# Prepare mapping key in task.xlsx (ID mapping from task.xlsx)
df_t['sid'] = df_t['BaseID'].str.replace('test', 'id')

# Create lookup dictionary: {(sid, Condition): (AUC, MAE_50, MAE_All)}
lookup = {}
for _, row in df_t.iterrows():
    lookup[(row['sid'], row['Condition'])] = (row['AUC'], row['50%MAE'], row['Overall_MAE'])

# ID SWAP HYPOTHESIS (Applied ONLY to Land)
# Neuro SID mapping to Behavioral SID
# Based on earlier finding: id010->id015, id011->id010, id015->id011
swaps = {
    'id010': 'id015',
    'id011': 'id010',
    'id015': 'id011'
}

# Update measurement2.xlsx
df_m['sid'] = df_m['file_id'].astype(str).str[:5]

# Track metrics for new rows if they don't exist in existing df_m
available_pairs = set(zip(df_m['sid'], df_m['condition']))

updated_count = 0
for i, row in df_m.iterrows():
    sid = row['sid']
    cond = row['condition']
    
    # Determine which behavioral SID to use
    if cond == 'Land' and sid in swaps:
        behav_sid = swaps[sid]
    else:
        behav_sid = sid
    
    key = (behav_sid, cond)
    if key in lookup:
        auc, mae50, mae_all = lookup[key]
        df_m.at[i, 'AUC'] = auc
        df_m.at[i, 'MAE_50'] = mae50
        df_m.at[i, 'MAE_All'] = mae_all
        updated_count += 1

# Check for rows that might be missing in df_m but present in lookup (considering swap for Land)
# We need to make sure neuro SIDs from lookup are represented
new_rows = []
for (behav_sid, cond), (auc, mae50, mae_all) in lookup.items():
    # Reverse lookup for Land: if this behav_sid is part of a swap, which neuro_sid corresponds to it?
    # Actually, easier to just check which (neuro_sid, cond) pairs should exist
    # If neuro_sid is in swaps, the behav_sid is the target of the swap
    
    # Find neuro_sid such that neuro_sid maps to current behav_sid for Land
    target_neuro_sid = None
    if cond == 'Land':
        # Find k such that swaps[k] == behav_sid
        for k, v in swaps.items():
            if v == behav_sid:
                target_neuro_sid = k
                break
        if target_neuro_sid is None:
            # If not in swaps, it's 1:1
            if behav_sid not in swaps: target_neuro_sid = behav_sid
    else:
        target_neuro_sid = behav_sid
        
    if target_neuro_sid and (target_neuro_sid, cond) not in available_pairs:
        new_row = {
            'file_id': f"{target_neuro_sid}0001",
            'condition': cond,
            'phase': 'Pre',
            'AUC': auc,
            'MAE_50': mae50,
            'MAE_All': mae_all,
            'sid': target_neuro_sid
        }
        new_rows.append(new_row)
        available_pairs.add((target_neuro_sid, cond))
        print(f"Adding new mapped subject-condition: {target_neuro_sid} {cond} (using Behav from {behav_sid})")

if new_rows:
    df_new = pd.DataFrame(new_rows)
    df_m = pd.concat([df_m, df_new], ignore_index=True)

# Save back to a NEW file
if 'sid' in df_m.columns:
    df_m.drop(columns=['sid'], inplace=True)

df_m.to_excel(swapped_output_path, index=False)
print(f"Updated {swapped_output_path} with swapped ID mapping for Land 10/11/15.")
print(f"Total rows updated/created: {updated_count + len(new_rows)}")
