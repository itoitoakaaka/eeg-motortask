import pandas as pd
import numpy as np

# Paths
measurement_path = '/Users/itoakane/Research/SEP_processed/measurement2.xlsx'
task_path = '/Users/itoakane/Research/task/task.xlsx'

# Load
df_m = pd.read_excel(measurement_path)
df_t = pd.read_excel(task_path)

# sid helper
df_t['sid'] = df_t['BaseID'].str.replace('test', 'id')

# Create lookup
lookup = {}
for _, row in df_t.iterrows():
    lookup[(row['sid'], row['Condition'])] = (row['AUC'], row['50%MAE'], row['Overall_MAE'])

# Prepare to track additions
df_m['sid'] = df_m['file_id'].astype(str).str[:5]

# Update existing
for i, row in df_m.iterrows():
    key = (row['sid'], row['condition'])
    if key in lookup:
        auc, mae50, mae_all = lookup[key]
        df_m.at[i, 'AUC'] = auc
        df_m.at[i, 'MAE_50'] = mae50
        df_m.at[i, 'MAE_All'] = mae_all

# Add missing (sid, condition) pairs
existing_pairs = set(zip(df_m['sid'], df_m['condition']))

new_rows = []
for (sid, cond), (auc, mae50, mae_all) in lookup.items():
    if (sid, cond) not in existing_pairs:
        # Create a new row (or two for Pre/Post, but at least Pre for behavioral)
        # We'll add it as 'Pre' since Fig 1 uses 'Pre'
        new_row = {
            'file_id': f"{sid}0001", # dummy placeholder
            'condition': cond,
            'phase': 'Pre',
            'AUC': auc,
            'MAE_50': mae50,
            'MAE_All': mae_all
        }
        new_rows.append(new_row)
        print(f"Adding new subject-condition: {sid} {cond}")

if new_rows:
    df_new = pd.DataFrame(new_rows)
    df_m = pd.concat([df_m, df_new], ignore_index=True)

# Post-process: ensure all numeric columns from the original template are present if they were missing in df_new
# Concatenation handles this (adds NaNs).

# Drop sid helper
if 'sid' in df_m.columns:
    df_m.drop(columns=['sid'], inplace=True)

# Save
df_m.to_excel(measurement_path, index=False)
print(f"Updated {measurement_path} with all available task data. New rows added: {len(new_rows)}.")
