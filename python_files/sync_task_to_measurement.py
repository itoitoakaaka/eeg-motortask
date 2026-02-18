import pandas as pd
import numpy as np

# Paths
measurement_path = '/Users/itoakane/Research/SEP_processed/measurement2.xlsx'
task_path = '/Users/itoakane/Research/task/task.xlsx'

# Load
df_m = pd.read_excel(measurement_path)
df_t = pd.read_excel(task_path)

# Prepare mapping key in task.xlsx
# BaseID (test002) -> id002
df_t['sid'] = df_t['BaseID'].str.replace('test', 'id')

# Create a lookup dictionary
# Key: (sid, Condition), Value: (AUC, MAE_50, MAE_All)
lookup = {}
for _, row in df_t.iterrows():
    lookup[(row['sid'], row['Condition'])] = (row['AUC'], row['50%MAE'], row['Overall_MAE'])

# Update measurement2.xlsx
df_m['sid'] = df_m['file_id'].astype(str).str[:5]

updated_count = 0
for i, row in df_m.iterrows():
    key = (row['sid'], row['condition'])
    if key in lookup:
        auc, mae50, mae_all = lookup[key]
        df_m.at[i, 'AUC'] = auc
        df_m.at[i, 'MAE_50'] = mae50
        df_m.at[i, 'MAE_All'] = mae_all
        updated_count += 1

# Drop sid helper col
df_m.drop(columns=['sid'], inplace=True)

# Save back
df_m.to_excel(measurement_path, index=False)
print(f"Updated {updated_count} rows in {measurement_path} with new AUC/MAE data.")
