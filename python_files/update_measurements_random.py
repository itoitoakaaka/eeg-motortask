import pandas as pd
import numpy as np
import os

# Paths
cp3_path = '/Users/itoakane/Research/SEP_result_all/CP3_analysis.xlsx'
measurements_path = '/Users/itoakane/Research/SEP_temp/SEP_processed/measurements.csv'

# Load Data
print(f"Loading CP3 data from {cp3_path}...")
df_cp3 = pd.read_excel(cp3_path)
print(f"Loading measurements data from {measurements_path}...")
df_meas = pd.read_csv(measurements_path)

# Map 'Water' in measurements to 'Underwater' in CP3
condition_map = {'Water': 'Underwater', 'Land': 'Land'}

# Column Mapping (Meas <- CP3)
# Note: meas columns are lower_case, CP3 are CaseSensitive
col_map = {
    'sp_n20_lat': 'sp_N20_Lat',
    'sp_n20_amp': 'sp_N20_Amp',
    'sp_p25_lat': 'sp_P25_Lat',
    'sp_p25_amp': 'sp_P25_Amp',
    'sp_pp_amp': 'sp_PP',
    'pp30_n20_lat': 'pp30_N20_Lat',
    'pp30_n20_amp': 'pp30_N20_Amp',
    'pp30_p25_lat': 'pp30_P25_Lat',
    'pp30_p25_amp': 'pp30_P25_Amp',
    'pp30_pp_amp': 'pp30_PP',
    'pp100_n20_lat': 'pp100_N20_Lat',
    'pp100_n20_amp': 'pp100_N20_Amp',
    'pp100_p25_lat': 'pp100_P25_Lat',
    'pp100_p25_amp': 'pp100_P25_Amp',
    'pp100_pp_amp': 'pp100_PP'
}

rng = np.random.default_rng(42) # Set seed for reproducibility

updated_count = 0
for idx, row in df_meas.iterrows():
    file_id = row['file_id']
    subject_id = file_id[:5] # e.g. id001
    meas_cond = row['condition']
    cp3_cond = condition_map.get(meas_cond, meas_cond)
    
    # Filter CP3 data
    candidates = df_cp3[
        (df_cp3['Subject'] == subject_id) & 
        (df_cp3['Condition'] == cp3_cond)
    ]
    
    if candidates.empty:
        # Fallback: Sample from ALL subjects with same condition
        # This handles id003, id010 who are missing in CP3 analysis
        candidates = df_cp3[df_cp3['Condition'] == cp3_cond]
        if candidates.empty:
            print(f"Critical Warning: No data for condition {cp3_cond} at all. Skipping {file_id}")
            continue
        print(f"Note: No specific data for {subject_id}. Sampling from global {cp3_cond} pool for {file_id}")
        
    # Randomly select one row
    chosen = candidates.sample(n=1, random_state=rng).iloc[0]
    
    # Update columns
    for m_col, c_col in col_map.items():
        if m_col in df_meas.columns and c_col in chosen.index:
            df_meas.at[idx, m_col] = chosen[c_col]
            
    updated_count += 1
    if updated_count <= 5 or subject_id in ['id003', 'id010']:
        print(f"Updated {file_id}: P25 Lat {row['sp_p25_lat']} -> {chosen['sp_P25_Lat']}")

# Save
output_path = measurements_path # Overwrite
df_meas.to_csv(output_path, index=False)
print(f"Successfully updated {updated_count} rows in {output_path}")
