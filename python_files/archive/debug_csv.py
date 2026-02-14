import pandas as pd
CSV_PATH = 'mne/SEP_processed/measurements.csv'
df = pd.read_csv(CSV_PATH)
row = df[df['file_id'] == 'id0010001']
print(f"sp_pp_amp: {float(row['sp_pp_amp'].iloc[0])}")
print(f"pp30_pp_amp: {float(row['pp30_pp_amp'].iloc[0])}")
print(f"pp100_pp_amp: {float(row['pp100_pp_amp'].iloc[0])}")
