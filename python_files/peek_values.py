import numpy as np
import os

# Paths
RAW_PATH = '/Users/itoakane/Research/Analyzer/rawdata/id0010001.eeg'
FILT_PATH = '/Users/itoakane/Research/Analyzer/id0010001_Filters_3-1000_50_US.eeg'

def peek_data(path, n_ch, name):
    print(f"--- {name} ---")
    data = np.fromfile(path, dtype='<f4')
    print(f"Total Floats: {len(data)}")
    print(f"First 10 values: {data[:10]}")
    print(f"Min/Max: {np.min(data)} / {np.max(data)}")
    
    # Check if channel counts match assumption
    # Raw: 6ch?
    # Filt: 2ch? 
    # Just checking the stream to guess scale.

peek_data(RAW_PATH, 6, "Raw Data")
peek_data(FILT_PATH, 2, "Filtered US Data")
