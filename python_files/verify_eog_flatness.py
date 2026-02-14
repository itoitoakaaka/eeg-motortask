import numpy as np

# Check id0030001 and id0030002
files = ['/Users/itoakane/Research/Analyzer/id0030001.eeg', '/Users/itoakane/Research/Analyzer/id0030002.eeg']

for path in files:
    fid = path.split('/')[-1].replace('.eeg', '')
    print(f"--- Checking {fid} ---")
    data = np.fromfile(path, dtype='<f4')
    # 6 Channels
    n_ch = 6
    n_samples = len(data) // n_ch
    data = data.reshape((n_samples, n_ch)).T
    
    # EOG is Ch6 (Index 5)
    eog = data[5]
    print(f"EOG Min: {np.min(eog)}")
    print(f"EOG Max: {np.max(eog)}")
    print(f"EOG Mean: {np.mean(eog)}")
    print(f"EOG Std: {np.std(eog)}")
    if np.all(eog == 0):
        print(">> VERIFIED: EOG is perfectly FLAT (0.0).")
    else:
        print(">> WARNING: EOG is NOT flat.")
