import numpy as np
import os
import pandas as pd

# Paths
AVG_PATH = 'Analyzer/id0010001_spSEP_US.eeg'
SR = 2500

# Load Data (2 channels, 550 points)
# Channels: C3, CP3
# float32
data = np.fromfile(AVG_PATH, dtype='<f4')
n_samples = 550
n_ch = 2
# Correct Multiplexed Reshape: (Time, Ch)
# Sequence: t0c0, t0c1, t1c0, t1c1 ...
data = data.reshape((550, 2)).T # Transpose to (Ch, Time) for easy indexing 

# Assuming Ch2 (CP3) is the target? Or Ch1 (C3)?
# spSEP analysis usually targets CP3 or C3 depending on laterality. 
# Let's check both or sum? Usually measurements.csv implies a specific channel.
# id0010001 measurements (Land, Pre) usually C3 or CP3.
# Let's look for peaks in the 20-25ms range.

t = np.linspace(-20, 200, 550) # -20ms to 200ms

def find_peaks(ch_data, ch_name):
    print(f"--- Analyzing {ch_name} ---")
    # N20 window: 18-23ms
    # P25 window: 23-30ms
    win_n20 = (t >= 18) & (t <= 23)
    win_p25 = (t >= 23) & (t <= 30)
    
    if not np.any(win_n20) or not np.any(win_p25):
        print("Windows out of range")
        return

    n20_idx = np.where(win_n20)[0][0] + np.argmin(ch_data[win_n20])
    p25_idx = np.where(win_p25)[0][0] + np.argmax(ch_data[win_p25])
    
    n20_val = ch_data[n20_idx]
    p25_val = ch_data[p25_idx]
    n20_lat = t[n20_idx]
    p25_lat = t[p25_idx]
    
    pp_amp = p25_val - n20_val
    print(f"N20: {n20_val:.4f} uV at {n20_lat:.2f} ms")
    print(f"P25: {p25_val:.4f} uV at {p25_lat:.2f} ms")
    print(f"P-P Amp: {pp_amp:.4f} uV")

find_peaks(data[0], "Ch1 (C3)")
find_peaks(data[1], "Ch2 (CP3)")
