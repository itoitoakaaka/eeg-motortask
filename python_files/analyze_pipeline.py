import numpy as np
import os
import re

# Logic Analysis Configuration
ID = 'id0010001'
COND = 'A1'
BASE_DIR = '/Users/itoakane/Research/Analyzer'

# File Paths
P_EPOCH  = os.path.join(BASE_DIR, f'{ID}_{COND}_epoch_US')
P_BASE   = os.path.join(BASE_DIR, f'{ID}_{COND}_epoch_base_US')
P_REJECT = os.path.join(BASE_DIR, f'{ID}_{COND}_epoch_rejection_US')
P_AVG    = os.path.join(BASE_DIR, f'{ID}_spSEP_US')

def load_bv_data(base_path):
    vhdr = base_path + '.vhdr'
    eeg  = base_path + '.eeg'
    vmrk = base_path + '.vmrk'
    
    if not os.path.exists(eeg):
        print(f"Missing: {eeg}")
        return None, None
        
    # Read Segments info from VHDR
    segments = 0
    points = 0
    channels = 0
    try:
        with open(vhdr, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            m_ch = re.search(r'NumberOfChannels=(\d+)', content)
            m_pt = re.search(r'DataPoints=(\d+)', content) # Total points? Or seg points?
            m_seg = re.search(r'SegmentDataPoints=(\d+)', content)
            # Analyzer VHDR usually has DataPoints = Total, SegmentDataPoints = per segment
            if m_ch: channels = int(m_ch.group(1))
            if m_seg: 
                points = int(m_seg.group(1))
            
            # Need strict segment count for reshaping
            # Usually strict data size / (ch * points * 4)
            file_size = os.path.getsize(eeg)
            expected_floats = file_size // 4
            if channels * points > 0:
                segments = expected_floats // (channels * points)
    except Exception as e:
        print(f"Error reading header {vhdr}: {e}")
        return None, None

    # Load Data
    data = np.fromfile(eeg, dtype='<f4')
    # Reshape: (Segments, Points, Channels) if Multiplexed
    # Multiplexed in BrainVision means: S1_T1_C1, S1_T1_C2, S1_T2_C1...
    # BUT Analyzer Segmented data is often just concatenated segments.
    # checking file size
    if len(data) != segments * points * channels:
        print(f"Size mismatch {base_path}: Hdr Segs={segments}, Pts={points}, Ch={channels}. Data Len={len(data)}")
        # Try to infer segments
        if points * channels > 0:
            segments = len(data) // (points * channels)
            
    try:
        data = data.reshape((segments, points, channels)) # (Seg, Time, Ch) usually works for Analyzer export
        # Wait, BrainVision Multiplexed is (Points, Channels) stored.
        # So for segmented, it's (Segments * Points, Channels)
        # Then we reshape to (Segments, Points, Channels)
        # Actually, let's verify channel coherence.
    except:
        print("Reshape failed")
    
    # Read Markers
    markers = []
    if os.path.exists(vmrk):
        with open(vmrk, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('Mk'):
                    markers.append(line.strip())
    
    return data, markers

# 1. Load Data
print("--- Loading Data ---")
d_epoch, m_epoch = load_bv_data(P_EPOCH)
d_base, m_base   = load_bv_data(P_BASE)
d_reject, m_reject = load_bv_data(P_REJECT)
d_avg, m_avg     = load_bv_data(P_AVG)

if d_epoch is None or d_base is None:
    print("Critical files missing for analysis.")
    exit()

print(f"Epochs: {d_epoch.shape}")
print(f"Base: {d_base.shape}")

# 2. Analyze Baseline Correction
print("\n--- Baseline Correction Analysis ---")
# Calc difference
diff = d_epoch - d_base # Should be constant per segment
# Check if diff is constant across time within segment
# offset[seg, ch]
offsets = diff.mean(axis=1) # (Seg, Ch)
std_variation = diff.std(axis=1).mean() # Should be 0 (constant offset)
print(f"Variation within segment (should be 0): {std_variation:.6f}")

# Check if offset = Mean(-20ms to 0ms)
# Sampling 2500Hz -> 0.4ms/sample.
# Window: -20ms to 200ms. Length 220ms.
# 0ms is at index 50 (20ms / 0.4ms = 50).
# Baseline range: 0 to 50.
baseline_calc = d_epoch[:, 0:50, :].mean(axis=1)
error = np.abs(offsets - baseline_calc).mean()
print(f"Error vs Mean(0-50pts): {error:.6f}")
if error < 0.01:
    print(">> Baseline Logic Confirmed: Mean of points 0-50 (-20ms to 0ms) is subtracted.")
else:
    print(f">> Baseline Logic Mismatch. Offsets: {offsets[0]}, Calc: {baseline_calc[0]}")

# 3. Analyze Rejection
print("\n--- Rejection Analysis ---")
bad_intervals = []
if m_reject:
    for m in m_reject:
        if 'Bad Interval' in m:
            # Parse Mk<N>=Bad Interval,,<Pos>,<Size>,<Ch>
            # Pos in segmented file is usually relative to file start points?
            # Or global?
            # In segmented vmrk from Analyzer, Pos is usually linear index.
            # Segment size 550.
            parts = m.split(',')
            pos = int(parts[2])
            seg_idx = (pos - 1) // 550
            bad_intervals.append(seg_idx)

bad_set = set(bad_intervals)
print(f"Bad Intervals found: {len(bad_set)}")

# Check amplitude of bad segments
raw_bad_max = 0
raw_good_max = 0
for i in range(d_base.shape[0]):
    # Check max abs amp in this segment (using Baseline Corrected data)
    amp = np.max(np.abs(d_base[i]))
    if i in bad_set:
        raw_bad_max = max(raw_bad_max, amp)
        # Verify threshold
        if amp < 70.0:
            pass # print(f"Warning: Rejected segment {i} has max amp {amp:.2f} < 70uV")
    else:
        raw_good_max = max(raw_good_max, amp)
        if amp > 70.0:
            print(f"Warning: Accepted segment {i} has max amp {amp:.2f} > 70uV")

print(f"Max Amp in Rejected: {raw_bad_max:.2f} uV")
print(f"Max Amp in Accepted: {raw_good_max:.2f} uV")

# 4. Analyze Averaging
print("\n--- Averaging Analysis ---")
# Manually average 'Good' segments from d_base
# d_base is the one used for averaging (after BC, before Reject)
# or d_reject (might define the set, but data is same as base?)
# Usually Rejection step just adds markers, doesn't change data values.
# Let's verify d_base == d_reject
if np.allclose(d_base, d_reject):
    print("Base and Reject data content is identical (Rejection adds markers only).")
else:
    print("Base and Reject data differ.")

good_indices = [i for i in range(d_base.shape[0]) if i not in bad_set]
print(f"Valid Segments for Average: {len(good_indices)}")

if len(good_indices) > 0:
    my_avg = d_base[good_indices].mean(axis=0)
    
    # Compare with spSEP (d_avg)
    # d_avg should reflect 'A1' condition average.
    # Note: spSEP usually contains ALL conditions if they were averaged together?
    # Or is spSEP_US just the A1 average?
    # The file name is id0010001_spSEP_US.eeg. 
    # Usually Analyzer outputs per-node. If "spSEP" node combines A, B, C, then d_avg is 3 chunks?
    # Let's check d_avg shape.
    print(f"spSEP Shape: {d_avg.shape}")
    
    # If d_avg has multiple segments, correspond to conditions.
    # Assuming A1, B1, C1 in order?
    # Or just A1 if we are comparing A1_epoch pipeline?
    # Let's check overlap with A1 logic.
    
    # Compare my_avg (A1) with d_avg
    # If d_avg is (1, 550, 2), just compare.
    # If d_avg is (3, 550, 2), A1 might be index 0.
    
    target_avg = d_avg[0] if d_avg.shape[0] > 1 else d_avg[0]
    
    # Correlation/Error
    diff_avg = np.abs(my_avg - target_avg).mean()
    print(f"Averaging Replication Error: {diff_avg:.6f} uV")
    
    if diff_avg < 0.05:
        print(">> Averaging Logic Confirmed: Simple Mean of Non-Rejected Segments.")
    else:
        print(">> Averaging Logic Mismatch.")
