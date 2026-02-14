import numpy as np
import os

fid = 'id0030002'
path = f'/Users/itoakane/Research/Analyzer/{fid}.eeg'
n_ch = 6
res = 0.048828125

if not os.path.exists(path):
    print("File not found")
    exit()

data = np.fromfile(path, dtype=np.float32)
n_samples = data.shape[0] // n_ch
data = data.reshape(n_samples, n_ch).T
data = data * res

print(f"--- File: {fid} ---")
print(f"Max Val (uV): {np.max(data):.2f}")
print(f"Min Val (uV): {np.min(data):.2f}")
print(f"RMS (uV): {np.std(data):.2f}")

# Average CP3 (Ch 2)
vmrk = f'/Users/itoakane/Research/Analyzer/{fid}.vmrk'
m_pos = []
with open(vmrk, 'r') as f:
    for line in f:
        if 'A1,A  1' in line:
            m_pos.append(int(line.split(',')[2]))

epochs = []
for p in m_pos:
    s, e = p, p + 125
    if e < data.shape[1]:
        epochs.append(data[2, s:e])

if len(epochs) > 0:
    avg = np.mean(epochs, axis=0)
    pp = np.max(avg) - np.min(avg)
    print(f"Averaged P-P (A): {pp:.4f} uV")
