
import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

# Add python_files to path if needed, though we only use mne/matplotlib
sys.path.append('python_files')

RAW_DIR = 'SEP_raw_temp'
OUT_DIR = 'SEP_raw_temp/raw_previews'

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def process_file(fpath):
    basename = os.path.basename(fpath)
    out_name = os.path.splitext(basename)[0] + ".png"
    out_path = os.path.join(OUT_DIR, out_name)
    
    try:
        raw = mne.io.read_raw_brainvision(fpath, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"Error loading {basename}: {e}")
        return

    # Pick CP3
    ch_names = raw.ch_names
    cp3_idx = [i for i, c in enumerate(ch_names) if 'CP3' in c]
    if not cp3_idx:
        cp3_idx = 0
    else:
        cp3_idx = cp3_idx[0]
        
    data = raw.get_data()[cp3_idx] * 1e6 # uV
    times = raw.times
    
    # Find events
    events, _ = mne.events_from_annotations(raw, verbose='ERROR')
    
    if len(events) == 0:
        print(f"No events in {basename}")
        t_start = 0
    else:
        t_start = times[events[0, 0]]
        
    win = 5.0 # seconds
    t_end = min(times[-1], t_start + win)
    
    idx_start = raw.time_as_index(t_start)[0]
    idx_end = raw.time_as_index(t_end)[0]
    
    # Ensure sufficient length for visualization
    if idx_end <= idx_start:
        print(f"File too short? {basename}")
        return

    snippet_t = times[idx_start:idx_end]
    snippet_d = data[idx_start:idx_end]
    
    # Time relative to trigger
    plot_t_ms = (snippet_t - t_start) * 1000.0
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(plot_t_ms, snippet_d, linewidth=0.8, color='blue')
    
    # Add events markers
    mask_ev = (events[:, 0] >= idx_start) & (events[:, 0] < idx_end)
    local_ev = events[mask_ev]
    
    for ev in local_ev:
        t_ev_sec = times[ev[0]]
        t_ev_ms = (t_ev_sec - t_start) * 1000.0
        plt.axvline(t_ev_ms, color='red', linestyle='--', alpha=0.5)
        plt.text(t_ev_ms, np.max(snippet_d)*0.9, "Trig", color='red', fontsize=8)

    plt.title(f"{basename} (First 5s)")
    plt.xlabel("Time relative to 1st Trigger (ms)")
    plt.ylabel("Amplitude (uV) [Neg Up]")
    plt.gca().invert_yaxis() # Negative Up
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Generated {out_name}")

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.vhdr")))
    print(f"Found {len(files)} files.")
    for f in files:
        process_file(f)

if __name__ == "__main__":
    main()
