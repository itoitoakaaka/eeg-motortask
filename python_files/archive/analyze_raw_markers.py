import re
import numpy as np

VMRK_PATH = 'Analyzer/rawdata/id0010001.vmrk'

def analyze_markers():
    markers = []
    try:
        with open(VMRK_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('Mk'):
                    # Mk<N>=<Type>,<Desc>,<Pos>,<Size>,<Ch>
                    # Mk2=B1,B  1,50000,1,0
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        desc = parts[1]
                        pos = int(parts[2])
                        markers.append({'desc': desc, 'pos': pos})
    except Exception as e:
        print(f"Error reading {VMRK_PATH}: {e}")
        return

    # Count Conditions
    a1 = [m['pos'] for m in markers if 'A  1' in m['desc'] or 'A1' in m['desc']]
    b1 = [m['pos'] for m in markers if 'B  1' in m['desc'] or 'B1' in m['desc']]
    c1 = [m['pos'] for m in markers if 'C  1' in m['desc'] or 'C1' in m['desc']]

    print(f"A1 Count: {len(a1)}")
    print(f"B1 Count: {len(b1)}")
    print(f"C1 Count: {len(c1)}")
    print(f"Total Stim Count: {len(a1) + len(b1) + len(c1)}")

    # Calculate ITI
    all_pos = sorted(a1 + b1 + c1)
    if len(all_pos) > 1:
        diffs = np.diff(all_pos)
        mean_iti = np.mean(diffs)
        min_iti = np.min(diffs)
        max_iti = np.max(diffs)
        print(f"Mean ITI: {mean_iti:.2f} samples ({mean_iti/2500:.4f} sec)")
        print(f"Min ITI: {min_iti} samples")
        print(f"Max ITI: {max_iti} samples")
        
        # Calculate Duration required
        duration_samples = all_pos[-1] - all_pos[0] + 50000 # Add start/end buffer
        print(f"Estimated Duration: {duration_samples} samples ({duration_samples/2500:.2f} sec)")

if __name__ == "__main__":
    analyze_markers()
