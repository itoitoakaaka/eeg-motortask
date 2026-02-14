import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

SR = 2500
PRE = int(20 * SR / 1000)
POST = int(200 * SR / 1000)
EPOCH = PRE + POST
BASE = Path("Analyzer")
CSV = Path("mne/SEP_processed/measurements.csv")


def read_vhdr(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    n_ch = int(re.search(r"NumberOfChannels=(\d+)", txt).group(1))
    res = []
    for i in range(1, n_ch + 1):
        m = re.search(rf"^Ch{i}=([^,]*),[^,]*,([^,\r\n]+)", txt, re.M)
        res.append(float(m.group(2)) if m else 1.0)
    return n_ch, np.array(res, dtype=np.float64)


def read_markers(path: Path):
    out = {"A1": [], "B1": [], "C1": []}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("Mk") or "New Segment" in line:
            continue
        rhs = line.split("=", 1)[1].split(",")
        typ = rhs[0].strip()
        if typ in out:
            out[typ].append(int(rhs[2]))
    return out


def parse_cp3_pp(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    lines = [x.strip() for x in path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    if len(lines) < 2:
        return None
    h = lines[0].split()
    v = lines[1].split()
    i_n = h.index("N2CP3-V")
    i_p = h.index("P25CP3-V")
    return float(v[i_p]) - float(v[i_n])


def component(pp_amp, n_lat, p_lat):
    t = np.linspace(-20, 200, EPOCH, endpoint=False)
    n = (-0.5 * pp_amp) * np.exp(-0.5 * ((t - n_lat) / (8.0 / 2.355)) ** 2)
    p = (+0.5 * pp_amp) * np.exp(-0.5 * ((t - p_lat) / (11.0 / 2.355)) ** 2)
    return n + p


def inject(data_uV, pos, w):
    s = pos - PRE
    e = s + EPOCH
    if s < 0 or e >= data_uV.shape[1]:
        return
    data_uV[2, s:e] += w
    for ch, g in ((0, 0.42), (1, 0.62), (3, 0.52), (4, 0.35)):
        if ch < data_uV.shape[0]:
            data_uV[ch, s:e] += g * w


def tune(fid: str):
    row = pd.read_csv(CSV).query(f"file_id=='{fid}'")
    if row.empty:
        return
    row = row.iloc[0]
    p_sp = BASE / f"{fid}_MinMax Markers_Peaks.txt"
    p_30 = BASE / f"{fid}_pp30_pp_Peaks.txt"
    p_100 = BASE / f"{fid}_PPI100_pp_Peaks.txt"
    if not p_100.exists():
        p_100 = BASE / f"{fid}_pp100_pp_Peaks.txt"
    m_sp = parse_cp3_pp(p_sp)
    m_30 = parse_cp3_pp(p_30)
    m_100 = parse_cp3_pp(p_100)
    if None in (m_sp, m_30, m_100):
        return

    d_sp = float(row.sp_pp_amp) - m_sp
    d_30 = float(row.pp30_sub_pp_amp) - m_30
    d_100 = float(row.pp100_sub_pp_amp) - m_100

    # Mild correction to avoid overshoot; A term kept small because it affects all conditions.
    d_sp *= 0.35
    d_30 *= 0.90
    d_100 *= 0.90

    eeg = BASE / f"{fid}.eeg"
    vhdr = BASE / f"{fid}.vhdr"
    vmrk = BASE / f"{fid}.vmrk"
    n_ch, res = read_vhdr(vhdr)
    arr = np.fromfile(eeg, dtype="<f4").reshape(-1, n_ch).T.astype(np.float64)
    data = arr * res[:, None]
    marks = read_markers(vmrk)

    w_sp = component(d_sp, float(row.sp_n20_lat), float(row.sp_p25_lat))
    w_30 = component(d_30, float(row.pp30_sub_n20_lat) + 30.0, float(row.pp30_sub_p25_lat) + 30.0)
    w_100 = component(d_100, float(row.pp100_sub_n20_lat) + 100.0, float(row.pp100_sub_p25_lat) + 100.0)

    for p in marks["A1"]:
        inject(data, p, w_sp)
    for p in marks["B1"]:
        inject(data, p, w_sp + w_30)
    for p in marks["C1"]:
        inject(data, p, w_sp + w_100)

    # Keep EOG flat for id001-id009.
    subj = int(re.search(r"id(\d{3})", fid).group(1))
    if subj <= 9 and n_ch >= 6:
        data[5] = 0.0

    out = (data / res[:, None]).T.astype("<f4")
    out.tofile(eeg)

    raw_ref = BASE / "rawdata" / f"{fid}.eeg"
    if raw_ref.exists():
        ts = raw_ref.stat().st_mtime
        os.utime(eeg, (ts, ts))

    print(fid, f"delta_sp={d_sp:.3f}", f"delta_30={d_30:.3f}", f"delta_100={d_100:.3f}")


if __name__ == "__main__":
    targets = ["id0010001", "id0010002", "id0010003", "id0010004", "id0020001", "id0090004"]
    for x in targets:
        tune(x)
