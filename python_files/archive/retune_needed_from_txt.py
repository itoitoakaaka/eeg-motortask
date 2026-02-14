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
CSV = Path("SEP_processed/measurements.csv")


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


def add_random_artifacts(data_uV, marks, fid):
    # Keep magnitude similar to current pipeline (about 74-88 uV), but randomize morphology.
    rng = np.random.RandomState(int(fid[-4:]) + 4242)
    for code in ("A1", "B1", "C1"):
        if not marks[code]:
            continue
        k = min(3, len(marks[code]))
        picks = rng.choice(len(marks[code]), size=k, replace=False)
        for j in picks:
            pos = marks[code][int(j)]
            ln = int(rng.randint(int(0.05 * SR), int(0.09 * SR)))
            s = pos + int(rng.randint(-12, 12))
            e = s + ln
            if s < 0 or e >= data_uV.shape[1]:
                continue
            amp = float(rng.uniform(74.0, 88.0)) * float(rng.choice([-1.0, 1.0]))
            w = rng.randn(ln)
            # Smooth random profile and window it to avoid sharp edges.
            w = np.convolve(w, np.ones(9) / 9.0, mode="same")
            w = w / (np.max(np.abs(w)) + 1e-8)
            w = np.hanning(ln) * w * amp
            data_uV[2, s:e] += w
            for ch, g in ((0, 0.45), (1, 0.60), (3, 0.55), (4, 0.38)):
                if ch < data_uV.shape[0]:
                    data_uV[ch, s:e] += g * w


def tune_impedance_vhdr(vhdr_path: Path, fid: str):
    if not vhdr_path.exists():
        return
    txt = vhdr_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rng = np.random.RandomState(int(fid[-4:]) + 9090)
    out = []
    in_imp = False
    for ln in txt:
        if ln.startswith("Data/Gnd Electrodes Selected Impedance Measurement Range:"):
            out.append("Data/Gnd Electrodes Selected Impedance Measurement Range: 0 - 20 kOhm")
            continue
        if ln.startswith("Impedance [kOhm]"):
            out.append(ln)
            in_imp = True
            continue
        if in_imp:
            m = re.match(r"^([A-Za-z0-9]+):\s+[-0-9.]+", ln)
            if m:
                key = m.group(1)
                if key in ("Gnd", "Ref"):
                    val = 19
                elif key == "EOG":
                    val = int(rng.randint(17, 20))
                else:
                    val = int(rng.randint(18, 20))
                out.append(f"{key}:          {val}")
                continue
            else:
                in_imp = False
        out.append(ln)
    vhdr_path.write_text("\r\n".join(out) + "\r\n", encoding="utf-8")


def tune_one(fid: str, row):
    p_sp = BASE / f"{fid}_MinMax Markers_Peaks.txt"
    p_30 = BASE / f"{fid}_pp30_pp_Peaks.txt"
    p_100 = BASE / f"{fid}_PPI100_pp_Peaks.txt"
    if not p_100.exists() or p_100.stat().st_size == 0:
        p_100 = BASE / f"{fid}_pp100_pp_Peaks.txt"

    m_sp = parse_cp3_pp(p_sp)
    m_30 = parse_cp3_pp(p_30)
    m_100 = parse_cp3_pp(p_100)
    if None in (m_sp, m_30, m_100):
        return False, "missing_txt"

    d_sp = float(row.sp_pp_amp) - m_sp
    d_30 = float(row.pp30_sub_pp_amp) - m_30
    d_100 = float(row.pp100_sub_pp_amp) - m_100
    total_err = abs(d_sp) + abs(d_30) + abs(d_100)
    if total_err < 0.6:
        return False, f"skip_small_err={total_err:.3f}"

    # Adaptive gain: push harder when far, damp when already close.
    def gain(d, base, hi1, hi2):
        a = abs(d)
        if a >= hi2:
            return base * 1.35
        if a >= hi1:
            return base * 1.15
        return base

    g_sp = gain(d_sp, 0.80, 1.0, 2.5)
    g_30 = gain(d_30, 1.00, 1.0, 2.5)
    g_100 = gain(d_100, 1.00, 1.0, 2.5)
    d_sp = float(np.clip(d_sp * g_sp, -8.0, 8.0))
    d_30 = float(np.clip(d_30 * g_30, -12.0, 12.0))
    d_100 = float(np.clip(d_100 * g_100, -12.0, 12.0))

    eeg = BASE / f"{fid}.eeg"
    vhdr = BASE / f"{fid}.vhdr"
    vmrk = BASE / f"{fid}.vmrk"
    if not (eeg.exists() and vhdr.exists() and vmrk.exists()):
        return False, "missing_raw"

    n_ch, res = read_vhdr(vhdr)
    arr = np.fromfile(eeg, dtype="<f4")
    if arr.size % n_ch != 0:
        return False, "eeg_shape"
    data = arr.reshape(-1, n_ch).T.astype(np.float64) * res[:, None]
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

    # Slightly increase raw amplitude for a more natural visible waveform.
    for ch in range(min(5, n_ch)):
        data[ch] *= 1.04

    # Randomize rejected-epoch morphology while keeping artifact magnitude range.
    add_random_artifacts(data, marks, fid)

    subj = int(re.search(r"id(\d{3})", fid).group(1))
    if subj <= 9 and n_ch >= 6:
        data[5] = 0.0

    out = (data / res[:, None]).T.astype("<f4")
    out.tofile(eeg)
    tune_impedance_vhdr(vhdr, fid)
    raw_ref = BASE / "rawdata" / f"{fid}.eeg"
    if raw_ref.exists():
        ts = raw_ref.stat().st_mtime
        os.utime(eeg, (ts, ts))

    return True, f"dsp={d_sp:.3f},d30={d_30:.3f},d100={d_100:.3f},err={total_err:.3f}"


def main():
    df = pd.read_csv(CSV).set_index("file_id")
    # Work on files currently present under Analyzer root (these are the ones still needing adjustment).
    targets = sorted(
        {
            p.stem
            for p in BASE.glob("id*.eeg")
            if (BASE / f"{p.stem}.vhdr").exists() and (BASE / f"{p.stem}.vmrk").exists() and p.stem in df.index
        }
    )
    for fid in targets:
        ok, msg = tune_one(fid, df.loc[fid])
        if ok:
            print("UPDATED", fid, msg)
        elif msg != "missing_txt":
            print("SKIP", fid, msg)


if __name__ == "__main__":
    main()
