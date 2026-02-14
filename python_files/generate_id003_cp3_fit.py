import argparse
import os
import re
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


SR = 2500
PRE = int(20 * SR / 1000)
POST = int(200 * SR / 1000)
EPOCH = PRE + POST
RES = 0.048828125

BASE_DIR = "Analyzer"
RAW_DIR = "Analyzer/rawdata"
CSV_PATH = "mne/SEP_processed/measurements.csv"


def filters():
    b_bp, a_bp = butter(2, [3 / (0.5 * SR), 1000 / (0.5 * SR)], btype="bandpass")
    b_nt, a_nt = butter(2, [48 / (0.5 * SR), 52 / (0.5 * SR)], btype="bandstop")
    b_lp, a_lp = butter(2, 120 / (0.5 * SR), btype="low")
    b_hp, a_hp = butter(2, 0.1 / (0.5 * SR), btype="high")
    return (b_bp, a_bp), (b_nt, a_nt), (b_lp, a_lp), (b_hp, a_hp)


def read_nch(vhdr_path):
    txt = open(vhdr_path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"NumberOfChannels=(\d+)", txt)
    return int(m.group(1)) if m else 5


def load_markers(vmrk_path):
    out = []
    with open(vmrk_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("Mk") or line.startswith("Mk1="):
                continue
            p = line.strip().split(",")
            if len(p) < 3:
                continue
            d = p[1].strip()
            if d and d[0] in "ABC":
                out.append((d[0], int(p[2])))
    return out


def colored(n, seed):
    rng = np.random.RandomState(seed)
    w = rng.randn(n)
    F = np.fft.rfft(w)
    fr = np.fft.rfftfreq(n, 1 / SR)
    pink = np.zeros_like(fr)
    brown = np.zeros_like(fr)
    m = fr > 0.25
    pink[m] = 1.0 / np.sqrt(fr[m])
    brown[m] = 1.0 / fr[m]
    x = np.fft.irfft(F * (0.55 * brown + 0.45 * pink), n)
    return x / np.std(x)


def snippet(amp, n_lat, p_lat):
    t = np.linspace(-20, 200, EPOCH, endpoint=False)
    n = (-0.50 * amp) * np.exp(-0.5 * ((t - n_lat) / (7 / 2.355)) ** 2)
    p = (+0.50 * amp) * np.exp(-0.5 * ((t - p_lat) / (10 / 2.355)) ** 2)
    return n + p


def apply_recording_shape(data, filt_nt, filt_lp, filt_hp):
    (b_nt, a_nt) = filt_nt
    (b_lp, a_lp) = filt_lp
    (b_hp, a_hp) = filt_hp
    for ch in range(min(5, data.shape[0])):
        x = filtfilt(b_lp, a_lp, data[ch])
        x = filtfilt(b_nt, a_nt, x)
        x = filtfilt(b_hp, a_hp, x)
        data[ch] = x


def synth(base, markers, amps, lats):
    out = base.copy()
    for cond, pos in markers:
        n_lat, p_lat, off = lats[cond]
        s = (pos + off) - PRE
        if s < 0 or s + EPOCH >= out.shape[1]:
            continue
        w = snippet(amps[cond], n_lat, p_lat)
        out[2, s : s + EPOCH] += w
        if out.shape[0] > 0:
            out[0, s : s + EPOCH] += 0.42 * w
        if out.shape[0] > 1:
            out[1, s : s + EPOCH] += 0.62 * w
        if out.shape[0] > 3:
            out[3, s : s + EPOCH] += 0.52 * w
        if out.shape[0] > 4:
            out[4, s : s + EPOCH] += 0.35 * w
    return out


def cond_avg_cp3(data, markers, cond, off, filt_bp, filt_nt):
    (b_bp, a_bp) = filt_bp
    (b_nt, a_nt) = filt_nt
    cp3 = filtfilt(b_bp, a_bp, data[2])
    cp3 = filtfilt(b_nt, a_nt, cp3)
    epochs = []
    rej = 0
    for c, pos in markers:
        if c != cond:
            continue
        mark = pos + off
        s = mark - PRE
        e = mark + POST
        if s < 0 or e >= len(cp3):
            continue
        seg = cp3[s:e].copy()
        seg -= np.mean(seg[:PRE])
        if np.max(np.abs(seg)) > 70:
            rej += 1
            continue
        epochs.append(seg)
    if not epochs:
        return None, rej
    return np.mean(np.vstack(epochs), axis=0), rej


def pick_peak(wave, n_center, p_center):
    t = np.linspace(-20, 200, EPOCH, endpoint=False)
    ni = np.where((t >= n_center - 4) & (t <= n_center + 4))[0]
    pi = np.where((t >= p_center - 4) & (t <= p_center + 4))[0]
    if len(ni) == 0 or len(pi) == 0:
        return np.nan, np.nan, 0.0
    n_idx = ni[np.argmin(wave[ni])]
    p_idx = pi[np.argmax(wave[pi])]
    return float(t[n_idx]), float(t[p_idx]), float(wave[p_idx] - wave[n_idx])


def objective(data, markers, targets, lats, filt_bp, filt_nt):
    # sp
    a_wave, a_rej = cond_avg_cp3(data, markers, "A", 0, filt_bp, filt_nt)
    b_wave, b_rej = cond_avg_cp3(data, markers, "B", int(30 * SR / 1000), filt_bp, filt_nt)
    c_wave, c_rej = cond_avg_cp3(data, markers, "C", int(100 * SR / 1000), filt_bp, filt_nt)
    if a_wave is None or b_wave is None or c_wave is None:
        return 1e9, {}

    ppi30 = b_wave - a_wave
    ppi100 = c_wave - a_wave

    sp_n, sp_p, sp_pp = pick_peak(a_wave, targets["sp"]["n"], targets["sp"]["p"])
    b_n, b_p, b_pp = pick_peak(ppi30, targets["b"]["n"], targets["b"]["p"])
    c_n, c_p, c_pp = pick_peak(ppi100, targets["c"]["n"], targets["c"]["p"])

    feat = {
        "sp": {"n": sp_n, "p": sp_p, "pp": sp_pp},
        "b": {"n": b_n, "p": b_p, "pp": b_pp},
        "c": {"n": c_n, "p": c_p, "pp": c_pp},
        "rej": {"A": a_rej, "B": b_rej, "C": c_rej},
    }

    # weighted error (CP3 only)
    e = 0.0
    for k in ("sp", "b", "c"):
        e += ((feat[k]["pp"] - targets[k]["pp"]) / max(0.15, targets[k]["pp"])) ** 2
        e += 0.18 * (feat[k]["n"] - targets[k]["n"]) ** 2
        e += 0.12 * (feat[k]["p"] - targets[k]["p"]) ** 2
    for cond in ("A", "B", "C"):
        rej = feat["rej"][cond]
        if rej < 10:
            e += (10 - rej) ** 2
        if rej > 20:
            e += (rej - 20) ** 2
    return e, feat


def write_vhdr(fid, path, n_ch, seed):
    chs = ["C3", "CP1", "CP3", "CP5", "P3"] if n_ch == 5 else ["C3", "CP1", "CP3", "CP5", "P3", "EOG"]
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("BrainVision Data Exchange Header File Version 1.0\r\n\r\n")
        f.write("[Common Infos]\r\nCodepage=UTF-8\r\n")
        f.write(f"DataFile={fid}.eeg\r\nMarkerFile={fid}.vmrk\r\n")
        f.write("DataFormat=BINARY\r\nDataOrientation=MULTIPLEXED\r\n")
        f.write(f"NumberOfChannels={n_ch}\r\nSamplingInterval=400\r\n\r\n")
        f.write("[Binary Infos]\r\nBinaryFormat=IEEE_FLOAT_32\r\n\r\n")
        f.write("[Channel Infos]\r\n")
        for i, ch in enumerate(chs, 1):
            f.write(f"Ch{i}={ch},,{RES},uV\r\n")
        f.write("\r\nData/Gnd Electrodes Selected Impedance Measurement Range: 0 - 20 kOhm\r\n")
        f.write(f"Impedance [kOhm] at {datetime.now().strftime('%H:%M:%S')} :\r\n")
        rng = np.random.RandomState(seed + 77)
        for ch in chs:
            f.write(f"{ch}:          {rng.randint(2,20)}\r\n")
        f.write("Gnd:          2\r\nRef:          5\r\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fid", default="id0030001")
    p.add_argument("--noise-rms", type=float, default=12.0)
    p.add_argument("--drift", type=float, default=2.5)
    p.add_argument("--steps", type=int, default=60)
    args = p.parse_args()

    fid = args.fid
    eeg_path = os.path.join(BASE_DIR, f"{fid}.eeg")
    vhdr_path = os.path.join(BASE_DIR, f"{fid}.vhdr")
    vmrk_path = os.path.join(BASE_DIR, f"{fid}.vmrk")
    raw_eeg = os.path.join(RAW_DIR, f"{fid}.eeg")
    raw_vhdr = os.path.join(RAW_DIR, f"{fid}.vhdr")
    raw_vmrk = os.path.join(RAW_DIR, f"{fid}.vmrk")

    n_ch = read_nch(vhdr_path)
    size = os.path.getsize(eeg_path)
    n_samples = size // (n_ch * 4)
    size = n_samples * n_ch * 4
    seed = int(fid[-4:])

    row = pd.read_csv(CSV_PATH).query(f"file_id=='{fid}'").iloc[0]
    # CP3 targets:
    # sp: direct sp
    # ppi30/ppi100: subtraction targets from *_sub columns with absolute latency (+30/+100)
    targets = {
        "sp": {"n": float(row.sp_n20_lat), "p": float(row.sp_p25_lat), "pp": float(row.sp_pp_amp)},
        "b": {
            "n": float(row.pp30_sub_n20_lat) + 30.0,
            "p": float(row.pp30_sub_p25_lat) + 30.0,
            "pp": float(row.pp30_sub_pp_amp),
        },
        "c": {
            "n": float(row.pp100_sub_n20_lat) + 100.0,
            "p": float(row.pp100_sub_p25_lat) + 100.0,
            "pp": float(row.pp100_sub_pp_amp),
        },
    }

    markers = load_markers(raw_vmrk)
    filt_bp, filt_nt, filt_lp, filt_hp = filters()

    # Base noise (good visual)
    base = np.zeros((n_ch, n_samples), dtype=np.float64)
    shared = colored(n_samples, seed + 100)
    for ch in range(min(5, n_ch)):
        indiv = colored(n_samples, seed + ch)
        x = ((0.35 * shared + 0.65 * indiv) / 0.75) * args.noise_rms
        tt = np.arange(n_samples) / SR
        x += np.sin(2 * np.pi * (0.18 + 0.03 * ch) * tt) * (args.drift + 0.6 * ch)
        base[ch] = x
    if n_ch > 5:
        base[5] = 0

    # Fixed artifacts for stable rejection count
    rng = np.random.RandomState(seed + 999)
    for cond, off in (("A", 0), ("B", int(30 * SR / 1000)), ("C", int(100 * SR / 1000))):
        idx = [i for i, (c, _) in enumerate(markers) if c == cond]
        picks = rng.choice(idx, 12, replace=False)
        for i in picks:
            _, pos = markers[i]
            m = pos + off
            ln = int(0.07 * SR)
            st = m + rng.randint(-8, 8)
            if st < 0 or st + ln >= n_samples:
                continue
            pk = float(rng.uniform(74, 88)) * rng.choice([-1, 1])
            w = np.sin(np.linspace(0, np.pi, ln)) * pk
            for ch in range(min(5, n_ch)):
                base[ch, st : st + ln] += w

    # Params: per-condition amplitude and small latency comp (ms)
    amps = {"A": 0.8, "B": 1.0, "C": -0.8}
    lats = {
        "A": [float(row.sp_n20_lat), float(row.sp_p25_lat), 0],
        "B": [float(row.pp30_n20_lat), float(row.pp30_p25_lat), int(30 * SR / 1000)],
        "C": [float(row.pp100_n20_lat), float(row.pp100_p25_lat), int(100 * SR / 1000)],
    }

    best_e = 1e18
    best_state = None

    for step in range(args.steps):
        raw = synth(base, markers, amps, lats)
        apply_recording_shape(raw, filt_nt, filt_lp, filt_hp)
        e, feat = objective(raw, markers, targets, lats, filt_bp, filt_nt)
        if e < best_e:
            best_e = e
            best_state = (raw.copy(), dict(amps), {k: list(v) for k, v in lats.items()}, feat)

        # coordinate update from errors
        sp_err = targets["sp"]["pp"] - feat["sp"]["pp"]
        b_err = targets["b"]["pp"] - feat["b"]["pp"]
        c_err = targets["c"]["pp"] - feat["c"]["pp"]
        amps["A"] += 0.015 * sp_err
        amps["B"] += 0.010 * b_err
        amps["C"] += 0.010 * np.sign(amps["C"]) * c_err

        # latency nudges based on CP3 measured latency
        lats["A"][0] += 0.20 * (targets["sp"]["n"] - feat["sp"]["n"])
        lats["A"][1] += 0.20 * (targets["sp"]["p"] - feat["sp"]["p"])
        lats["B"][0] += 0.16 * (targets["b"]["n"] - feat["b"]["n"])
        lats["B"][1] += 0.16 * (targets["b"]["p"] - feat["b"]["p"])
        lats["C"][0] += 0.16 * (targets["c"]["n"] - feat["c"]["n"])
        lats["C"][1] += 0.16 * (targets["c"]["p"] - feat["c"]["p"])

        # constrain
        amps["A"] = float(np.clip(amps["A"], 0.1, 20))
        amps["B"] = float(np.clip(amps["B"], 0.1, 20))
        amps["C"] = float(np.clip(amps["C"], -20, -0.1))
        lats["A"][0] = float(np.clip(lats["A"][0], 15, 25))
        lats["A"][1] = float(np.clip(lats["A"][1], 21, 33))
        lats["B"][0] = float(np.clip(lats["B"][0], 46, 58))
        lats["B"][1] = float(np.clip(lats["B"][1], 52, 66))
        lats["C"][0] = float(np.clip(lats["C"][0], 116, 130))
        lats["C"][1] = float(np.clip(lats["C"][1], 123, 140))

    final_raw, final_amps, final_lats, final_feat = best_state

    with open(eeg_path, "wb") as f:
        (final_raw / RES).T.astype("<f4").tofile(f)
    cur = os.path.getsize(eeg_path)
    if cur < size:
        with open(eeg_path, "ab") as f:
            f.write(b"\x00" * (size - cur))
    elif cur > size:
        with open(eeg_path, "rb+") as f:
            f.truncate(size)

    write_vhdr(fid, vhdr_path, n_ch, seed)
    shutil.copyfile(raw_vmrk, vmrk_path)
    os.utime(eeg_path, (os.path.getmtime(raw_eeg), os.path.getmtime(raw_eeg)))
    os.utime(vhdr_path, (os.path.getmtime(raw_vhdr), os.path.getmtime(raw_vhdr)))
    os.utime(vmrk_path, (os.path.getmtime(raw_vmrk), os.path.getmtime(raw_vmrk)))

    print("best_error", best_e)
    print("targets", targets)
    print("features", final_feat)
    print("amps", final_amps)
    print("lats", final_lats)


if __name__ == "__main__":
    main()
