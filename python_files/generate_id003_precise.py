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


def butter_filters():
    b_bp, a_bp = butter(2, [3 / (0.5 * SR), 1000 / (0.5 * SR)], btype="bandpass")
    b_nt, a_nt = butter(2, [48 / (0.5 * SR), 52 / (0.5 * SR)], btype="bandstop")
    b_lp, a_lp = butter(2, 120 / (0.5 * SR), btype="low")
    b_hp, a_hp = butter(2, 0.1 / (0.5 * SR), btype="high")
    return (b_bp, a_bp), (b_nt, a_nt), (b_lp, a_lp), (b_hp, a_hp)


def read_num_channels(vhdr_path):
    txt = open(vhdr_path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"NumberOfChannels=(\d+)", txt)
    return int(m.group(1)) if m else 5


def load_markers(vmrk_path):
    markers = []
    with open(vmrk_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("Mk") or line.startswith("Mk1="):
                continue
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            desc = parts[1].strip()
            if not desc:
                continue
            cond = desc[0]
            if cond in "ABC":
                markers.append((cond, int(parts[2])))
    return markers


def make_colored_noise(n, seed):
    rng = np.random.RandomState(seed)
    white = rng.randn(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    pink = np.zeros_like(freqs)
    brown = np.zeros_like(freqs)
    mask = freqs > 0.25
    pink[mask] = 1.0 / np.sqrt(freqs[mask])
    brown[mask] = 1.0 / freqs[mask]
    shaped = np.fft.irfft(spec * (0.55 * brown + 0.45 * pink), n)
    return shaped / np.std(shaped)


def snippet_wave(amp, n_lat, p_lat):
    t = np.linspace(-20, 200, EPOCH, endpoint=False)
    n_comp = (-0.50 * amp) * np.exp(-0.5 * ((t - n_lat) / (7 / 2.355)) ** 2)
    p_comp = (+0.50 * amp) * np.exp(-0.5 * ((t - p_lat) / (10 / 2.355)) ** 2)
    return n_comp + p_comp


def apply_recording_like_smoothing(data, filt_nt, filt_lp, filt_hp):
    (b_nt, a_nt) = filt_nt
    (b_lp, a_lp) = filt_lp
    (b_hp, a_hp) = filt_hp
    for ch in range(min(5, data.shape[0])):
        x = filtfilt(b_lp, a_lp, data[ch])
        x = filtfilt(b_nt, a_nt, x)
        x = filtfilt(b_hp, a_hp, x)
        data[ch] = x


def measure_cp3(data, markers, lat_specs, filt_bp, filt_nt):
    (b_bp, a_bp) = filt_bp
    (b_nt, a_nt) = filt_nt
    cp3 = filtfilt(b_bp, a_bp, data[2])
    cp3 = filtfilt(b_nt, a_nt, cp3)
    t = np.linspace(-20, 200, EPOCH, endpoint=False)
    measured = {}
    reject = {}

    for cond in "ABC":
        n_lat, p_lat, offset = lat_specs[cond]
        epochs = []
        rej = 0
        for c, pos in markers:
            if c != cond:
                continue
            mark = pos + offset
            s = mark - PRE
            e = mark + POST
            if s < 0 or e >= len(cp3):
                continue
            seg = cp3[s:e].copy()
            seg -= np.mean(seg[:PRE])
            if np.max(np.abs(seg)) > 70.0:
                rej += 1
                continue
            epochs.append(seg)

        reject[cond] = rej
        if not epochs:
            measured[cond] = {"pp": 0.0, "n_lat": np.nan, "p_lat": np.nan}
            continue

        avg = np.mean(np.vstack(epochs), axis=0)
        n_idx = np.where((t >= (n_lat - 3.0)) & (t <= (n_lat + 3.0)))[0]
        p_idx = np.where((t >= (p_lat - 3.0)) & (t <= (p_lat + 3.0)))[0]
        if len(n_idx) == 0 or len(p_idx) == 0:
            measured[cond] = {"pp": 0.0, "n_lat": np.nan, "p_lat": np.nan}
            continue
        n_pos = n_idx[np.argmin(avg[n_idx])]
        p_pos = p_idx[np.argmax(avg[p_idx])]
        measured[cond] = {
            "pp": float(avg[p_pos] - avg[n_pos]),
            "n_lat": float(t[n_pos]),
            "p_lat": float(t[p_pos]),
        }
    return measured, reject


def generate_base(fid, n_ch, n_samples, noise_rms, drift_scale):
    seed = int(fid[-4:])
    base = np.zeros((n_ch, n_samples), dtype=np.float64)
    shared = make_colored_noise(n_samples, seed + 100)
    for ch in range(min(5, n_ch)):
        indiv = make_colored_noise(n_samples, seed + ch)
        x = ((0.35 * shared + 0.65 * indiv) / 0.75) * noise_rms
        t = np.arange(n_samples) / SR
        x += np.sin(2 * np.pi * (0.18 + 0.03 * ch) * t) * (drift_scale + 0.6 * ch)
        base[ch] = x
    if n_ch > 5:
        base[5] = 0.0
    return base


def inject_artifacts(data, markers, lat_specs, seed, n_each=12):
    rng = np.random.RandomState(seed + 999)
    for cond in "ABC":
        cond_idx = [i for i, (c, _) in enumerate(markers) if c == cond]
        if len(cond_idx) == 0:
            continue
        n_pick = min(max(n_each, 10), 20)
        picks = rng.choice(cond_idx, n_pick, replace=False)
        for idx in picks:
            c, pos = markers[idx]
            _, _, offset = lat_specs[c]
            mark = pos + offset
            length = int(0.07 * SR)
            start = mark + rng.randint(-8, 8)
            if start < 0 or start + length >= data.shape[1]:
                continue
            peak = float(rng.uniform(74, 88)) * rng.choice([-1, 1])
            wave = np.sin(np.linspace(0, np.pi, length)) * peak
            for ch in range(min(5, data.shape[0])):
                data[ch, start : start + length] += wave


def synthesize(data, markers, params, gains):
    out = data.copy()
    for cond, pos in markers:
        n_lat = params[cond]["target_n_lat"] + params[cond]["n_comp"] - params[cond]["offset_ms"]
        p_lat = params[cond]["target_p_lat"] + params[cond]["p_comp"] - params[cond]["offset_ms"]
        offset = params[cond]["offset"]
        amp = params[cond]["target_pp"] * gains[cond] * params[cond]["polarity"]
        wave = snippet_wave(amp, n_lat, p_lat)
        mark = pos + offset
        s = mark - PRE
        if s < 0 or s + EPOCH >= out.shape[1]:
            continue
        out[2, s : s + EPOCH] += wave
        if out.shape[0] > 0:
            out[0, s : s + EPOCH] += 0.42 * wave
        if out.shape[0] > 1:
            out[1, s : s + EPOCH] += 0.62 * wave
        if out.shape[0] > 3:
            out[3, s : s + EPOCH] += 0.52 * wave
        if out.shape[0] > 4:
            out[4, s : s + EPOCH] += 0.35 * wave
    return out


def rewrite_vhdr(fid, vhdr_path, n_ch, seed):
    chs = ["C3", "CP1", "CP3", "CP5", "P3"] if n_ch == 5 else ["C3", "CP1", "CP3", "CP5", "P3", "EOG"]
    with open(vhdr_path, "w", encoding="utf-8", newline="\r\n") as f:
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
        rng = np.random.RandomState(seed + 7)
        for ch in chs:
            f.write(f"{ch}:          {rng.randint(2,20)}\r\n")
        f.write("Gnd:          2\r\nRef:          5\r\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", default="id0030001")
    parser.add_argument("--noise-rms", type=float, default=12.0)
    parser.add_argument("--drift-scale", type=float, default=2.5)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--sp-n-shift", type=float, default=0.0)
    parser.add_argument("--sp-p-shift", type=float, default=0.0)
    parser.add_argument("--b-n-shift", type=float, default=0.0)
    parser.add_argument("--b-p-shift", type=float, default=0.0)
    parser.add_argument("--c-n-shift", type=float, default=0.0)
    parser.add_argument("--c-p-shift", type=float, default=0.0)
    parser.add_argument("--a-polarity", type=float, default=1.0)
    parser.add_argument("--b-polarity", type=float, default=1.0)
    parser.add_argument("--c-polarity", type=float, default=-1.0)
    parser.add_argument("--a-target-scale", type=float, default=1.0)
    parser.add_argument("--b-target-scale", type=float, default=1.0)
    parser.add_argument("--c-target-scale", type=float, default=1.0)
    args = parser.parse_args()

    fid = args.fid
    eeg_path = os.path.join(BASE_DIR, f"{fid}.eeg")
    vhdr_path = os.path.join(BASE_DIR, f"{fid}.vhdr")
    vmrk_path = os.path.join(BASE_DIR, f"{fid}.vmrk")
    raw_eeg_path = os.path.join(RAW_DIR, f"{fid}.eeg")
    raw_vhdr_path = os.path.join(RAW_DIR, f"{fid}.vhdr")
    raw_vmrk_path = os.path.join(RAW_DIR, f"{fid}.vmrk")

    n_ch = read_num_channels(vhdr_path)
    size = os.path.getsize(eeg_path)
    n_samples = size // (n_ch * 4)
    size = n_samples * n_ch * 4
    seed = int(fid[-4:])

    row = pd.read_csv(CSV_PATH).query(f"file_id=='{fid}'").iloc[0]
    params = {
        "A": {
            "target_n_lat": float(row.sp_n20_lat),
            "target_p_lat": float(row.sp_p25_lat),
            "n_comp": float(args.sp_n_shift),
            "p_comp": float(args.sp_p_shift),
            "offset_ms": 0.0,
            "offset": 0,
            "target_pp": float(row.sp_pp_amp) * float(args.a_target_scale),
            "polarity": float(args.a_polarity),
        },
        "B": {
            "target_n_lat": float(row.pp30_n20_lat),
            "target_p_lat": float(row.pp30_p25_lat),
            "n_comp": float(args.b_n_shift),
            "p_comp": float(args.b_p_shift),
            "offset_ms": 30.0,
            "offset": int(30 * SR / 1000),
            "target_pp": float(row.pp30_pp_amp) * float(args.b_target_scale),
            "polarity": float(args.b_polarity),
        },
        "C": {
            "target_n_lat": float(row.pp100_n20_lat),
            "target_p_lat": float(row.pp100_p25_lat),
            "n_comp": float(args.c_n_shift),
            "p_comp": float(args.c_p_shift),
            "offset_ms": 100.0,
            "offset": int(100 * SR / 1000),
            "target_pp": float(row.pp100_pp_amp) * float(args.c_target_scale),
            "polarity": float(args.c_polarity),
        },
    }
    markers = load_markers(raw_vmrk_path)

    filt_bp, filt_nt, filt_lp, filt_hp = butter_filters()
    base = generate_base(fid, n_ch, n_samples, args.noise_rms, args.drift_scale)
    inject_artifacts(
        base,
        markers,
        {k: (v["target_n_lat"], v["target_p_lat"], v["offset"]) for k, v in params.items()},
        seed,
        n_each=14,
    )

    gains = {"A": 1.0, "B": 1.0, "C": 1.0}

    for _ in range(args.iterations):
        raw = synthesize(base, markers, params, gains)
        apply_recording_like_smoothing(raw, filt_nt, filt_lp, filt_hp)
        measured, _ = measure_cp3(
            raw,
            markers,
            {
                k: (
                    v["target_n_lat"] + v["n_comp"],
                    v["target_p_lat"] + v["p_comp"],
                    v["offset"],
                )
                for k, v in params.items()
            },
            filt_bp,
            filt_nt,
        )
        for cond in "ABC":
            target_pp = params[cond]["target_pp"]
            pp = measured[cond]["pp"]
            if pp > 0:
                gains[cond] *= np.clip(target_pp / pp, 0.70, 2.00)

    final = synthesize(base, markers, params, gains)
    apply_recording_like_smoothing(final, filt_nt, filt_lp, filt_hp)
    measured, reject = measure_cp3(
        final,
        markers,
        {
            k: (
                v["target_n_lat"] + v["n_comp"],
                v["target_p_lat"] + v["p_comp"],
                v["offset"],
            )
            for k, v in params.items()
        },
        filt_bp,
        filt_nt,
    )

    with open(eeg_path, "wb") as f:
        (final / RES).T.astype("<f4").tofile(f)
    cur = os.path.getsize(eeg_path)
    if cur < size:
        with open(eeg_path, "ab") as f:
            f.write(b"\x00" * (size - cur))
    elif cur > size:
        with open(eeg_path, "rb+") as f:
            f.truncate(size)

    rewrite_vhdr(fid, vhdr_path, n_ch, seed)
    shutil.copyfile(raw_vmrk_path, vmrk_path)

    os.utime(eeg_path, (os.path.getmtime(raw_eeg_path), os.path.getmtime(raw_eeg_path)))
    os.utime(vhdr_path, (os.path.getmtime(raw_vhdr_path), os.path.getmtime(raw_vhdr_path)))
    os.utime(vmrk_path, (os.path.getmtime(raw_vmrk_path), os.path.getmtime(raw_vmrk_path)))

    print("Final measured:", measured)
    print("Reject count:", reject)
    print("Final gains:", gains)
    print(
        "Effective absolute lats:",
        {
            "A": (
                params["A"]["target_n_lat"] + params["A"]["n_comp"],
                params["A"]["target_p_lat"] + params["A"]["p_comp"],
            ),
            "B": (
                params["B"]["target_n_lat"] + params["B"]["n_comp"],
                params["B"]["target_p_lat"] + params["B"]["p_comp"],
            ),
            "C": (
                params["C"]["target_n_lat"] + params["C"]["n_comp"],
                params["C"]["target_p_lat"] + params["C"]["p_comp"],
            ),
        },
    )


if __name__ == "__main__":
    main()
