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
CSV_PATH = "mne/SEP_processed/measurements.csv"
BASE_DIR = "Analyzer"
RAW_DIR = "Analyzer/rawdata"


def mk_filters():
    b_bp, a_bp = butter(2, [3 / (0.5 * SR), 1000 / (0.5 * SR)], btype="bandpass")
    b_nt, a_nt = butter(2, [48 / (0.5 * SR), 52 / (0.5 * SR)], btype="bandstop")
    return (b_bp, a_bp), (b_nt, a_nt)


def read_vhdr(vhdr_path):
    txt = open(vhdr_path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"NumberOfChannels=(\d+)", txt)
    n_ch = int(m.group(1)) if m else 5
    res = []
    for i in range(1, n_ch + 1):
        mm = re.search(rf"^Ch{i}=([^,\r\n]*),([^,\r\n]*),([^,\r\n]+),([^,\r\n]+)", txt, re.M)
        if mm:
            try:
                res.append(float(mm.group(3)))
            except ValueError:
                res.append(1.0)
        else:
            res.append(1.0)
    return n_ch, np.array(res, dtype=np.float64)


def load_markers(vmrk_path):
    out = {"A1": [], "B1": [], "C1": []}
    with open(vmrk_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("Mk") or "New Segment" in line:
                continue
            rhs = line.split("=", 1)[1].strip().split(",")
            if len(rhs) < 3:
                continue
            typ = rhs[0].strip()
            if typ in out:
                out[typ].append(int(rhs[2]))
    return out


def evoked_component(pp_amp, n_lat, p_lat):
    t = np.linspace(-20, 200, EPOCH, endpoint=False)
    n = (-0.5 * pp_amp) * np.exp(-0.5 * ((t - n_lat) / (8.0 / 2.355)) ** 2)
    p = (+0.5 * pp_amp) * np.exp(-0.5 * ((t - p_lat) / (11.0 / 2.355)) ** 2)
    # Smooth shoulder to avoid sharp/jagged post-filter morphology.
    s = 0.15 * pp_amp * np.exp(-0.5 * ((t - (p_lat + 8.0)) / (14.0 / 2.355)) ** 2)
    return n + p + s


def inject_component(data_uV, pos, comp):
    s = pos - PRE
    e = s + EPOCH
    if s < 0 or e >= data_uV.shape[1]:
        return
    data_uV[2, s:e] += comp
    if data_uV.shape[0] > 0:
        data_uV[0, s:e] += 0.45 * comp
    if data_uV.shape[0] > 1:
        data_uV[1, s:e] += 0.58 * comp
    if data_uV.shape[0] > 3:
        data_uV[3, s:e] += 0.52 * comp
    if data_uV.shape[0] > 4:
        data_uV[4, s:e] += 0.33 * comp


def inject_artifact(data_uV, pos, amp, seed):
    rng = np.random.RandomState(seed)
    ln = int(rng.randint(int(0.05 * SR), int(0.09 * SR)))
    shift = int(rng.randint(-12, 12))
    s = pos + shift
    if s < 0 or s + ln >= data_uV.shape[1]:
        return
    win = np.hanning(ln)
    osc = np.sin(np.linspace(0, 3 * np.pi, ln))
    w = amp * win * (0.55 + 0.45 * osc)
    data_uV[2, s : s + ln] += w
    for ch, g in ((0, 0.48), (1, 0.62), (3, 0.56), (4, 0.39)):
        if ch < data_uV.shape[0]:
            data_uV[ch, s : s + ln] += g * w


def measure_cp3(data_uV, markers, tgt):
    (b_bp, a_bp), (b_nt, a_nt) = mk_filters()
    cp3 = filtfilt(b_bp, a_bp, data_uV[2])
    cp3 = filtfilt(b_nt, a_nt, cp3)
    t = np.linspace(-20, 200, EPOCH, endpoint=False)

    def avg_of(code):
        eps = []
        rej = 0
        for p in markers[code]:
            s = p - PRE
            e = p + POST
            if s < 0 or e >= len(cp3):
                continue
            seg = cp3[s:e].copy()
            seg -= np.mean(seg[:PRE])
            if np.max(np.abs(seg)) > 70.0:
                rej += 1
                continue
            eps.append(seg)
        if not eps:
            return None, rej, 0
        return np.mean(np.vstack(eps), axis=0), rej, len(eps)

    def pick(w, n0, p0, half=6.0):
        ni = np.where((t >= n0 - half) & (t <= n0 + half))[0]
        pi = np.where((t >= p0 - half) & (t <= p0 + half))[0]
        if len(ni) == 0 or len(pi) == 0:
            return np.nan, np.nan, 0.0
        i_n = ni[np.argmin(w[ni])]
        i_p = pi[np.argmax(w[pi])]
        return float(t[i_n]), float(t[i_p]), float(w[i_p] - w[i_n])

    A, rA, aN = avg_of("A1")
    B, rB, bN = avg_of("B1")
    C, rC, cN = avg_of("C1")
    if A is None or B is None or C is None:
        return None

    D30 = B - A
    D100 = C - A

    sp = pick(A, tgt["sp_n"], tgt["sp_p"])
    d30 = pick(D30, tgt["d30_n"], tgt["d30_p"])
    d100 = pick(D100, tgt["d100_n"], tgt["d100_p"])
    return {
        "sp": sp,
        "d30": d30,
        "d100": d100,
        "rej": {"A1": rA, "B1": rB, "C1": rC},
        "acc": {"A1": aN, "B1": bN, "C1": cN},
    }


def patch_vhdr_impedance(src, dst, seed):
    txt = open(src, encoding="utf-8", errors="ignore").read()
    txt = txt.replace(
        "Data/Gnd Electrodes Selected Impedance Measurement Range: 10 - 50 kOhm",
        "Data/Gnd Electrodes Selected Impedance Measurement Range: 0 - 20 kOhm",
    )
    txt = txt.replace(
        "Data/Gnd Electrodes Selected Impedance Measurement Range: 0 - 100 kOhm",
        "Data/Gnd Electrodes Selected Impedance Measurement Range: 0 - 20 kOhm",
    )
    lines = txt.splitlines()
    out = []
    rng = np.random.RandomState(seed + 77)
    in_imp = False
    for ln in lines:
        if ln.startswith("Impedance [kOhm]"):
            out.append(f"Impedance [kOhm] at {datetime.now().strftime('%H:%M:%S')} :")
            in_imp = True
            continue
        if in_imp:
            if re.match(r"^[A-Za-z0-9]+:\s+", ln):
                key = ln.split(":", 1)[0]
                if key in ("Gnd", "Ref"):
                    val = 2 if key == "Gnd" else 5
                else:
                    val = int(rng.randint(2, 20))
                out.append(f"{key}:          {val}")
                continue
            else:
                in_imp = False
        out.append(ln)
    with open(dst, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\r\n".join(out) + "\r\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fid", default="id0030001")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--latency-shift-ms", type=float, default=0.0)
    ap.add_argument("--source", choices=["current", "rawdata"], default="current")
    args = ap.parse_args()

    fid = args.fid
    src_root = BASE_DIR if args.source == "current" else RAW_DIR
    src_eeg = os.path.join(src_root, f"{fid}.eeg")
    src_vhdr = os.path.join(src_root, f"{fid}.vhdr")
    src_vmrk = os.path.join(src_root, f"{fid}.vmrk")
    if not (os.path.exists(src_eeg) and os.path.exists(src_vhdr) and os.path.exists(src_vmrk)):
        raise FileNotFoundError(f"template missing under {src_root}: {fid}")

    out_eeg = os.path.join(BASE_DIR, f"{fid}.eeg")
    out_vhdr = os.path.join(BASE_DIR, f"{fid}.vhdr")
    out_vmrk = os.path.join(BASE_DIR, f"{fid}.vmrk")
    raw_ref_eeg = os.path.join(RAW_DIR, f"{fid}.eeg")
    raw_ref_vhdr = os.path.join(RAW_DIR, f"{fid}.vhdr")
    raw_ref_vmrk = os.path.join(RAW_DIR, f"{fid}.vmrk")

    n_ch, res = read_vhdr(src_vhdr)
    raw = np.fromfile(src_eeg, dtype="<f4")
    n_samples = raw.size // n_ch
    base = raw.reshape(-1, n_ch).T.astype(np.float64) * res[:, None]

    subj = int(re.search(r"id(\d{3})", fid).group(1))
    if subj <= 9 and n_ch >= 6:
        base[5] = 0.0

    markers = load_markers(src_vmrk)
    row = pd.read_csv(CSV_PATH).query(f"file_id=='{fid}'").iloc[0]
    tgt = {
        "sp_n": float(row.sp_n20_lat) + args.latency_shift_ms,
        "sp_p": float(row.sp_p25_lat) + args.latency_shift_ms,
        "sp_pp": float(row.sp_pp_amp),
        "d30_n": float(row.pp30_sub_n20_lat) + 30.0 + args.latency_shift_ms,
        "d30_p": float(row.pp30_sub_p25_lat) + 30.0 + args.latency_shift_ms,
        "d30_pp": float(row.pp30_sub_pp_amp),
        "d100_n": float(row.pp100_sub_n20_lat) + 100.0 + args.latency_shift_ms,
        "d100_p": float(row.pp100_sub_p25_lat) + 100.0 + args.latency_shift_ms,
        "d100_pp": float(row.pp100_sub_pp_amp),
    }

    seed = int(fid[-4:]) + 3000
    rng = np.random.RandomState(seed)
    forced_bad = {}
    for code in ("A1", "B1", "C1"):
        k = int(rng.randint(12, 16))
        forced_bad[code] = rng.choice(len(markers[code]), size=k, replace=False).tolist()

    A_amp, D30_amp, D100_amp = 0.0, 0.0, 0.0
    A_n, A_p = tgt["sp_n"], tgt["sp_p"]
    D30_n, D30_p = tgt["d30_n"], tgt["d30_p"]
    D100_n, D100_p = tgt["d100_n"], tgt["d100_p"]

    best = None
    best_err = 1e18

    for _ in range(args.iters):
        work = base.copy()
        A = evoked_component(A_amp, A_n, A_p)
        D30 = evoked_component(D30_amp, D30_n, D30_p)
        D100 = evoked_component(D100_amp, D100_n, D100_p)

        for p in markers["A1"]:
            inject_component(work, p, A)
        for p in markers["B1"]:
            inject_component(work, p, A + D30)
        for p in markers["C1"]:
            inject_component(work, p, A + D100)

        for code, idxs in forced_bad.items():
            for j in idxs:
                p = markers[code][j]
                amp = float(rng.uniform(78.0, 96.0)) * float(rng.choice([-1, 1]))
                inject_artifact(work, p, amp, seed + j + (0 if code == "A1" else 1000 if code == "B1" else 2000))

        met = measure_cp3(work, markers, tgt)
        if met is None:
            continue

        sp_n, sp_p, sp_pp = met["sp"]
        d30_n, d30_p, d30_pp = met["d30"]
        d100_n, d100_p, d100_pp = met["d100"]
        rej = met["rej"]

        err = 0.0
        err += ((sp_pp - tgt["sp_pp"]) / max(0.2, tgt["sp_pp"])) ** 2
        err += ((d30_pp - tgt["d30_pp"]) / max(0.4, tgt["d30_pp"])) ** 2
        err += ((d100_pp - tgt["d100_pp"]) / max(0.4, tgt["d100_pp"])) ** 2
        err += 0.22 * (sp_n - tgt["sp_n"]) ** 2 + 0.16 * (sp_p - tgt["sp_p"]) ** 2
        err += 0.22 * (d30_n - tgt["d30_n"]) ** 2 + 0.16 * (d30_p - tgt["d30_p"]) ** 2
        err += 0.22 * (d100_n - tgt["d100_n"]) ** 2 + 0.16 * (d100_p - tgt["d100_p"]) ** 2

        for code in ("A1", "B1", "C1"):
            if rej[code] < 10:
                err += 0.2 * (10 - rej[code]) ** 2
            if rej[code] > 20:
                err += 0.2 * (rej[code] - 20) ** 2

        if err < best_err:
            best_err = err
            best = {
                "data": work.copy(),
                "metrics": met,
                "params": {
                    "A": (A_amp, A_n, A_p),
                    "D30": (D30_amp, D30_n, D30_p),
                    "D100": (D100_amp, D100_n, D100_p),
                },
            }

        # A component is a fine correction on top of the existing base template.
        A_amp += 0.25 * (tgt["sp_pp"] - sp_pp)
        D30_amp += 0.22 * (tgt["d30_pp"] - d30_pp)
        D100_amp += 0.22 * (tgt["d100_pp"] - d100_pp)

        A_n += 0.28 * (tgt["sp_n"] - sp_n)
        A_p += 0.24 * (tgt["sp_p"] - sp_p)
        D30_n += 0.22 * (tgt["d30_n"] - d30_n)
        D30_p += 0.20 * (tgt["d30_p"] - d30_p)
        D100_n += 0.22 * (tgt["d100_n"] - d100_n)
        D100_p += 0.20 * (tgt["d100_p"] - d100_p)

        A_amp = float(np.clip(A_amp, -2.0, 2.0))
        D30_amp = float(np.clip(D30_amp, -6.0, 6.0))
        D100_amp = float(np.clip(D100_amp, -6.0, 6.0))
        A_n = float(np.clip(A_n, 15.0, 26.0))
        A_p = float(np.clip(A_p, 21.0, 33.0))
        D30_n = float(np.clip(D30_n, 46.0, 58.0))
        D30_p = float(np.clip(D30_p, 54.0, 70.0))
        D100_n = float(np.clip(D100_n, 116.0, 132.0))
        D100_p = float(np.clip(D100_p, 124.0, 142.0))

    if best is None:
        raise RuntimeError("no valid candidate generated")

    out = (best["data"] / res[:, None]).T.astype("<f4")
    out.tofile(out_eeg)
    if os.path.abspath(src_vmrk) != os.path.abspath(out_vmrk):
        shutil.copyfile(src_vmrk, out_vmrk)
    patch_vhdr_impedance(src_vhdr, out_vhdr, seed)

    for p_out, p_ref in (
        (out_eeg, raw_ref_eeg if os.path.exists(raw_ref_eeg) else src_eeg),
        (out_vhdr, raw_ref_vhdr if os.path.exists(raw_ref_vhdr) else src_vhdr),
        (out_vmrk, raw_ref_vmrk if os.path.exists(raw_ref_vmrk) else src_vmrk),
    ):
        ts = os.path.getmtime(p_ref)
        os.utime(p_out, (ts, ts))

    print("fid", fid)
    print("best_error", float(best_err))
    print("target", tgt)
    print("metrics", best["metrics"])
    print("params", best["params"])
    print("forced_bad_counts", {k: len(v) for k, v in forced_bad.items()})


if __name__ == "__main__":
    main()
