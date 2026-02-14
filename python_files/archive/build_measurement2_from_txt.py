from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("Analyzer")
SRC = Path("SEP_processed/measurements.csv")
OUT_CSV = Path("SEP_processed/measurement2.csv")


def parse_peak_txt(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    lines = [x.strip() for x in path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    if len(lines) < 2:
        return None
    h = lines[0].split()
    v = lines[1].split()
    try:
        n_lat = float(v[h.index("N2C3-L")])
        p_lat = float(v[h.index("P25C3-L")])
        n_amp = float(v[h.index("N2CP3-V")])
        p_amp = float(v[h.index("P25CP3-V")])
    except Exception:
        return None
    return {
        "n_lat": n_lat,
        "n_amp": n_amp,
        "p_lat": p_lat,
        "p_amp": p_amp,
        "pp": p_amp - n_amp,
    }


def find_pp100_file(fid: str):
    a = BASE / f"{fid}_PPI100_pp_Peaks.txt"
    b = BASE / f"{fid}_pp100_pp_Peaks.txt"
    if a.exists() and a.stat().st_size > 0:
        return a
    if b.exists() and b.stat().st_size > 0:
        return b
    return None


def recompute_changes(df: pd.DataFrame):
    out = df.copy()
    out["subject"] = out["file_id"].str.slice(2, 5)

    # suppression = (pp / sp) * 100
    out["pp30_ratio"] = out["pp30_sub_pp_amp"] / out["sp_pp_amp"] * 100.0
    out["pp100_ratio"] = out["pp100_sub_pp_amp"] / out["sp_pp_amp"] * 100.0

    out["sp_pp_amp_change"] = np.nan
    out["pp30_ratio_change"] = np.nan
    out["pp100_ratio_change"] = np.nan

    for (_, cond), g in out.groupby(["subject", "condition"]):
        pre = g[g["phase"].str.lower() == "pre"]
        post = g[g["phase"].str.lower() == "post"]
        if pre.empty or post.empty:
            continue
        pre_i = pre.index[0]
        post_i = post.index[0]

        pre_sp = out.at[pre_i, "sp_pp_amp"]
        pre_r30 = out.at[pre_i, "pp30_ratio"]
        pre_r100 = out.at[pre_i, "pp100_ratio"]

        # Change is stored as percentage.
        sp_chg = ((out.at[post_i, "sp_pp_amp"] - pre_sp) / pre_sp * 100.0) if pre_sp != 0 else np.nan
        r30_chg = ((out.at[post_i, "pp30_ratio"] - pre_r30) / pre_r30 * 100.0) if pre_r30 != 0 else np.nan
        r100_chg = ((out.at[post_i, "pp100_ratio"] - pre_r100) / pre_r100 * 100.0) if pre_r100 != 0 else np.nan

        # Store change on post row only; keep pre as NaN.
        out.at[post_i, "sp_pp_amp_change"] = sp_chg
        out.at[post_i, "pp30_ratio_change"] = r30_chg
        out.at[post_i, "pp100_ratio_change"] = r100_chg

    return out.drop(columns=["subject"])


def main():
    df = pd.read_csv(SRC)
    updated = 0

    for i, row in df.iterrows():
        fid = row["file_id"]
        sp = parse_peak_txt(BASE / f"{fid}_MinMax Markers_Peaks.txt")
        p30 = parse_peak_txt(BASE / f"{fid}_pp30_pp_Peaks.txt")
        p100_path = find_pp100_file(fid)
        p100 = parse_peak_txt(p100_path) if p100_path else None

        if sp:
            df.at[i, "sp_n20_lat"] = sp["n_lat"]
            df.at[i, "sp_n20_amp"] = sp["n_amp"]
            df.at[i, "sp_p25_lat"] = sp["p_lat"]
            df.at[i, "sp_p25_amp"] = sp["p_amp"]
            df.at[i, "sp_pp_amp"] = sp["pp"]
            updated += 1
        if p30:
            df.at[i, "pp30_sub_n20_lat"] = p30["n_lat"]
            df.at[i, "pp30_sub_n20_amp"] = p30["n_amp"]
            df.at[i, "pp30_sub_p25_lat"] = p30["p_lat"]
            df.at[i, "pp30_sub_p25_amp"] = p30["p_amp"]
            df.at[i, "pp30_sub_pp_amp"] = p30["pp"]
        if p100:
            df.at[i, "pp100_sub_n20_lat"] = p100["n_lat"]
            df.at[i, "pp100_sub_n20_amp"] = p100["n_amp"]
            df.at[i, "pp100_sub_p25_lat"] = p100["p_lat"]
            df.at[i, "pp100_sub_p25_amp"] = p100["p_amp"]
            df.at[i, "pp100_sub_pp_amp"] = p100["pp"]

    df2 = recompute_changes(df)
    df2.to_csv(OUT_CSV, index=False)
    print(f"rows={len(df2)} updated_sp_rows={updated}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
