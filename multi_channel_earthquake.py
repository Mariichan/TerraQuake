#!/usr/bin/env python3
import argparse, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------- Parsing ----------------------------

def parse_three_channel_log(txt_path: Path):
    """
    Parse lines like any of:
      From Sensor CH0 acc: 0.631641,-1.186719,9.110937
      From Sensor CH1 acc: -1.416406,-1.052734,19.466015
      CH2 acc: -1.416406,-1.052734,19.466015
      CH0: 0.1, 0.2, 0.3
    Returns dict {0: df0, 1: df1, 2: df2} with columns: Sample, Time, Ax, Ay, Az
    Time is synthesized as (sample-1) seconds (assumes ~1 Hz per channel).
    """
    pat = re.compile(
        r'(?:From\s+Sensor\s+)?CH\s*(?P<ch>[012])(?:\s*acc)?\s*:\s*'
        r'(?P<x>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)[,\s]+'
        r'(?P<y>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)[,\s]+'
        r'(?P<z>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    )

    samples = {0: [], 1: [], 2: []}
    counts  = {0: 0, 1: 0, 2: 0}

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            ch = int(m.group("ch"))
            x  = float(m.group("x"))
            y  = float(m.group("y"))
            z  = float(m.group("z"))
            counts[ch] += 1
            sample = counts[ch]
            time_s = float(sample - 1)  # assume 1 Hz per channel
            samples[ch].append((sample, time_s, x, y, z))

    out = {}
    for ch in (0, 1, 2):
        df = pd.DataFrame(samples[ch], columns=["Sample", "Time", "Ax", "Ay", "Az"])
        out[ch] = df
    return out

# ------------------------- Tilt & Safety -------------------------

def compute_angles(ax, ay, az, up_axis="z"):
    norm  = np.sqrt(ax*ax + ay*ay + az*az) + 1e-12
    roll  = np.degrees(np.arctan2(ay, az))
    pitch = np.degrees(np.arctan2(-ax, np.sqrt(ay*ay + az*az)))
    up_comp = {"x": ax, "y": ay, "z": az}[up_axis.lower()]
    theta = np.degrees(np.arccos(np.clip(up_comp / norm, -1.0, 1.0)))
    return roll, pitch, theta

def assess_safety(resid_mean_deg: float, warn_deg: float, alert_deg: float):
    a = abs(resid_mean_deg)
    if a < warn_deg:
        return "SAFE", f"|Δθ|={a:.2f}° < {warn_deg:.2f}° (within noise)"
    elif a < alert_deg:
        return "NOTE", f"|Δθ|={a:.2f}° between {warn_deg:.2f}°–{alert_deg:.2f}° (minor residual tilt)"
    else:
        return "INSPECT", f"|Δθ|={a:.2f}° ≥ {alert_deg:.2f}° (consider structural check)"

def analyze_channel(df, out_stem: Path, baseline_sec=5.0, smooth_sec=1.0, window_sec=5.0,
                    up="z", warn_deg=0.5, alert_deg=2.0, show=False):
    """
    df: DataFrame with columns Sample, Time, Ax, Ay, Az
    Saves:
      - tilt CSV with safety columns: <stem>_tilt.csv (semicolon)
      - plot PNG: <stem>.png
    Returns: (tilt_csv_path, png_path, verdict, reason, df_out)
    """
    # Estimate dt for smoothing window
    if len(df["Time"]) > 1:
        dt = df["Time"].diff().median()
        if pd.isna(dt) or dt <= 0:
            dt = 1.0
    else:
        dt = 1.0

    # Optional smoothing (centered moving average)
    if smooth_sec > 0:
        win = max(1, int(round(smooth_sec / max(dt, 1e-9))))
        if win % 2 == 0:
            win += 1
        ax = df["Ax"].rolling(win, center=True, min_periods=1).mean().to_numpy()
        ay = df["Ay"].rolling(win, center=True, min_periods=1).mean().to_numpy()
        az = df["Az"].rolling(win, center=True, min_periods=1).mean().to_numpy()
    else:
        ax, ay, az = df["Ax"].to_numpy(), df["Ay"].to_numpy(), df["Az"].to_numpy()

    roll, pitch, theta = compute_angles(ax, ay, az, up_axis=up)
    out = df.copy()
    out["roll_deg"]  = roll
    out["pitch_deg"] = pitch
    out["theta_deg"] = theta

    # Baseline = first N seconds → zero reference
    t0 = out["Time"] - out["Time"].iloc[0]
    mask0 = t0 <= baseline_sec
    if not mask0.any():
        mask0 = out.index < min(5, len(out))
    roll0  = out.loc[mask0, "roll_deg"].mean()
    pitch0 = out.loc[mask0, "pitch_deg"].mean()
    theta0 = out.loc[mask0, "theta_deg"].mean()

    out["d_roll_deg"]  = out["roll_deg"]  - roll0
    out["d_pitch_deg"] = out["pitch_deg"] - pitch0
    out["d_theta_deg"] = out["theta_deg"] - theta0

    # Residual assessment over the LAST window_sec
    t_end = out["Time"].max()
    mask_end = out["Time"] >= (t_end - window_sec)
    resid_mean = float(out.loc[mask_end, "d_theta_deg"].mean())
    resid_std  = float(out.loc[mask_end, "d_theta_deg"].std())
    resid_last = float(out.loc[mask_end, "d_theta_deg"].iloc[-1])
    verdict, reason = assess_safety(resid_mean, warn_deg, alert_deg)

    # Save tilt CSV (semicolon to match your analyzer)
    tilt_csv = out_stem.with_suffix("").as_posix() + "_tilt.csv"
    out_semicol = out[[
        "Sample","Time","Ax","Ay","Az",
        "roll_deg","pitch_deg","theta_deg",
        "d_roll_deg","d_pitch_deg","d_theta_deg"
    ]].copy()
    out_semicol["plot_file"]            = out_stem.with_suffix(".png").name
    out_semicol["residual_window_sec"]  = window_sec
    out_semicol["residual_mean_deg"]    = resid_mean
    out_semicol["residual_std_deg"]     = resid_std
    out_semicol["residual_last_deg"]    = resid_last
    out_semicol["warn_deg"]             = warn_deg
    out_semicol["alert_deg"]            = alert_deg
    out_semicol["safety_assessment"]    = verdict
    out_semicol["assessment_reason"]    = reason
    out_semicol.to_csv(tilt_csv, sep=";", index=False)

    # Plot Δθ vs Time with verdict box; use only channel name in title
    png = out_stem.with_suffix(".png")
    plt.figure(figsize=(8,5))
    plt.plot(out["Time"], out["d_theta_deg"], label="Residual Tilt (Δθ)")
    plt.xlabel("Time [s]")
    plt.ylabel("Residual Tilt [deg]")
    ch_name = out_stem.name.split("_")[-1]  # -> "CH0", "CH1", "CH2"
    plt.title(f"{ch_name} — Tilt vs Time")
    lines = [
        f"Assessment: {verdict}",
        f"Δθ_mean(last {window_sec:.0f}s) = {resid_mean:.2f}°  (std {resid_std:.2f}°)",
        f"Δθ_last = {resid_last:.2f}°",
        f"Thresholds: SAFE < {warn_deg:.2f}°, INSPECT ≥ {alert_deg:.2f}°",
    ]
    # OLD style: in-axes box (overlays the graph)
    plt.figtext(0.02, 0.02, "\n".join(lines), ha="left", va="bottom",
                bbox=dict(boxstyle="round", alpha=0.2))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png)
    if show:
        plt.show()
    else:
        plt.close()

    return tilt_csv, png, verdict, reason, out

# --------------------------- Main CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Parse 3-channel accel log -> per-channel CSVs, analyze, and plot."
    )
    ap.add_argument("input_txt", type=str, help="Path to the text log (CH0/CH1/CH2 lines).")
    ap.add_argument("--up", choices=["x","y","z"], default="z", help="Axis that was 'up' at install.")
    ap.add_argument("--baseline_sec", type=float, default=5.0, help="Seconds for baseline mean.")
    ap.add_argument("--smooth_sec", type=float, default=1.0, help="Centered moving-average window in seconds.")
    ap.add_argument("--window_sec", type=float, default=5.0, help="Seconds at END used for residual assessment.")
    ap.add_argument("--warn_deg", type=float, default=0.5, help="|Δθ| < warn_deg => SAFE.")
    ap.add_argument("--alert_deg", type=float, default=2.0, help="|Δθ| >= alert_deg => INSPECT.")
    ap.add_argument("--combined_png", type=str, default=None,
                    help="Optional path to save a combined overlay plot of CH0/CH1/CH2.")
    ap.add_argument("--show", action="store_true", help="Show plots interactively.")
    args = ap.parse_args()

    txt_path = Path(args.input_txt)
    base = txt_path.with_suffix("")  # remove .txt extension

    # 1) Parse to three channel DataFrames
    chans = parse_three_channel_log(txt_path)

    # 2) Write raw-style CSVs (semicolon) for each channel so your existing analyzer can read them
    raw_csvs = {}
    for ch, df in chans.items():
        raw_csv = base.as_posix() + f"_CH{ch}.csv"
        df.to_csv(raw_csv, sep=";", index=False)
        raw_csvs[ch] = raw_csv

    # 3) Analyze each channel and plot
    results = {}
    outs = {}
    for ch, df in chans.items():
        out_stem = base.parent / f"{base.name}_CH{ch}"
        tilt_csv, png, verdict, reason, out_df = analyze_channel(
            df, out_stem,
            baseline_sec=args.baseline_sec,
            smooth_sec=args.smooth_sec,
            window_sec=args.window_sec,
            up=args.up,
            warn_deg=args.warn_deg,
            alert_deg=args.alert_deg,
            show=args.show
        )
        results[ch] = (tilt_csv, png, verdict, reason)
        outs[ch] = out_df

    # 4) Optional combined overlay plot (three lines on one figure)
    if args.combined_png is not None and all(ch in outs for ch in (0,1,2)):
        plt.figure(figsize=(9,5))
        for ch in (0,1,2):
            dfc = outs[ch]
            plt.plot(dfc["Time"], dfc["d_theta_deg"], label=f"CH{ch} Δθ")
        plt.xlabel("Time [s]")
        plt.ylabel("Residual Tilt [deg]")
        plt.title("CH0 / CH1 / CH2 — Combined Δθ")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(args.combined_png)
        if args.show:
            plt.show()
        else:
            plt.close()

    # 5) Console summary
    print("\n=== Summary ===")
    for ch in (0,1,2):
        tilt_csv, png, verdict, reason = results[ch]
        print(f"CH{ch}: {verdict} — {reason}")
        print(f"     raw CSV:   {raw_csvs[ch]}")
        print(f"     tilt CSV:  {tilt_csv}")
        print(f"     plot PNG:  {png}")
    if args.combined_png:
        print(f"\nCombined plot saved: {args.combined_png}")

if __name__ == "__main__":
    main()
