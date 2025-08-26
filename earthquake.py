#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def main():
    p = argparse.ArgumentParser(description="Tilt analysis + safety interpretation; saves CSV + plot (and shows with --show).")
    p.add_argument("input_csv", type=str, help="Input CSV (auto-detects delimiter). Must include Ax, Ay, Az, Time.")
    p.add_argument("--output_csv", type=str, default=None, help="Output CSV (default: <input>_tilt.csv)")
    p.add_argument("--plot_png", type=str, default=None, help="Output plot PNG (default: <input>_tilt.png)")
    p.add_argument("--baseline_sec", type=float, default=5.0, help="Seconds of initial data to set baseline")
    p.add_argument("--smooth_sec", type=float, default=1.0, help="Centered moving-average window in seconds")
    p.add_argument("--window_sec", type=float, default=5.0, help="Seconds at the END used for residual assessment")
    p.add_argument("--warn_deg", type=float, default=0.5, help="|Δθ| < warn_deg => SAFE")
    p.add_argument("--alert_deg", type=float, default=2.0, help="|Δθ| >= alert_deg => INSPECT")
    p.add_argument("--up", type=str, default="z", choices=["x","y","z"], help="Axis that was up at install")
    p.add_argument("--show", action="store_true", help="Show plot window")
    args = p.parse_args()

    in_path = args.input_csv
    out_csv = args.output_csv or in_path.replace(".csv", "_tilt.csv")
    out_png = args.plot_png or in_path.replace(".csv", "_tilt.png")

    # Load (auto-detect delimiter)
    df = pd.read_csv(in_path, sep=None, engine="python")
    # Normalize core numerics
    for c in ["Time","Ax","Ay","Az"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Time","Ax","Ay","Az"]).reset_index(drop=True)

    # Sampling interval (for smoothing window size)
    if len(df["Time"]) > 1:
        dt = df["Time"].diff().median()
        if pd.isna(dt) or dt <= 0: dt = 1.0
    else:
        dt = 1.0

    # Smoothing
    if args.smooth_sec > 0:
        win = max(1, int(round(args.smooth_sec / max(dt, 1e-9))))
        if win % 2 == 0: win += 1
        df["Ax_f"] = df["Ax"].rolling(win, center=True, min_periods=1).mean()
        df["Ay_f"] = df["Ay"].rolling(win, center=True, min_periods=1).mean()
        df["Az_f"] = df["Az"].rolling(win, center=True, min_periods=1).mean()
    else:
        df["Ax_f"], df["Ay_f"], df["Az_f"] = df["Ax"], df["Ay"], df["Az"]

    # Angles
    ax, ay, az = df["Ax_f"].to_numpy(), df["Ay_f"].to_numpy(), df["Az_f"].to_numpy()
    roll, pitch, theta = compute_angles(ax, ay, az, up_axis=args.up)
    df["roll_deg"]  = roll
    df["pitch_deg"] = pitch
    df["theta_deg"] = theta

    # Baseline (first N seconds → zero reference)
    t0 = df["Time"] - df["Time"].iloc[0]
    mask0 = t0 <= args.baseline_sec
    if not mask0.any(): mask0 = df.index < min(5, len(df))
    roll0  = df.loc[mask0, "roll_deg"].mean()
    pitch0 = df.loc[mask0, "pitch_deg"].mean()
    theta0 = df.loc[mask0, "theta_deg"].mean()

    df["d_roll_deg"]  = df["roll_deg"]  - roll0
    df["d_pitch_deg"] = df["pitch_deg"] - pitch0
    df["d_theta_deg"] = df["theta_deg"] - theta0

    # Residual assessment over the *end* window
    t_end = df["Time"].max()
    mask_end = df["Time"] >= (t_end - args.window_sec)
    resid_mean = float(df.loc[mask_end, "d_theta_deg"].mean())
    resid_std  = float(df.loc[mask_end, "d_theta_deg"].std())
    resid_last = float(df.loc[mask_end, "d_theta_deg"].iloc[-1])

    verdict, reason = assess_safety(resid_mean, args.warn_deg, args.alert_deg)

    # Plot (Residual tilt vs time) + annotation with interpretation
    plt.figure(figsize=(8,5))
    plt.plot(df["Time"], df["d_theta_deg"], label="Residual Tilt (Δθ)")
    plt.xlabel("Time [s]")
    plt.ylabel("Residual Tilt [deg]")
    plt.title("Building Tilt vs Time")
    # Text box with result
    lines = [
        f"Assessment: {verdict}",
        f"Δθ_mean(last {args.window_sec:.0f}s) = {resid_mean:.2f}°  (std {resid_std:.2f}°)",
        f"Δθ_last = {resid_last:.2f}°",
        f"Thresholds: SAFE < {args.warn_deg:.2f}°, INSPECT ≥ {args.alert_deg:.2f}°",
    ]
    txt = "\n".join(lines)
    plt.figtext(0.02, 0.02, txt, ha="left", va="bottom",
                bbox=dict(boxstyle="round", alpha=0.2))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    if args.show:
        plt.show()
    else:
        plt.close()

    # Add interpretation columns to **every row** (constant values)
    df["plot_file"] = out_png
    df["residual_window_sec"] = args.window_sec
    df["residual_mean_deg"] = resid_mean
    df["residual_std_deg"]  = resid_std
    df["residual_last_deg"] = resid_last
    df["warn_deg"] = args.warn_deg
    df["alert_deg"] = args.alert_deg
    df["safety_assessment"] = verdict
    df["assessment_reason"] = reason

    # Save CSV (robust: only write columns that exist)
    desired = [
        "Sample","Time","Ax","Ay","Az",
        "roll_deg","pitch_deg","theta_deg",
        "d_roll_deg","d_pitch_deg","d_theta_deg",
        "plot_file",
        "residual_window_sec","residual_mean_deg","residual_std_deg","residual_last_deg",
        "warn_deg","alert_deg","safety_assessment","assessment_reason",
        "BatVol","Temp"
    ]
    cols_to_write = [c for c in desired if c in df.columns]
    df[cols_to_write].to_csv(out_csv, index=False)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved plot: {out_png}")
    print(f"Assessment: {verdict} — {reason}")

if __name__ == "__main__":
    main()
