#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def robust_displacement_from_tracks(
    tracks,
    visibility=None,
    axis="y",
    vis_thresh=0.5,
    trim_ratio=0.1,
):
    """
    tracks: [T, N, 2]
    visibility: [T, N] or [T, N, 1] or None
    axis: "x" or "y"
    returns:
        disp_per_point: [T, N]
        disp_curve: [T]
    """
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise RuntimeError(f"Expected tracks shape [T,N,2], got {tracks.shape}")

    T, N, _ = tracks.shape

    if visibility is not None:
        if visibility.ndim == 3 and visibility.shape[-1] == 1:
            visibility = visibility[..., 0]
        if visibility.shape != (T, N):
            raise RuntimeError(f"Expected visibility shape [T,N], got {visibility.shape}")

    coord_idx = 0 if axis == "x" else 1

    # displacement relative to frame 0
    disp_per_point = tracks[..., coord_idx] - tracks[0:1, :, coord_idx]

    disp_curve = np.zeros(T, dtype=np.float32)

    for t in range(T):
        vals = disp_per_point[t].copy()

        if visibility is not None:
            vals = vals[visibility[t] >= vis_thresh]

        if vals.size == 0:
            disp_curve[t] = np.nan
            continue

        vals = np.sort(vals)
        k = int(len(vals) * trim_ratio)

        if 2 * k < len(vals):
            vals = vals[k:len(vals)-k]

        disp_curve[t] = np.median(vals)

    # fill NaN if any
    if np.isnan(disp_curve).any():
        good = np.where(~np.isnan(disp_curve))[0]
        bad = np.where(np.isnan(disp_curve))[0]
        if len(good) > 0:
            disp_curve[bad] = np.interp(bad, good, disp_curve[good])
        else:
            disp_curve[:] = 0.0

    # zero mean
    disp_curve = disp_curve - np.mean(disp_curve)

    return disp_per_point, disp_curve


def compute_fft(signal, fps):
    """
    signal: [T]
    """
    signal = np.asarray(signal, dtype=np.float32)
    signal = signal - np.mean(signal)

    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    spec = np.abs(np.fft.rfft(signal))
    return freqs, spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True, help="tracks.npy path")
    parser.add_argument("--visibility", required=True, help="visibility.npy path")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--axis", choices=["x", "y"], default="y")
    parser.add_argument("--vis_thresh", type=float, default=0.5)
    parser.add_argument("--trim_ratio", type=float, default=0.1)
    parser.add_argument("--max_freq", type=float, default=15.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    tracks = np.load(args.tracks)           # [T,N,2]
    visibility = np.load(args.visibility)   # [T,N,1] or [T,N]

    disp_per_point, disp_curve = robust_displacement_from_tracks(
        tracks,
        visibility=visibility,
        axis=args.axis,
        vis_thresh=args.vis_thresh,
        trim_ratio=args.trim_ratio,
    )

    T = len(disp_curve)
    t = np.arange(T) / args.fps

    freqs, spec = compute_fft(disp_curve, args.fps)

    # save arrays
    np.save(out_dir / "disp_per_point.npy", disp_per_point)
    np.save(out_dir / "displacement_curve.npy", disp_curve)
    np.save(out_dir / "fft_freqs.npy", freqs)
    np.save(out_dir / "fft_spec.npy", spec)

    with open(out_dir / "displacement_curve.csv", "w") as f:
        f.write("index,time,displacement\n")
        for i, (tt, dd) in enumerate(zip(t, disp_curve)):
            f.write(f"{i},{tt:.8f},{dd:.8f}\n")

    # plot time curve
    plt.figure(figsize=(12, 4))
    plt.plot(t, disp_curve, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel(f"{args.axis}-displacement (pixels)")
    plt.title("Displacement from CoTracker trajectories")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "displacement_curve.png", dpi=200)
    plt.close()

    # plot fft
    plt.figure(figsize=(10, 4))
    sel = freqs <= args.max_freq
    plt.plot(freqs[sel], spec[sel], linewidth=1.0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("FFT of displacement curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "displacement_fft.png", dpi=200)
    plt.close()

    meta = {
        "tracks": str(Path(args.tracks).resolve()),
        "visibility": str(Path(args.visibility).resolve()),
        "fps": args.fps,
        "axis": args.axis,
        "vis_thresh": args.vis_thresh,
        "trim_ratio": args.trim_ratio,
        "num_frames": int(T),
        "num_points": int(tracks.shape[1]),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] saved to {out_dir}")
    print(f"curve: {out_dir / 'displacement_curve.png'}")
    print(f"fft  : {out_dir / 'displacement_fft.png'}")


if __name__ == "__main__":
    main()