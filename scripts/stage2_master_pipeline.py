#!/usr/bin/env python3
"""
Master pipeline for Stage 2 bridge/video analysis.

What it does in one run:
1) Read an input video and extract only a chosen frame interval.
2) Run SAM2 on the original interval frames with a user box.
3) Save masks and build an overlay video.
4) Sample query points from the first mask automatically.
5) Run CoTracker on the original interval.
6) Save tracking overlay video.
7) Save per-point displacement, aggregate displacement, plots, and FFT.
8) Run FD4MM magnification with user-selected amplification factor.
9) Save magnified video and side-by-side original vs magnified video.
10) Run SAM2 again on the magnified frames with a second user box.
11) Save masks and build an overlay video for the magnified video.
12) Build a side-by-side overlay comparison video.
13) Run CoTracker on the magnified frames.
14) Save tracking overlay video for the magnified frames.
15) Build a side-by-side tracking comparison video.
16) Save per-point displacement, aggregate displacement, plots, FFT, and comparison plots.

This script is designed as an orchestrator. It reuses the scripts you already have:
- scripts/segment_bridge_sam2.py
- scripts/run_cotracker_bridge.py
- scripts/magnify_bridge_video_fd4mm.py

Point sampling, frame extraction, video stitching, displacement plotting, and comparison plotting
are implemented here so the whole experiment can be launched with one command.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path):
    s = str(path.name)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    files = sorted(files, key=natural_key)
    if not files:
        raise RuntimeError(f"No images found in {folder}")
    return files


def run_cmd(cmd: Sequence[str], cwd: Path | None = None):
    print("[CMD]", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None, check=True)


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def put_text(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.putText(
        out,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


@dataclass
class PipelineConfig:
    project_root: str
    video_path: str
    run_name: str
    start_frame: int
    end_frame: int
    fps: float
    orig_box: Tuple[int, int, int, int]
    mag_box: Tuple[int, int, int, int]
    amp: float
    max_points: int
    grid_spacing: int
    x_start_ratio: float
    x_end_ratio: float
    y_start_ratio: float
    y_end_ratio: float
    device: str
    sam2_cfg: str
    sam2_ckpt: str
    fd4mm_root: str
    fd4mm_ckpt: str
    tail_len: int
    axis: int


class MasterPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.root = Path(cfg.project_root).resolve()
        self.results_root = self.root / "results" / cfg.run_name
        ensure_dir(self.results_root)
        self._save_config()

    def _save_config(self):
        with open(self.results_root / "config.json", "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

    def extract_interval_frames(self) -> Path:
        out_dir = self.results_root / "orig" / "frames"
        ensure_dir(out_dir)

        cap = cv2.VideoCapture(self.cfg.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.cfg.video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = self.cfg.start_frame
        end = self.cfg.end_frame
        if start < 0 or end < start or end >= total:
            raise ValueError(f"Invalid frame interval [{start}, {end}] for total frames={total}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = 0
        while True:
            cur = start + idx
            if cur > end:
                break
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame)
            idx += 1
        cap.release()

        if idx != end - start + 1:
            print(f"[WARN] expected {end-start+1} frames, extracted {idx}")
        print(f"[DONE] extracted {idx} frames to {out_dir}")
        return out_dir

    def build_video_from_frames(self, frames_dir: Path, out_video: Path, fps: float, label: str | None = None):
        frame_paths = list_images(frames_dir)
        first = read_bgr(frame_paths[0])
        H, W = first.shape[:2]
        ensure_dir(out_video.parent)
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open writer: {out_video}")
        for i, p in enumerate(frame_paths):
            frame = read_bgr(p)
            if label:
                frame = put_text(frame, label)
            writer.write(frame)
            if (i + 1) % 100 == 0 or (i + 1) == len(frame_paths):
                print(f"[INFO] wrote {i+1}/{len(frame_paths)} frames to {out_video.name}")
        writer.release()

    def build_side_by_side_video(self, video1: Path, video2: Path, out_video: Path, label1: str, label2: str):
        cap1 = cv2.VideoCapture(str(video1))
        cap2 = cv2.VideoCapture(str(video2))
        if not cap1.isOpened() or not cap2.isOpened():
            raise RuntimeError(f"Failed to open side-by-side inputs: {video1}, {video2}")

        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        fps = fps1 if fps1 > 0 else (fps2 if fps2 > 0 else self.cfg.fps)

        ok1, f1 = cap1.read()
        ok2, f2 = cap2.read()
        if not ok1 or not ok2:
            raise RuntimeError("Failed to read first frame for side-by-side video")

        H = max(f1.shape[0], f2.shape[0])

        def resize_to_height(img: np.ndarray, H_: int) -> np.ndarray:
            h, w = img.shape[:2]
            if h == H_:
                return img
            new_w = int(round(w * H_ / h))
            return cv2.resize(img, (new_w, H_), interpolation=cv2.INTER_LINEAR)

        f1 = resize_to_height(f1, H)
        f2 = resize_to_height(f2, H)
        W = f1.shape[1] + f2.shape[1]
        ensure_dir(out_video.parent)
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open writer: {out_video}")

        idx = 0
        while True:
            if idx > 0:
                ok1, f1 = cap1.read()
                ok2, f2 = cap2.read()
                if not ok1 or not ok2:
                    break
                f1 = resize_to_height(f1, H)
                f2 = resize_to_height(f2, H)

            left = put_text(f1, label1)
            right = put_text(f2, label2)
            combo = np.concatenate([left, right], axis=1)
            writer.write(combo)
            idx += 1
        cap1.release()
        cap2.release()
        writer.release()
        print(f"[DONE] side-by-side video saved: {out_video}")

    def run_sam2(self, frames_dir: Path, out_dir: Path, box: Tuple[int, int, int, int]):
        ensure_dir(out_dir)
        cmd = [
            sys.executable,
            str(self.root / "scripts" / "segment_bridge_sam2.py"),
            "--frames_dir", str(frames_dir),
            "--out_dir", str(out_dir),
            "--sam2_cfg", self.cfg.sam2_cfg,
            "--sam2_ckpt", self.cfg.sam2_ckpt,
            "--box", *map(str, box),
            "--save_overlay",
            "--largest_only",
        ]
        run_cmd(cmd, cwd=self.root)

        overlay_dir = out_dir / "overlay"
        overlay_video = out_dir / "overlay.mp4"
        self.build_video_from_frames(overlay_dir, overlay_video, self.cfg.fps, label="SAM2 overlay")
        return overlay_video

    def _keep_middle_stripe(self, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise RuntimeError("Mask is empty, cannot sample points")
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = x_max - x_min + 1
        h = y_max - y_min + 1

        xs0 = int(round(x_min + self.cfg.x_start_ratio * w))
        xs1 = int(round(x_min + self.cfg.x_end_ratio * w))
        ys0 = int(round(y_min + self.cfg.y_start_ratio * h))
        ys1 = int(round(y_min + self.cfg.y_end_ratio * h))

        out = np.zeros_like(mask)
        out[ys0:ys1, xs0:xs1] = mask[ys0:ys1, xs0:xs1]
        return out

    def sample_points_from_mask(self, first_mask_path: Path, out_dir: Path) -> Path:
        ensure_dir(out_dir)
        mask = cv2.imread(str(first_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {first_mask_path}")
        mask = (mask > 127).astype(np.uint8)
        stripe = self._keep_middle_stripe(mask)

        ys, xs = np.where(stripe > 0)
        if len(xs) == 0:
            raise RuntimeError("Stripe mask is empty, cannot sample points")

        h, w = stripe.shape
        points = []
        step = max(1, self.cfg.grid_spacing)
        for y in range(0, h, step):
            for x in range(0, w, step):
                if stripe[y, x] > 0:
                    points.append((x, y))
        if not points:
            raise RuntimeError("No points sampled from stripe mask")

        if len(points) > self.cfg.max_points:
            # deterministic downsample
            idx = np.linspace(0, len(points) - 1, self.cfg.max_points).round().astype(int)
            points = [points[i] for i in idx]

        points_xy = np.array(points, dtype=np.float32)
        queries = np.concatenate([
            np.zeros((len(points_xy), 1), dtype=np.float32),
            points_xy
        ], axis=1)

        np.save(out_dir / "bridge_points_xy.npy", points_xy)
        np.save(out_dir / "bridge_queries.npy", queries)

        color = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        color_stripe = cv2.cvtColor((stripe * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for x, y in points:
            cv2.circle(color, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.circle(color_stripe, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.imwrite(str(out_dir / "bridge_points_on_mask.png"), color)
        cv2.imwrite(str(out_dir / "bridge_points_on_stripe_mask.png"), color_stripe)

        meta = {
            "first_mask": str(first_mask_path),
            "num_points": int(len(points_xy)),
            "grid_spacing": int(self.cfg.grid_spacing),
            "max_points": int(self.cfg.max_points),
            "x_start_ratio": self.cfg.x_start_ratio,
            "x_end_ratio": self.cfg.x_end_ratio,
            "y_start_ratio": self.cfg.y_start_ratio,
            "y_end_ratio": self.cfg.y_end_ratio,
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[DONE] sampled {len(points_xy)} query points to {out_dir}")
        return out_dir / "bridge_queries.npy"

    def run_cotracker(self, frames_dir: Path, queries_path: Path, out_dir: Path) -> Path:
        ensure_dir(out_dir)
        num_frames = len(list_images(frames_dir))
        cmd = [
            sys.executable,
            str(self.root / "scripts" / "run_cotracker_bridge.py"),
            "--frames_dir", str(frames_dir),
            "--queries", str(queries_path),
            "--out_dir", str(out_dir),
            "--device", self.cfg.device,
            "--max_frames", str(num_frames),
            "--save_video",
            "--fps", str(self.cfg.fps),
            "--tail_len", str(self.cfg.tail_len),
        ]
        run_cmd(cmd, cwd=self.root)
        return out_dir / "tracks_overlay.mp4"

    def run_magnify(self, frames_dir: Path, out_dir: Path) -> Tuple[Path, Path, Path]:
        ensure_dir(out_dir)
        out_video = out_dir / f"magnified_x{self.cfg.amp:g}.mp4"
        mag_frames = out_dir / "frames"
        cmd = [
            sys.executable,
            str(self.root / "scripts" / "magnify_bridge_video_fd4mm.py"),
            "--fd4mm_root", self.cfg.fd4mm_root,
            "--checkpoint", self.cfg.fd4mm_ckpt,
            "--frames_dir", str(frames_dir),
            "--out_video", str(out_video),
            "--amp", str(self.cfg.amp),
            "--fps", str(self.cfg.fps),
            "--side_by_side",
            "--save_mag_frames_dir", str(mag_frames),
            "--device", self.cfg.device,
        ]
        run_cmd(cmd, cwd=self.root)

        compare_video = out_dir / "orig_vs_magnified.mp4"
        orig_video = self.results_root / "orig" / "interval.mp4"
        self.build_side_by_side_video(orig_video, out_video, compare_video, "Original", f"Magnified x{self.cfg.amp:g}")
        return out_video, mag_frames, compare_video

    def compute_displacement(self, tracks_path: Path, vis_path: Path, out_dir: Path, label: str):
        ensure_dir(out_dir)
        tracks = np.load(tracks_path).astype(np.float32)  # [T,N,2]
        vis = np.load(vis_path).astype(np.float32)
        if vis.ndim == 3:
            vis = vis[..., 0]
        disp = tracks[:, :, self.cfg.axis] - tracks[0:1, :, self.cfg.axis]
        disp[vis < 0.5] = np.nan

        curve_med = np.nanmedian(disp, axis=1)
        curve_mean = np.nanmean(disp, axis=1)

        np.save(out_dir / "disp_per_point.npy", disp)
        np.save(out_dir / "displacement_curve_median.npy", curve_med)
        np.save(out_dir / "displacement_curve_mean.npy", curve_mean)

        with open(out_dir / "displacement_curve.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "median", "mean"])
            for i, (m1, m2) in enumerate(zip(curve_med, curve_mean)):
                writer.writerow([i, float(m1) if np.isfinite(m1) else "nan", float(m2) if np.isfinite(m2) else "nan"])

        # all points + aggregate
        plt.figure(figsize=(11, 4))
        for i in range(disp.shape[1]):
            plt.plot(disp[:, i], color="gray", alpha=0.18, linewidth=1)
        plt.plot(curve_med, color="red", linewidth=2, label="median")
        plt.plot(curve_mean, color="blue", linewidth=1.5, label="mean")
        plt.xlabel("Frame")
        plt.ylabel("Displacement (pixels)")
        plt.title(f"Displacement: {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "displacement_all_points.png", dpi=150)
        plt.close()

        plt.figure(figsize=(11, 4))
        plt.plot(curve_med, color="red", linewidth=2, label="median")
        plt.plot(curve_mean, color="blue", linewidth=1.5, label="mean")
        plt.xlabel("Frame")
        plt.ylabel("Displacement (pixels)")
        plt.title(f"Aggregate displacement: {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "displacement_curve.png", dpi=150)
        plt.close()

        freqs = np.fft.rfftfreq(len(curve_med), d=1.0 / self.cfg.fps)
        curve_fft = np.abs(np.fft.rfft(np.nan_to_num(curve_med - np.nanmean(curve_med))))
        np.save(out_dir / "fft_freqs.npy", freqs)
        np.save(out_dir / "fft_spec.npy", curve_fft)
        plt.figure(figsize=(11, 4))
        plt.plot(freqs, curve_fft, linewidth=1.5)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title(f"FFT: {label}")
        plt.tight_layout()
        plt.savefig(out_dir / "displacement_fft.png", dpi=150)
        plt.close()

        amp = np.nanmax(disp, axis=0) - np.nanmin(disp, axis=0)
        plt.figure(figsize=(6, 4))
        plt.hist(amp[np.isfinite(amp)], bins=30)
        plt.xlabel("Per-point amplitude (pixels)")
        plt.ylabel("Count")
        plt.title(f"Amplitude histogram: {label}")
        plt.tight_layout()
        plt.savefig(out_dir / "amplitude_hist.png", dpi=150)
        plt.close()

        with open(out_dir / "meta.json", "w") as f:
            json.dump({
                "tracks": str(tracks_path),
                "visibility": str(vis_path),
                "axis": self.cfg.axis,
                "fps": self.cfg.fps,
                "label": label,
            }, f, indent=2)

    def compare_displacements(self, orig_dir: Path, mag_dir: Path, out_dir: Path):
        ensure_dir(out_dir)
        curve1 = np.load(orig_dir / "displacement_curve_median.npy")
        curve2 = np.load(mag_dir / "displacement_curve_median.npy")
        T = min(len(curve1), len(curve2))
        curve1 = curve1[:T]
        curve2 = curve2[:T]
        curve2_scaled = curve2 / self.cfg.amp

        plt.figure(figsize=(11, 4))
        plt.plot(curve1, linewidth=2, label="Original median")
        plt.plot(curve2, linewidth=2, label=f"Magnified median x{self.cfg.amp:g}")
        plt.plot(curve2_scaled, "--", linewidth=2, label=f"Magnified / {self.cfg.amp:g}")
        plt.xlabel("Frame")
        plt.ylabel("Displacement (pixels)")
        plt.title("Original vs magnified displacement")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "compare_aggregate.png", dpi=150)
        plt.close()

        np.save(out_dir / "curve_orig.npy", curve1)
        np.save(out_dir / "curve_mag.npy", curve2)
        np.save(out_dir / "curve_mag_scaled.npy", curve2_scaled)

    def archive_results(self):
        archive_path = self.results_root.with_suffix(".tar.gz")
        parent = self.results_root.parent
        name = self.results_root.name
        run_cmd(["tar", "-czvf", str(archive_path), name], cwd=parent)
        print(f"[DONE] archived results to {archive_path}")

    def run(self):
        # A. original interval
        orig_frames = self.extract_interval_frames()
        self.build_video_from_frames(orig_frames, self.results_root / "orig" / "interval.mp4", self.cfg.fps, label="Original interval")

        # B. SAM2 on original
        orig_sam_dir = self.results_root / "orig" / "sam2"
        orig_overlay_video = self.run_sam2(orig_frames, orig_sam_dir, self.cfg.orig_box)

        # C. sample points and track original
        orig_queries_dir = self.results_root / "orig" / "queries"
        orig_queries = self.sample_points_from_mask(orig_sam_dir / "masks" / "000000.png", orig_queries_dir)
        orig_track_dir = self.results_root / "orig" / "tracking"
        orig_tracking_video = self.run_cotracker(orig_frames, orig_queries, orig_track_dir)
        self.compute_displacement(orig_track_dir / "tracks.npy", orig_track_dir / "visibility.npy", self.results_root / "orig" / "displacement", label="Original")

        # D. magnify original interval
        mag_dir = self.results_root / "magnified"
        mag_video, mag_frames, orig_vs_mag_video = self.run_magnify(orig_frames, mag_dir)

        # E. SAM2 on magnified frames
        mag_sam_dir = self.results_root / "magnified" / "sam2"
        mag_overlay_video = self.run_sam2(mag_frames, mag_sam_dir, self.cfg.mag_box)
        self.build_side_by_side_video(orig_overlay_video, mag_overlay_video, self.results_root / "comparisons" / "sam2_overlay_compare.mp4", "Original overlay", "Magnified overlay")

        # F. sample points and track magnified
        mag_queries_dir = self.results_root / "magnified" / "queries"
        mag_queries = self.sample_points_from_mask(mag_sam_dir / "masks" / "000000.png", mag_queries_dir)
        mag_track_dir = self.results_root / "magnified" / "tracking"
        mag_tracking_video = self.run_cotracker(mag_frames, mag_queries, mag_track_dir)
        self.compute_displacement(mag_track_dir / "tracks.npy", mag_track_dir / "visibility.npy", self.results_root / "magnified" / "displacement", label=f"Magnified x{self.cfg.amp:g}")

        # G. comparisons
        ensure_dir(self.results_root / "comparisons")
        self.build_side_by_side_video(orig_tracking_video, mag_tracking_video, self.results_root / "comparisons" / "tracking_compare.mp4", "Original tracking", "Magnified tracking")
        self.compare_displacements(self.results_root / "orig" / "displacement", self.results_root / "magnified" / "displacement", self.results_root / "comparisons")

        # H. pack
        self.archive_results()


def parse_box(vals: Sequence[str]) -> Tuple[int, int, int, int]:
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("Box needs 4 ints: x1 y1 x2 y2")
    return tuple(map(int, vals))  # type: ignore


def build_parser():
    p = argparse.ArgumentParser(description="One-script master pipeline for original/magnified SAM2 + CoTracker + displacement analysis")
    p.add_argument("--project_root", default=".")
    p.add_argument("--video_path", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--start_frame", type=int, required=True)
    p.add_argument("--end_frame", type=int, required=True)
    p.add_argument("--fps", type=float, required=True)

    p.add_argument("--orig_box", nargs=4, required=True)
    p.add_argument("--mag_box", nargs=4, required=True)

    p.add_argument("--amp", type=float, default=10.0)
    p.add_argument("--max_points", type=int, default=100)
    p.add_argument("--grid_spacing", type=int, default=20)
    p.add_argument("--x_start_ratio", type=float, default=0.30)
    p.add_argument("--x_end_ratio", type=float, default=0.70)
    p.add_argument("--y_start_ratio", type=float, default=0.35)
    p.add_argument("--y_end_ratio", type=float, default=0.65)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    p.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    p.add_argument("--sam2_ckpt", default="models/sam2.1_hiera_small.pt")
    p.add_argument("--fd4mm_root", default="external/fd4mm")
    p.add_argument("--fd4mm_ckpt", default="results/stage2_train_walk_amp10_e5_200_epoch_baseline_20_epoch/best_weights.pth")
    p.add_argument("--tail_len", type=int, default=15)
    p.add_argument("--axis", type=int, default=1, choices=[0, 1], help="0=x, 1=y displacement")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = PipelineConfig(
        project_root=args.project_root,
        video_path=args.video_path,
        run_name=args.run_name,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        fps=args.fps,
        orig_box=tuple(map(int, args.orig_box)),
        mag_box=tuple(map(int, args.mag_box)),
        amp=args.amp,
        max_points=args.max_points,
        grid_spacing=args.grid_spacing,
        x_start_ratio=args.x_start_ratio,
        x_end_ratio=args.x_end_ratio,
        y_start_ratio=args.y_start_ratio,
        y_end_ratio=args.y_end_ratio,
        device=args.device,
        sam2_cfg=args.sam2_cfg,
        sam2_ckpt=args.sam2_ckpt,
        fd4mm_root=args.fd4mm_root,
        fd4mm_ckpt=args.fd4mm_ckpt,
        tail_len=args.tail_len,
        axis=args.axis,
    )

    pipe = MasterPipeline(cfg)
    pipe.run()


if __name__ == "__main__":
    main()
