#!/usr/bin/env python3
import re
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path):
    s = str(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(folder):
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS]
    files = sorted(files, key=natural_key)
    if not files:
        raise RuntimeError(f"No images found in {folder}")
    return files


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_video_tensor(frame_paths, max_frames=None):
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    frames = []
    for i, p in enumerate(frame_paths):
        img = read_rgb(p)
        frames.append(img)
        if (i + 1) % 200 == 0:
            print(f"[INFO] loaded {i+1}/{len(frame_paths)} frames")

    arr = np.stack(frames, axis=0)  # T,H,W,3
    video = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).float()  # B,T,C,H,W
    return video


def draw_tracks_on_first_frame(first_frame_bgr, queries_xy, out_path):
    vis = first_frame_bgr.copy()
    for x, y in queries_xy:
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
    cv2.imwrite(str(out_path), vis)

def draw_tracks_frame(frame_bgr, tracks_t, vis_t=None, tail=None):
    """
    frame_bgr: H,W,3
    tracks_t: [N,2] current frame points
    vis_t: [N] or [N,1]
    tail: optional [L,N,2], previous track history including current frame
    """
    out = frame_bgr.copy()

    if vis_t is not None and vis_t.ndim == 2:
        vis_t = vis_t[:, 0]

    N = tracks_t.shape[0]

    # draw tails first
    if tail is not None:
        L = tail.shape[0]
        for i in range(N):
            if vis_t is not None and vis_t[i] < 0.5:
                continue
            for k in range(1, L):
                p0 = tail[k - 1, i]
                p1 = tail[k, i]
                x0, y0 = int(round(p0[0])), int(round(p0[1]))
                x1, y1 = int(round(p1[0])), int(round(p1[1]))
                cv2.line(out, (x0, y0), (x1, y1), (0, 255, 0), 1)

    # draw current points
    for i in range(N):
        if vis_t is not None and vis_t[i] < 0.5:
            continue
        x, y = tracks_t[i]
        cv2.circle(out, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)

    return out


def save_tracking_video(frame_paths, tracks, visibility, out_video, fps=50.0, tail_len=10):
    """
    frame_paths: list of image paths, length T
    tracks: [T,N,2]
    visibility: [T,N] or [T,N,1]
    """
    T = min(len(frame_paths), tracks.shape[0])

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
    H, W = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_video}")

    for t in range(T):
        frame = cv2.imread(str(frame_paths[t]), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_paths[t]}")

        start = max(0, t - tail_len + 1)
        tail = tracks[start:t + 1] if tail_len > 1 else None

        vis_t = visibility[t]
        vis_t = vis_t[:, 0] if vis_t.ndim == 2 else vis_t

        vis_frame = draw_tracks_frame(
            frame_bgr=frame,
            tracks_t=tracks[t],
            vis_t=vis_t,
            tail=tail
        )
        writer.write(vis_frame)

        if (t + 1) % 100 == 0 or (t + 1) == T:
            print(f"[INFO] wrote {t+1}/{T} video frames")

    writer.release()

def main():
    parser = argparse.ArgumentParser(description="Run CoTracker3 offline on bridge query points.")
    parser.add_argument("--frames_dir", required=True, help="Directory of original frames")
    parser.add_argument("--queries", required=True, help="Path to bridge_queries.npy, shape [N,3]")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_frames", type=int, default=None, help="Optional limit for quick testing")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional local checkpoint path. If omitted, torch.hub pretrained model is used.")
    parser.add_argument("--save_video", action="store_true",
                        help="Save tracking overlay video")
    parser.add_argument("--fps", type=float, default=50.0,
                        help="FPS for saved tracking video")
    parser.add_argument("--tail_len", type=int, default=10,
                        help="Length of trajectory tail to draw")
    parser.add_argument("--video_name", default="tracks_overlay.mp4",
                        help="Output video filename inside out_dir")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    frame_paths = list_images(args.frames_dir)
    if args.max_frames is not None:
        frame_paths = frame_paths[:args.max_frames]

    queries = np.load(args.queries)  # [N,3] = [frame_idx, x, y]
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise RuntimeError(f"Expected queries shape [N,3], got {queries.shape}")

    # If max_frames is used, make sure query frame is within range
    if args.max_frames is not None:
        valid = queries[:, 0] < args.max_frames
        queries = queries[valid]
        if len(queries) == 0:
            raise RuntimeError("All queries were removed by --max_frames")

    video = load_video_tensor(frame_paths, max_frames=None)
    H, W = video.shape[-2], video.shape[-1]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    video = video.to(device)
    queries_t = torch.from_numpy(queries).float().unsqueeze(0).to(device)  # [1,N,3]

    print("[INFO] loading CoTracker3 offline model...")
    if args.checkpoint is None:
        # Official quick start uses torch.hub with cotracker3_offline
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    else:
        # Keep this branch only if you later decide to use a local checkpoint
        # For now, the torch.hub route is the most robust official entrypoint.
        raise NotImplementedError("Local checkpoint loading not implemented in this minimal script.")

    model.eval()

    print(f"[INFO] video shape: {tuple(video.shape)}")       # B,T,C,H,W
    print(f"[INFO] queries shape: {tuple(queries_t.shape)}") # B,N,3
    print("[INFO] running tracker...")

    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, queries=queries_t)

    # Official output shapes: B,T,N,2 and B,T,N,1
    tracks = pred_tracks[0].detach().cpu().numpy().astype(np.float32)         # T,N,2
    visibility = pred_visibility[0].detach().cpu().numpy().astype(np.float32) # T,N,1

    np.save(Path(args.out_dir) / "tracks.npy", tracks)
    np.save(Path(args.out_dir) / "visibility.npy", visibility)
    np.save(Path(args.out_dir) / "queries_used.npy", queries.astype(np.float32))
    
    if args.save_video:
        out_video = Path(args.out_dir) / args.video_name
        print(f"[INFO] saving tracking video to: {out_video}")
        save_tracking_video(
            frame_paths=frame_paths,
            tracks=tracks,
            visibility=visibility,
            out_video=out_video,
            fps=args.fps,
            tail_len=args.tail_len,
        )

    # simple preview on first frame
    first_bgr = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    draw_tracks_on_first_frame(first_bgr, queries[:, 1:3], Path(args.out_dir) / "queries_on_first_frame.png")

    meta = {
        "frames_dir": str(Path(args.frames_dir).resolve()),
        "queries": str(Path(args.queries).resolve()),
        "num_frames": int(video.shape[1]),
        "num_queries": int(queries.shape[0]),
        "image_size_hw": [int(H), int(W)],
        "device": device,
        "model": "cotracker3_offline via torch.hub",
    }
    with open(Path(args.out_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] tracks saved to {args.out_dir}")
    print(f"[DONE] tracks.npy shape: {tracks.shape}")
    print(f"[DONE] visibility.npy shape: {visibility.shape}")


if __name__ == "__main__":
    main()