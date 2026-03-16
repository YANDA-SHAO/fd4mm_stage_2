#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    return (m > 127).astype(np.uint8)


def keep_middle_stripe(mask, x_start_ratio=0.25, x_end_ratio=0.75, y_shrink_ratio=0.15):
    """
    Keep only a stable middle stripe of the bridge mask.
    This is more robust than using the full bridge length.

    x_start_ratio, x_end_ratio:
        keep only the middle portion along x, e.g. 25% ~ 75%

    y_shrink_ratio:
        shrink the top/bottom a bit to avoid unstable boundaries
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w = x_max - x_min + 1
    h = y_max - y_min + 1

    sx1 = int(round(x_min + x_start_ratio * w))
    sx2 = int(round(x_min + x_end_ratio * w))

    sy1 = int(round(y_min + y_shrink_ratio * h))
    sy2 = int(round(y_max - y_shrink_ratio * h))

    sx1 = max(0, min(mask.shape[1] - 1, sx1))
    sx2 = max(0, min(mask.shape[1], sx2))
    sy1 = max(0, min(mask.shape[0] - 1, sy1))
    sy2 = max(0, min(mask.shape[0], sy2))

    stripe = np.zeros_like(mask, dtype=np.uint8)
    stripe[sy1:sy2, sx1:sx2] = 1

    out = (mask & stripe).astype(np.uint8)
    return out


def sample_grid_points_from_mask(mask, grid_spacing=16, max_points=100, border_margin=4):
    """
    Sample grid points inside a binary mask.
    Returns integer pixel coordinates as Nx2 array: [x, y]
    """
    h, w = mask.shape
    pts = []

    for y in range(border_margin, h - border_margin, grid_spacing):
        for x in range(border_margin, w - border_margin, grid_spacing):
            if mask[y, x] > 0:
                pts.append([x, y])

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    pts = np.array(pts, dtype=np.int32)

    if len(pts) > max_points:
        # uniform subsampling
        idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int32)
        pts = pts[idx]

    return pts


def build_queries(points_xy, query_frame=0):
    """
    CoTracker query format: [frame_idx, x, y]
    Output shape: [N, 3]
    """
    if len(points_xy) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    qf = np.full((len(points_xy), 1), query_frame, dtype=np.float32)
    qxy = points_xy.astype(np.float32)
    queries = np.concatenate([qf, qxy], axis=1)
    return queries


def draw_points_overlay(mask, points_xy, out_path):
    vis = np.stack([mask * 255, mask * 255, mask * 255], axis=-1).astype(np.uint8)
    for x, y in points_xy:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv2.imwrite(str(out_path), vis)


def main():
    parser = argparse.ArgumentParser(description="Sample stable bridge query points from SAM2 mask.")
    parser.add_argument("--mask", required=True, help="Path to first-frame bridge mask")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--grid_spacing", type=int, default=16, help="Grid spacing in pixels")
    parser.add_argument("--max_points", type=int, default=100, help="Maximum number of bridge points")
    parser.add_argument("--query_frame", type=int, default=0, help="CoTracker query frame index")
    parser.add_argument("--x_start_ratio", type=float, default=0.25, help="Middle stripe start ratio")
    parser.add_argument("--x_end_ratio", type=float, default=0.75, help="Middle stripe end ratio")
    parser.add_argument("--y_shrink_ratio", type=float, default=0.15, help="Top/bottom shrink ratio")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    mask = read_mask(args.mask)
    stripe_mask = keep_middle_stripe(
        mask,
        x_start_ratio=args.x_start_ratio,
        x_end_ratio=args.x_end_ratio,
        y_shrink_ratio=args.y_shrink_ratio,
    )

    points_xy = sample_grid_points_from_mask(
        stripe_mask,
        grid_spacing=args.grid_spacing,
        max_points=args.max_points,
        border_margin=4,
    )

    if len(points_xy) == 0:
        raise RuntimeError("No valid bridge points sampled. Try reducing shrink ratios or grid spacing.")

    queries = build_queries(points_xy, query_frame=args.query_frame)

    np.save(out_dir / "bridge_points_xy.npy", points_xy)
    np.save(out_dir / "bridge_queries.npy", queries)

    draw_points_overlay(mask, points_xy, out_dir / "bridge_points_on_mask.png")
    draw_points_overlay(stripe_mask, points_xy, out_dir / "bridge_points_on_stripe_mask.png")

    meta = {
        "mask": str(Path(args.mask).resolve()),
        "num_points": int(len(points_xy)),
        "grid_spacing": args.grid_spacing,
        "max_points": args.max_points,
        "query_frame": args.query_frame,
        "x_start_ratio": args.x_start_ratio,
        "x_end_ratio": args.x_end_ratio,
        "y_shrink_ratio": args.y_shrink_ratio,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] sampled {len(points_xy)} bridge points")
    print(f"[DONE] saved to {out_dir}")


if __name__ == "__main__":
    main()