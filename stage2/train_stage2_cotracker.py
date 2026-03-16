#!/usr/bin/env python3
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# helpers
# =========================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def add_fd4mm_to_path(fd4mm_root: str):
    fd4mm_root = str(Path(fd4mm_root).resolve())
    if fd4mm_root not in sys.path:
        sys.path.insert(0, fd4mm_root)


def add_cotracker_to_path(cotracker_root: str):
    cotracker_root = str(Path(cotracker_root).resolve())
    if cotracker_root not in sys.path:
        sys.path.insert(0, cotracker_root)


def list_images(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    files = sorted(files, key=lambda p: p.name)
    if not files:
        raise RuntimeError(f"No images found in {folder}")
    return files


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb_to_fd4mm_tensor(img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    RGB uint8 [H,W,3] -> [3,H,W] float in [-1,1]
    """
    x = torch.from_numpy(img_rgb).float() / 255.0
    x = x.permute(2, 0, 1)  # [3,H,W]
    x = x * 2.0 - 1.0
    return x.to(device)


def fd4mm_to_tracker_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    [-1,1] -> [0,255] float
    Input: [B,T,3,H,W] or [T,3,H,W]
    Output same shape.
    """
    y = (x + 1.0) / 2.0
    y = y.clamp(0.0, 1.0) * 255.0
    return y


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-6):
    mask = mask.float()
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(eps)
    return num / den


def safe_std(x: torch.Tensor, dim=None, keepdim=False, eps: float = 1e-6):
    mean = x.mean(dim=dim, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dim, keepdim=keepdim)
    return torch.sqrt(var + eps)


def make_roi_fd4mm_safe(roi: Tuple[int, int, int, int], divisor: int = 16) -> Tuple[int, int, int, int]:
    """
    Center-crop ROI inward so width/height are multiples of divisor.
    """
    x1, y1, x2, y2 = roi
    w = x2 - x1
    h = y2 - y1

    w2 = (w // divisor) * divisor
    h2 = (h // divisor) * divisor

    if w2 < divisor or h2 < divisor:
        raise ValueError(f"ROI too small after divisor={divisor} adjustment: {roi}")

    dx = (w - w2) // 2
    dy = (h - h2) // 2

    nx1 = x1 + dx
    ny1 = y1 + dy
    nx2 = nx1 + w2
    ny2 = ny1 + h2
    return nx1, ny1, nx2, ny2


# =========================
# data
# =========================
class WalkClipDataset:
    """
    Use one mother interval from full frames_dir.
    We assume:
    - full frames exist in data/video1/frames_jpg
    - subtle motion, so fixed query coordinates are acceptable for each clip start
    - ROI is fixed
    """

    def __init__(
        self,
        frames_dir: str,
        start_frame: int,
        end_frame: int,
        clip_len: int,
        stride: int,
        roi: Tuple[int, int, int, int],
        device: torch.device,
        queries_path: Optional[str] = None,
        max_points: int = 64,
        grid_spacing: int = 20,
    ):
        self.frames_dir = Path(frames_dir)
        self.device = device
        self.clip_len = clip_len
        self.stride = stride
        self.roi = make_roi_fd4mm_safe(roi)  # safe ROI
        self.raw_roi = roi

        all_frames = list_images(self.frames_dir)
        if end_frame >= len(all_frames):
            raise ValueError(f"end_frame={end_frame} >= total frames={len(all_frames)}")

        self.interval_frames = all_frames[start_frame:end_frame + 1]
        self.num_interval_frames = len(self.interval_frames)
        if self.num_interval_frames < clip_len:
            raise ValueError("Interval shorter than clip_len")

        self.starts = list(range(0, self.num_interval_frames - clip_len + 1, stride))
        if not self.starts:
            raise RuntimeError("No valid clip starts")

        self.queries_xy = self._load_or_make_queries(
            queries_path=queries_path,
            roi=self.roi,
            max_points=max_points,
            grid_spacing=grid_spacing,
        )

    def _load_or_make_queries(
        self,
        queries_path: Optional[str],
        roi: Tuple[int, int, int, int],
        max_points: int,
        grid_spacing: int,
    ) -> np.ndarray:
        x1, y1, x2, y2 = roi

        if queries_path is not None and Path(queries_path).exists():
            q = np.load(queries_path).astype(np.float32)  # [N,3] = [frame_idx,x,y]
            if q.ndim != 2 or q.shape[1] != 3:
                raise RuntimeError(f"Bad queries shape: {q.shape}")

            xy = q[:, 1:3]
            keep = (
                (xy[:, 0] >= x1) & (xy[:, 0] < x2) &
                (xy[:, 1] >= y1) & (xy[:, 1] < y2)
            )
            xy = xy[keep]
            xy[:, 0] -= x1
            xy[:, 1] -= y1

            if len(xy) == 0:
                raise RuntimeError("No query points remain inside ROI after filtering")
            if len(xy) > max_points:
                idx = np.linspace(0, len(xy) - 1, max_points).round().astype(int)
                xy = xy[idx]

            return xy

        # fallback: make a fixed grid in ROI
        pts = []
        w = x2 - x1
        h = y2 - y1
        for yy in range(0, h, grid_spacing):
            for xx in range(0, w, grid_spacing):
                pts.append((xx, yy))
        if not pts:
            raise RuntimeError("Fallback grid sampling produced no points")

        if len(pts) > max_points:
            idx = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
            pts = [pts[i] for i in idx]

        return np.array(pts, dtype=np.float32)

    def __len__(self):
        return len(self.starts)

    def get_clip(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            clip_fd4mm: [1,T,3,H,W] in [-1,1]
            queries:    [1,N,3] where [frame_idx=0, x, y]
        """
        start = self.starts[idx]
        selected = self.interval_frames[start:start + self.clip_len]

        x1, y1, x2, y2 = self.roi
        frames = []
        for p in selected:
            img = read_rgb(p)
            crop = img[y1:y2, x1:x2]
            frames.append(rgb_to_fd4mm_tensor(crop, self.device))

        clip = torch.stack(frames, dim=0).unsqueeze(0)  # [1,T,3,H,W]

        q = np.concatenate([
            np.zeros((len(self.queries_xy), 1), dtype=np.float32),
            self.queries_xy
        ], axis=1)
        queries = torch.from_numpy(q).float().unsqueeze(0).to(self.device)  # [1,N,3]

        return clip, queries


# =========================
# FD4MM
# =========================
def build_fd4mm_model(fd4mm_root: str, ckpt_path: str, device: torch.device) -> nn.Module:
    add_fd4mm_to_path(fd4mm_root)

    from magnet_FD4MM import MagNet
    from callbacks import gen_state_dict

    model = MagNet().to(device)
    state_dict = gen_state_dict(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)
    model.train()
    print(f"[INFO] loaded FD4MM from: {ckpt_path}")
    return model


def magnify_clip_fd4mm(model: nn.Module, clip: torch.Tensor, amp_value: float) -> torch.Tensor:
    """
    clip: [B,T,3,H,W] in [-1,1]
    return same shape
    """
    if clip.ndim != 5:
        raise ValueError(f"Expected [B,T,3,H,W], got {clip.shape}")

    B, T, C, H, W = clip.shape
    amp = torch.tensor([amp_value] * B, device=clip.device, dtype=clip.dtype)

    ref = clip[:, 0]
    out = [ref]
    for t in range(1, T):
        tgt = clip[:, t]
        pred = model(ref, tgt, amp, mode="evaluate")
        out.append(pred)

    return torch.stack(out, dim=1)


# =========================
# CoTracker measurement (official training-model path)
# =========================
class CoTrackerMeasurement(nn.Module):
    """
    Use the raw CoTracker model, not the predictor wrapper.

    Predictor path in official repo:
      CoTrackerPredictor.forward is decorated with @torch.no_grad()
    so it is unsuitable for gradient-based optimization through the input video.

    Here we:
    - build the raw model via build_cotracker(...)
    - freeze its parameters
    - keep the forward path differentiable with respect to input video
    - manually reproduce resize/query-rescale logic like predictor.py
    """

    def __init__(
        self,
        cotracker_root: str,
        checkpoint_path: str,
        device: torch.device,
        offline: bool = True,
        window_len: int = 16,
        model_resolution: Tuple[int, int] = (384, 512),
        iters: int = 4,
    ):
        super().__init__()
        add_cotracker_to_path(cotracker_root)

        from cotracker.models.build_cotracker import build_cotracker

        print("[INFO] loading CoTracker raw model via build_cotracker...")
        self.model = build_cotracker(
            checkpoint=checkpoint_path,
            offline=offline,
            window_len=window_len,
        ).to(device).eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.model_resolution = model_resolution
        self.iters = iters

    def _resize_video_and_queries(
        self,
        video_255: torch.Tensor,
        queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Resize video to tracker resolution and scale queries accordingly.
        """
        B, T, C, H, W = video_255.shape
        target_h, target_w = self.model_resolution

        if H == target_h and W == target_w:
            return video_255, queries, (H, W)

        video_rs = F.interpolate(
            video_255.reshape(B * T, C, H, W),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=True,
        ).reshape(B, T, C, target_h, target_w)

        queries_rs = queries.clone()
        if W > 1:
            queries_rs[..., 1] *= (target_w - 1) / (W - 1)
        if H > 1:
            queries_rs[..., 2] *= (target_h - 1) / (H - 1)

        return video_rs, queries_rs, (H, W)

    def _rescale_tracks_back(
        self,
        tracks_rs: torch.Tensor,
        orig_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Rescale predicted tracks from tracker resolution back to original ROI crop size.
        """
        H, W = orig_hw
        target_h, target_w = self.model_resolution

        tracks = tracks_rs.clone()
        if target_w > 1:
            tracks[..., 0] *= (W - 1) / (target_w - 1)
        if target_h > 1:
            tracks[..., 1] *= (H - 1) / (target_h - 1)
        return tracks

    def forward(self, video_255: torch.Tensor, queries: torch.Tensor):
        """
        video_255: [B,T,3,H,W], float 0..255
        queries:   [B,N,3] with [frame_idx, x, y] in original crop coordinates

        returns:
            tracks: [B,T,N,2] in original crop coordinates
            quality: [B,T,N]  = visibility * confidence
        """
        if video_255.ndim != 5:
            raise ValueError(f"Expected [B,T,3,H,W], got {video_255.shape}")
        if queries.ndim != 3 or queries.shape[-1] != 3:
            raise ValueError(f"Expected queries [B,N,3], got {queries.shape}")

        video_rs, queries_rs, orig_hw = self._resize_video_and_queries(video_255, queries)

        # raw model forward from official cotracker3_offline.py:
        # returns coords, vis, confidence, train_data
        tracks_rs, visibility, confidence, _ = self.model(
            video=video_rs,
            queries=queries_rs,
            iters=self.iters,
            is_train=False,
        )

        tracks = self._rescale_tracks_back(tracks_rs, orig_hw)
        quality = visibility * confidence
        return tracks, quality


# =========================
# displacement + loss
# =========================
def tracks_to_displacement(
    tracks: torch.Tensor,
    quality: torch.Tensor,
    axis: int = 1,
    vis_thresh: float = 0.5,
):
    """
    tracks:  [B,T,N,2]
    quality: [B,T,N] or [B,T,N,1]
    returns:
        disp: [B,T,N]
        valid: [B,T,N]
        global_curve: [B,T]
    """
    if quality.ndim == 4:
        quality = quality[..., 0]

    coord = tracks[..., axis]           # [B,T,N]
    disp = coord - coord[:, 0:1, :]     # relative to clip first frame
    valid = (quality >= vis_thresh)

    global_curve = masked_mean(disp, valid, dim=2)  # [B,T]
    return disp, valid, global_curve


def stage2_loss(
    disp_orig: torch.Tensor,
    valid_orig: torch.Tensor,
    global_orig: torch.Tensor,
    disp_mag: torch.Tensor,
    valid_mag: torch.Tensor,
    global_mag: torch.Tensor,
    amp_value: float,
    shape_weight: float = 0.2,
):
    """
    Point + global + shape
    """
    valid = valid_orig & valid_mag

    target_point = amp_value * disp_orig
    point_diff = disp_mag - target_point
    if valid.any():
        l_point = F.smooth_l1_loss(point_diff[valid], torch.zeros_like(point_diff[valid]))
    else:
        l_point = torch.tensor(0.0, device=disp_orig.device)

    target_global = amp_value * global_orig
    l_global = F.smooth_l1_loss(global_mag, target_global)

    # shape consistency on normalized global curves
    go = global_orig - global_orig.mean(dim=1, keepdim=True)
    gm = global_mag - global_mag.mean(dim=1, keepdim=True)

    go = go / safe_std(go, dim=1, keepdim=True)
    gm = gm / safe_std(gm, dim=1, keepdim=True)

    l_shape = F.mse_loss(gm, go)

    total = l_point + l_global + shape_weight * l_shape
    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_point": float(l_point.detach().cpu()),
        "loss_global": float(l_global.detach().cpu()),
        "loss_shape": float(l_shape.detach().cpu()),
    }
    return total, stats


# =========================
# training
# =========================
def grad_smoke_test(
    model_mg: nn.Module,
    model_measure: CoTrackerMeasurement,
    dataset: WalkClipDataset,
    amp_value: float,
    axis: int,
):
    print("[INFO] running gradient smoke test...")
    model_mg.zero_grad(set_to_none=True)

    clip, queries = dataset.get_clip(0)
    clip.requires_grad_(False)

    mag_clip = magnify_clip_fd4mm(model_mg, clip, amp_value=amp_value)
    mag_video = fd4mm_to_tracker_tensor(mag_clip)

    tracks_mag, quality_mag = model_measure(mag_video, queries)
    print(f"[INFO] tracks_mag.requires_grad = {tracks_mag.requires_grad}")
    print(f"[INFO] quality_mag.requires_grad = {quality_mag.requires_grad}")

    loss = tracks_mag[..., axis].mean()
    loss.backward()

    has_grad = False
    for p in model_mg.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            has_grad = True
            break

    if not has_grad:
        raise RuntimeError("Gradient smoke test failed: no gradient reached FD4MM parameters")

    print("[PASS] gradient smoke test passed.")


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_loss: float):
    ensure_dir(path.parent)
    train_ckpt = {
        "epoch": epoch,
        "best_loss": best_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(train_ckpt, str(path))
    # 2) raw weights only, same style as original pretrained weights
    weights_only_path = path.with_name(path.stem + "_weights.pth")
    torch.save(model.state_dict(), str(weights_only_path))


def main():
    parser = argparse.ArgumentParser()

    # fixed baseline data
    parser.add_argument("--frames_dir", type=str, default="data/video1/frames_jpg")
    parser.add_argument("--queries_path", type=str, default="data/video1/queries/bridge_queries.npy")
    parser.add_argument("--start_frame", type=int, default=100)
    parser.add_argument("--end_frame", type=int, default=400)

    # ROI = same as before, but will be auto-adjusted to FD4MM-safe size
    parser.add_argument("--roi", nargs=4, type=int, default=[5, 400, 1915, 490])

    # training
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--amp_value", type=float, default=5.0)
    parser.add_argument("--axis", type=int, default=1, choices=[0, 1])
    parser.add_argument("--vis_thresh", type=float, default=0.5)

    # points fallback
    parser.add_argument("--max_points", type=int, default=64)
    parser.add_argument("--grid_spacing", type=int, default=20)

    # model paths
    parser.add_argument("--fd4mm_root", type=str, default="external/fd4mm")
    parser.add_argument(
        "--fd4mm_ckpt",
        type=str,
        default="external/fd4mm/weights_FD4MM_deepmag_exp1_deepmag_only/magnet_epoch6_loss4.02e-01.pth",
    )

    parser.add_argument("--cotracker_root", type=str, default="external/co-tracker")
    parser.add_argument("--cotracker_ckpt", type=str, required=True)
    parser.add_argument("--cotracker_window_len", type=int, default=16)
    parser.add_argument("--cotracker_iters", type=int, default=4)
    parser.add_argument("--cotracker_resolution", nargs=2, type=int, default=[384, 512])

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save_dir", type=str, default="results/stage2_train_walk_100_400_amp5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape_weight", type=float, default=0.2)
    parser.add_argument("--smoke_test_only", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    safe_roi = make_roi_fd4mm_safe(tuple(args.roi))
    print(f"[INFO] raw ROI  = {tuple(args.roi)}")
    print(f"[INFO] safe ROI = {safe_roi}")

    cfg = vars(args).copy()
    cfg["safe_roi"] = list(safe_roi)
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    dataset = WalkClipDataset(
        frames_dir=args.frames_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        clip_len=args.clip_len,
        stride=args.stride,
        roi=tuple(args.roi),
        device=device,
        queries_path=args.queries_path,
        max_points=args.max_points,
        grid_spacing=args.grid_spacing,
    )

    print(f"[INFO] interval clips: {len(dataset)}")
    print(f"[INFO] queries used: {len(dataset.queries_xy)}")

    model_mg = build_fd4mm_model(args.fd4mm_root, args.fd4mm_ckpt, device)
    model_measure = CoTrackerMeasurement(
        cotracker_root=args.cotracker_root,
        checkpoint_path=args.cotracker_ckpt,
        device=device,
        offline=True,
        window_len=args.cotracker_window_len,
        model_resolution=tuple(args.cotracker_resolution),
        iters=args.cotracker_iters,
    )

    # must pass this first
    grad_smoke_test(
        model_mg=model_mg,
        model_measure=model_measure,
        dataset=dataset,
        amp_value=args.amp_value,
        axis=args.axis,
    )

    if args.smoke_test_only:
        print("[DONE] smoke test only.")
        return

    optimizer = torch.optim.AdamW(
        [p for p in model_mg.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_loss = float("inf")
    log_path = save_dir / "train_log.jsonl"

    for epoch in range(args.epochs):
        model_mg.train()

        order = list(range(len(dataset)))
        random.shuffle(order)

        if args.max_steps_per_epoch is not None:
            order = order[:args.max_steps_per_epoch]

        epoch_losses = []

        for step, idx in enumerate(order):
            clip, queries = dataset.get_clip(idx)      # [1,T,3,H,W], [1,N,3]

            # original measurement branch: no need for gradient
            with torch.no_grad():
                video_orig = fd4mm_to_tracker_tensor(clip)
                tracks_orig, quality_orig = model_measure(video_orig, queries)
                disp_orig, valid_orig, global_orig = tracks_to_displacement(
                    tracks_orig, quality_orig, axis=args.axis, vis_thresh=args.vis_thresh
                )

            # magnified branch: must keep gradient to FD4MM
            mag_clip = magnify_clip_fd4mm(model_mg, clip, amp_value=args.amp_value)
            video_mag = fd4mm_to_tracker_tensor(mag_clip)

            tracks_mag, quality_mag = model_measure(video_mag, queries)
            disp_mag, valid_mag, global_mag = tracks_to_displacement(
                tracks_mag, quality_mag, axis=args.axis, vis_thresh=args.vis_thresh
            )

            loss, stats = stage2_loss(
                disp_orig=disp_orig,
                valid_orig=valid_orig,
                global_orig=global_orig,
                disp_mag=disp_mag,
                valid_mag=valid_mag,
                global_mag=global_mag,
                amp_value=args.amp_value,
                shape_weight=args.shape_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_losses.append(stats["loss_total"])

            line = {
                "epoch": epoch,
                "step": step,
                "clip_index": idx,
                **stats,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")

            print(
                f"[epoch {epoch:03d} step {step:03d}] "
                f"loss={stats['loss_total']:.6f} "
                f"point={stats['loss_point']:.6f} "
                f"global={stats['loss_global']:.6f} "
                f"shape={stats['loss_shape']:.6f}"
            )

        mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        print(f"[INFO] epoch {epoch} mean loss = {mean_epoch_loss:.6f}")

        save_checkpoint(save_dir / "last.pth", model_mg, optimizer, epoch, best_loss)

        if mean_epoch_loss < best_loss:
            best_loss = mean_epoch_loss
            save_checkpoint(save_dir / "best.pth", model_mg, optimizer, epoch, best_loss)
            print(f"[INFO] new best checkpoint: {best_loss:.6f}")

    print("[DONE] training finished.")


if __name__ == "__main__":
    main()