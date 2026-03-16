#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse

import cv2
import torch
import numpy as np


def add_fd4mm_to_path(fd4mm_root):
    fd4mm_root = str(Path(fd4mm_root).resolve())
    if fd4mm_root not in sys.path:
        sys.path.insert(0, fd4mm_root)


def list_frames(frames_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in Path(frames_dir).iterdir() if p.suffix.lower() in exts]
    files = sorted(files, key=lambda p: p.name)
    if not files:
        raise RuntimeError(f"No frames found in {frames_dir}")
    return files


def img_to_tensor(img_bgr, device):
    """
    BGR uint8 [H,W,3] -> torch [1,3,H,W] in [-1,1]
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    x = x * 2.0 - 1.0
    return x.to(device)


def tensor_to_img(x):
    """
    torch [1,3,H,W] or [3,H,W] in [-1,1] -> RGB uint8 [H,W,3]
    """
    x = x.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255.0).astype(np.uint8)
    return x


def put_text(img_rgb, text):
    img = img_rgb.copy()
    cv2.putText(
        img,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def load_model(fd4mm_root, checkpoint_path, device):
    add_fd4mm_to_path(fd4mm_root)

    from magnet_FD4MM import MagNet
    from callbacks import gen_state_dict

    state_dict = gen_state_dict(checkpoint_path)

    model = MagNet().to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"[INFO] Loaded model from: {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd4mm_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--out_video", type=str, required=True)
    parser.add_argument("--amp", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--side_by_side", action="store_true",
                        help="Save original and magnified side by side")
    parser.add_argument("--save_mag_frames_dir", type=str, default=None,
                        help="Optional folder to save magnified frames")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    frames = list_frames(args.frames_dir)
    print(f"[INFO] Number of frames: {len(frames)}")

    model = load_model(args.fd4mm_root, args.checkpoint, device)

    # read first frame as reference
    ref_bgr = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    H, W = ref_bgr.shape[:2]

    ref_tensor = img_to_tensor(ref_bgr, device)

    amp = torch.tensor([args.amp], device=device).view(1, 1, 1, 1)

    if args.side_by_side:
        writer_size = (W * 2, H)
    else:
        writer_size = (W, H)

    out_video_path = Path(args.out_video)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        writer_size
    )

    if args.save_mag_frames_dir is not None:
        save_dir = Path(args.save_mag_frames_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    for i, frame_path in enumerate(frames):
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] skip unreadable frame: {frame_path}")
            continue

        tgt_tensor = img_to_tensor(bgr, device)

        if i == 0:
            # keep first frame unchanged
            mag_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            with torch.no_grad():
                pred = model(ref_tensor, tgt_tensor, amp, mode="evaluate")
            mag_rgb = tensor_to_img(pred)

        orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if args.side_by_side:
            left = put_text(orig_rgb, "Original")
            right = put_text(mag_rgb, f"Magnified x{args.amp:g}")
            frame_rgb = np.concatenate([left, right], axis=1)
        else:
            frame_rgb = put_text(mag_rgb, f"Magnified x{args.amp:g}")

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        if save_dir is not None:
            cv2.imwrite(str(save_dir / f"{i:06d}.jpg"), cv2.cvtColor(mag_rgb, cv2.COLOR_RGB2BGR))

        if (i + 1) % 100 == 0:
            print(f"[INFO] processed {i+1}/{len(frames)} frames")

    writer.release()
    print(f"[DONE] Saved video to: {out_video_path}")


if __name__ == "__main__":
    main()