#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path):
    s = str(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_frames(frames_dir):
    frames = [p for p in Path(frames_dir).iterdir() if p.suffix.lower() in IMG_EXTS]
    frames = sorted(frames, key=natural_key)
    if not frames:
        raise RuntimeError(f"No frames found in: {frames_dir}")
    return frames


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def keep_largest_component(mask):
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + np.argmax(areas)
    return (labels == largest).astype(np.uint8)


def fill_small_holes(mask, max_hole_area=500):
    """
    Fill background holes inside the object if they are small.
    """
    mask = (mask > 0).astype(np.uint8)
    inv = 1 - mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    h, w = mask.shape
    out = mask.copy()

    for lab in range(1, n):
        x, y, ww, hh, area = stats[lab]
        # ignore components touching the border -> not holes
        touches_border = x == 0 or y == 0 or (x + ww) >= w or (y + hh) >= h
        if not touches_border and area <= max_hole_area:
            out[labels == lab] = 1
    return out


def mask_thin_bridge_prior(mask, min_aspect_ratio=3.0):
    """
    Optional mild prior: if multiple components survive, keep the one most likely
    to be a long thin bridge-like object. Usually largest component is enough,
    but this can help when SAM attaches pedestrians or clutter.
    """
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask

    best_score = -1e18
    best_lab = 1

    for lab in range(1, n):
        x, y, w, h, area = stats[lab]
        if area <= 0:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        density = area / max(w * h, 1)
        # prefer larger and elongated components, but do not over-penalize density
        score = area + 0.15 * area * min(aspect, 20.0) + 500.0 * density
        if score > best_score:
            best_score = score
            best_lab = lab

    return (labels == best_lab).astype(np.uint8)


def postprocess_mask(
    mask,
    open_iter=0,
    close_iter=1,
    erode_iter=0,
    dilate_iter=0,
    largest_only=True,
    fill_holes=True,
    hole_area=500,
    bridge_prior=False,
):
    mask = (mask > 0).astype(np.uint8)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)

    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=open_iter)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=close_iter)
    if erode_iter > 0:
        mask = cv2.erode(mask, kernel3, iterations=erode_iter)
    if dilate_iter > 0:
        mask = cv2.dilate(mask, kernel3, iterations=dilate_iter)

    if fill_holes:
        mask = fill_small_holes(mask, max_hole_area=hole_area)

    if bridge_prior:
        mask = mask_thin_bridge_prior(mask)

    if largest_only:
        mask = keep_largest_component(mask)

    return (mask > 0).astype(np.uint8)


def draw_overlay(img_bgr, mask, box=None, alpha=0.35):
    out = img_bgr.copy()
    color = np.array([0, 255, 0], dtype=np.uint8).reshape(1, 1, 3)
    m3 = np.repeat(mask[:, :, None].astype(bool), 3, axis=2)
    out = np.where(m3, (out * (1 - alpha) + color * alpha).astype(np.uint8), out)

    # draw contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)

    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return out


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment and track a bridge through video frames using SAM2 video predictor."
    )
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory of sequential frames")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--sam2_cfg", type=str, required=True, help="SAM2 config yaml path")
    parser.add_argument("--sam2_ckpt", type=str, required=True, help="SAM2 checkpoint path")

    # prompt
    parser.add_argument(
        "--box", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"),
        help="Initial box prompt on frame 0"
    )

    # runtime
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--obj_id", type=int, default=1)
    parser.add_argument("--score_thresh", type=float, default=0.0, help="Logit threshold, default 0.0")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--save_logits", action="store_true")

    # postprocess
    parser.add_argument("--largest_only", action="store_true")
    parser.add_argument("--bridge_prior", action="store_true")
    parser.add_argument("--open_iter", type=int, default=0)
    parser.add_argument("--close_iter", type=int, default=1)
    parser.add_argument("--erode_iter", type=int, default=0)
    parser.add_argument("--dilate_iter", type=int, default=0)
    parser.add_argument("--hole_area", type=int, default=500)

    # optional debug / memory
    parser.add_argument("--save_every", type=int, default=1, help="Save every Nth frame")
    return parser.parse_args()


def main():
    args = parse_args()

    frames = list_frames(args.frames_dir)
    n_frames = len(frames)

    ensure_dir(args.out_dir)
    mask_dir = Path(args.out_dir) / "masks"
    overlay_dir = Path(args.out_dir) / "overlay"
    logits_dir = Path(args.out_dir) / "logits"
    ensure_dir(mask_dir)
    if args.save_overlay:
        ensure_dir(overlay_dir)
    if args.save_logits:
        ensure_dir(logits_dir)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # build predictor
    predictor = build_sam2_video_predictor(args.sam2_cfg, args.sam2_ckpt, device=device)

    # init state
    state = predictor.init_state(video_path=args.frames_dir)

    # official examples typically use inference_mode + autocast on CUDA
    # for performance; here we keep that behavior. :contentReference[oaicite:1]{index=1}
    box = np.array(args.box, dtype=np.float32)

    frame0_img = read_img(frames[0])
    H, W = frame0_img.shape[:2]

    meta = {
        "frames_dir": str(Path(args.frames_dir).resolve()),
        "out_dir": str(Path(args.out_dir).resolve()),
        "sam2_cfg": args.sam2_cfg,
        "sam2_ckpt": args.sam2_ckpt,
        "num_frames": n_frames,
        "image_size": [H, W],
        "init_box": args.box,
        "obj_id": args.obj_id,
        "score_thresh": args.score_thresh,
        "postprocess": {
            "largest_only": args.largest_only,
            "bridge_prior": args.bridge_prior,
            "open_iter": args.open_iter,
            "close_iter": args.close_iter,
            "erode_iter": args.erode_iter,
            "dilate_iter": args.dilate_iter,
            "hole_area": args.hole_area,
        },
    }

    if device == "cuda":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        class DummyCtx:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        autocast_ctx = DummyCtx()

    saved_frames = 0
    first_return = None

    with torch.inference_mode(), autocast_ctx:
        # add box prompt on frame 0
        # official API supports add_new_points_or_box and then propagate_in_video. :contentReference[oaicite:2]{index=2}
        first_return = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=args.obj_id,
            box=box,
        )

        # save prompt-frame result if available
        if first_return is not None and len(first_return) == 3:
            out_frame_idx, out_obj_ids, out_mask_logits = first_return
            obj_to_idx = {oid: i for i, oid in enumerate(out_obj_ids)}
            if args.obj_id in obj_to_idx:
                idx = obj_to_idx[args.obj_id]
                logit = out_mask_logits[idx]
                if torch.is_tensor(logit):
                    logit = logit.detach().float().cpu().numpy()
                if logit.ndim == 3:
                    logit = logit[0]
                mask = (logit > args.score_thresh).astype(np.uint8)
                mask = postprocess_mask(
                    mask,
                    open_iter=args.open_iter,
                    close_iter=args.close_iter,
                    erode_iter=args.erode_iter,
                    dilate_iter=args.dilate_iter,
                    largest_only=args.largest_only,
                    fill_holes=True,
                    hole_area=args.hole_area,
                    bridge_prior=args.bridge_prior,
                )

                stem = frames[out_frame_idx].stem
                cv2.imwrite(str(mask_dir / f"{stem}.png"), mask * 255)
                if args.save_logits:
                    np.save(str(logits_dir / f"{stem}.npy"), logit.astype(np.float32))
                if args.save_overlay:
                    vis = draw_overlay(frame0_img, mask, box=args.box)
                    cv2.imwrite(str(overlay_dir / f"{stem}.png"), vis)
                saved_frames += 1

        # propagate through video
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            if out_frame_idx % args.save_every != 0:
                continue

            obj_to_idx = {oid: i for i, oid in enumerate(out_obj_ids)}
            if args.obj_id not in obj_to_idx:
                print(f"[WARN] obj_id={args.obj_id} missing at frame {out_frame_idx}")
                continue

            idx = obj_to_idx[args.obj_id]
            logit = out_mask_logits[idx]
            if torch.is_tensor(logit):
                logit = logit.detach().float().cpu().numpy()
            if logit.ndim == 3:
                logit = logit[0]

            mask = (logit > args.score_thresh).astype(np.uint8)
            mask = postprocess_mask(
                mask,
                open_iter=args.open_iter,
                close_iter=args.close_iter,
                erode_iter=args.erode_iter,
                dilate_iter=args.dilate_iter,
                largest_only=args.largest_only,
                fill_holes=True,
                hole_area=args.hole_area,
                bridge_prior=args.bridge_prior,
            )

            stem = frames[out_frame_idx].stem
            cv2.imwrite(str(mask_dir / f"{stem}.png"), mask * 255)

            if args.save_logits:
                np.save(str(logits_dir / f"{stem}.npy"), logit.astype(np.float32))

            if args.save_overlay:
                img = read_img(frames[out_frame_idx])
                vis = draw_overlay(img, mask, box=None)
                cv2.imwrite(str(overlay_dir / f"{stem}.png"), vis)

            saved_frames += 1
            if saved_frames % 50 == 0:
                print(f"[INFO] saved {saved_frames} masks...")

    meta["saved_frames"] = saved_frames
    save_json(meta, Path(args.out_dir) / "meta.json")

    print(f"[DONE] Frames: {n_frames}")
    print(f"[DONE] Saved masks: {saved_frames}")
    print(f"[DONE] Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()