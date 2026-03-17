## How to Run

This section provides a **minimal step-by-step workflow** to run the project. It is recommended to test each component individually before running the full pipeline.

---

### Step 0. Setup

Make sure the external dependencies are correctly placed:

```text
fd4mm_stage_2/
├── external/
│   ├── fd4mm/
│   ├── co-tracker/
│   └── sam2/
```

Required checkpoints:

```text
models/sam2.1_hiera_small.pt
external/fd4mm/<your_fd4mm_checkpoint>.pth
external/co-tracker/<your_cotracker_checkpoint>.pth
```

---

### Step 1. Extract Frames

```bash
python scripts/extract_frames.py \
  --video data/video2/car.mp4 \
  --out results/test/frames
```

Check output:

```text
results/test/frames/*.jpg
```

---

### Step 2. Run SAM2 Segmentation

```bash
python scripts/segment_bridge_sam2.py \
  --frames_dir results/test/frames \
  --out_dir results/test/sam2 \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --sam2_ckpt models/sam2.1_hiera_small.pt \
  --box 5 400 1915 490 \
  --device cuda \
  --save_overlay \
  --largest_only
```

Check:

```text
results/test/sam2/overlay/
```

---

### Step 3. Sample Tracking Points

```bash
python scripts/sample_bridge_points.py \
  --mask_dir results/test/sam2/masks \
  --out_dir results/test/queries \
  --max_points 50
```

Output:

```text
results/test/queries/bridge_queries.npy
```

---

### Step 4. Run CoTracker

```bash
python scripts/run_cotracker_bridge.py \
  --frames_dir results/test/frames \
  --queries results/test/queries/bridge_queries.npy \
  --out_dir results/test/tracking \
  --device cuda \
  --save_video
```

Check:

```text
results/test/tracking/tracking.mp4
```

---

### Step 5. Compute Displacement and FFT

```bash
python scripts/plot_displacement_from_tracks.py \
  --tracks results/test/tracking/tracks.npy \
  --visibility results/test/tracking/visibility.npy \
  --out_dir results/test/displacement \
  --fps 50 \
  --axis y
```

Outputs:

```text
displacement_curve.png
displacement_fft.png
```

---

### Step 6. Run Motion Magnification

```bash
python scripts/magnify_bridge_video_fd4mm.py \
  --fd4mm_root external/fd4mm \
  --checkpoint path/to/your_fd4mm_checkpoint.pth \
  --frames_dir results/test/frames \
  --out_video results/test/magnified.mp4 \
  --amp 10 \
  --fps 50 \
  --side_by_side \
  --device cuda
```

---

### Step 7. Run Full Pipeline (Recommended)

Once all steps above work correctly, run the full pipeline:

```bash
python scripts/stage2_master_pipeline.py \
  --project_root . \
  --video_path data/video2/car.mp4 \
  --run_name test_run \
  --start_frame 400 \
  --end_frame 700 \
  --fps 50 \
  --orig_box 5 400 1915 490 \
  --mag_box 5 400 1915 490 \
  --amp 10 \
  --max_points 50 \
  --device cuda
```

---

### Step 8. Check Results

```text
results/test_run/
├── orig/
├── magnified/
├── comparisons/
```

Key outputs:

- tracking videos
- displacement curves
- FFT plots

---

## Notes and Tips

- **Segmentation issues**: usually caused by incorrect box. Adjust `--box`.
- **Tracking instability**: often due to boundary points. Use `--largest_only` and stable mask regions.
- **Magnification artifacts**: caused by domain gap between training and real data.
- **Path errors**: check all external paths (`fd4mm`, `sam2`, `cotracker`).

---

## Recommended Workflow

1. Run each module independently.
2. Verify outputs at each stage.
3. Then run the full pipeline.
4. Finally, analyze displacement and FFT results.
