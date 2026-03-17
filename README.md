# fd4mm_stage_2

Stage-2 self-supervised motion magnification pipeline for structural video analysis.

This repository combines:

- **FD4MM** for motion magnification
- **SAM2** for bridge/object segmentation
- **CoTracker** for dense point tracking
- **displacement and FFT analysis**
- **stage-2 self-supervised adaptation** using CoTracker-based motion consistency

The main use case is **bridge or structural vibration analysis** from real videos. The pipeline can:

1. extract a target frame interval from a video
2. segment the structure with SAM2
3. sample stable points on the segmented region
4. track those points with CoTracker
5. estimate displacement and frequency content
6. magnify motion using FD4MM
7. repeat segmentation and tracking on the magnified result
8. compare original vs magnified motion
9. optionally fine-tune FD4MM with a self-supervised stage-2 loss

---

## Repository Structure

```text
fd4mm_stage_2/
├── env/
├── external/
│   └── sam2/
├── models/
│   ├── sam2.1_hiera_small.pt.txt
│   └── sam2_hiera_small.pt.txt
├── scripts/
│   ├── extract_frames.py
│   ├── magnify_bridge_video_fd4mm.py
│   ├── plot_displacement_from_tracks.py
│   ├── run_cotracker_bridge.py
│   ├── run_stage2_preprocess.py
│   ├── sample_bridge_points.py
│   ├── segment_bridge_sam2.py
│   ├── stage2_master_pipeline.py
│   └── test_box.py
└── stage2/
    ├── __init__.py
    └── train_stage2_cotracker.py

## What This Project Does

### 1. Original-video analysis

Given an input structural video, the pipeline extracts a user-defined frame interval and performs:

- segmentation of the target structure using SAM2
- automatic point sampling inside the segmented region
- point tracking using CoTracker
- displacement estimation from trajectories
- FFT analysis of the displacement signal

### 2. Motion magnification

The same interval is passed to FD4MM to produce a magnified video. Then the pipeline performs the same segmentation, point tracking, and displacement analysis on the magnified result.

### 3. Comparison

The pipeline generates side-by-side outputs for:

- original vs magnified videos
- segmentation overlays
- tracking overlays
- displacement curves
- FFT curves

### 4. Stage-2 self-supervised adaptation

A dedicated training script fine-tunes FD4MM on real video clips using CoTracker-derived motion consistency, without requiring pixel-level magnified ground truth.

---

## Main Components

### 1) `scripts/stage2_master_pipeline.py`

This is the main one-command pipeline.

It orchestrates the full workflow:

- read an input video
- extract a chosen frame interval
- run SAM2 on original frames
- sample points from the original mask
- run CoTracker on original frames
- compute displacement and FFT
- magnify the interval with FD4MM
- run SAM2 on magnified frames
- sample points on the magnified result
- run CoTracker on magnified frames
- compute displacement and FFT again
- produce side-by-side comparison videos and plots
- archive the results

#### Main arguments

- `--project_root`: repository root
- `--video_path`: input video path
- `--run_name`: experiment name
- `--start_frame`: start frame index
- `--end_frame`: end frame index
- `--fps`: frame rate used for analysis
- `--orig_box x1 y1 x2 y2`: initial SAM2 box on original frames
- `--mag_box x1 y1 x2 y2`: initial SAM2 box on magnified frames
- `--amp`: magnification factor
- `--max_points`: maximum number of sampled tracking points
- `--grid_spacing`: point sampling spacing
- `--device`: `cuda` or `cpu`
- `--sam2_cfg`: SAM2 config path
- `--sam2_ckpt`: SAM2 checkpoint path
- `--fd4mm_root`: FD4MM code root
- `--fd4mm_ckpt`: FD4MM checkpoint path
- `--tail_len`: tracking tail length in visualization
- `--axis`: displacement axis (`0=x`, `1=y`)

#### Example

```bash
python scripts/stage2_master_pipeline.py \
  --project_root . \
  --video_path data/video2/car.mp4 \
  --run_name car_400_700_x10 \
  --start_frame 400 \
  --end_frame 700 \
  --fps 50 \
  --orig_box 5 400 1915 490 \
  --mag_box 5 400 1915 490 \
  --amp 10 \
  --max_points 30 \
  --device cuda

### 2) `scripts/segment_bridge_sam2.py`

Runs SAM2 video segmentation over a directory of sequential frames.

#### Features

- box-prompt initialization on frame 0
- video propagation through the full interval
- optional post-processing
- optional overlay visualization
- optional logits saving

#### Important options

- `--frames_dir`
- `--out_dir`
- `--sam2_cfg`
- `--sam2_ckpt`
- `--box X1 Y1 X2 Y2`
- `--device`
- `--score_thresh`
- `--save_overlay`
- `--save_logits`
- `--largest_only`
- `--bridge_prior`
- `--open_iter`
- `--close_iter`
- `--erode_iter`
- `--dilate_iter`
- `--hole_area`

#### Notes

This script includes mask cleanup logic such as:

- keeping the largest connected component
- filling small holes
- optional thin-bridge prior
- morphological opening/closing/erosion/dilation

### 3) `scripts/sample_bridge_points.py`

Samples tracking points from a binary mask.

It is designed to avoid unstable regions by keeping a more reliable middle stripe of the bridge/object mask before grid sampling.

#### Main outputs

- sampled query points for CoTracker
- optional visualization files depending on the calling pipeline

#### Core idea

Instead of sampling points over the entire mask, it keeps a more stable interior region to reduce noisy boundaries and unstable tracking.

### 4) `scripts/run_cotracker_bridge.py`

Runs CoTracker3 offline on a sequence of frames using user-provided query points.

#### Inputs

- frame directory
- query points in `.npy` format with shape `[N, 3]`
- each row is `[frame_idx, x, y]`

#### Outputs

- `tracks.npy`
- `visibility.npy`
- `queries_used.npy`
- tracking overlay video
- first-frame preview with query points
- `meta.json`

#### Example

```bash
python scripts/run_cotracker_bridge.py \
  --frames_dir results/example/orig/frames \
  --queries results/example/orig/queries/bridge_queries.npy \
  --out_dir results/example/orig/tracking \
  --device cuda \
  --save_video \
  --fps 50 \
  --tail_len 10
````markdown
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
````

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

* tracking videos
* displacement curves
* FFT plots

---

## Notes and Tips

* **Segmentation issues**: usually caused by incorrect box. Adjust `--box`.
* **Tracking instability**: often due to boundary points. Use `--largest_only` and stable mask regions.
* **Magnification artifacts**: caused by domain gap between training and real data.
* **Path errors**: check all external paths (`fd4mm`, `sam2`, `cotracker`).

---

## Recommended Workflow

1. Run each module independently
2. Verify outputs at each stage
3. Then run the full pipeline
4. Finally, analyze displacement and FFT results

```
```
