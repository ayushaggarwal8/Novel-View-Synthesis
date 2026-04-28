# Novel View Synthesis

Two approaches to generating novel viewpoints of a robot scene from a single input video: **3D Gaussian Splatting** (scene-specific, high quality) and **Diffusion-based NVS** (single-image, no training required).

```
Novel-View-Synthesis/
├── gaussian_splatting/
│   ├── scripts/
│   │   ├── 01_extract_frames.py      # Extract frames from video
│   │   └── 02_run_colmap.py          # Run COLMAP SfM pipeline
│   └── kaggle_train_and_render.ipynb # Train 3DGS + render on Kaggle
├── diffusion_nvs/
│   └── colab_novel_views.ipynb       # Zero123++ inference on Colab/Kaggle
├── data/                             # Frames, COLMAP workspace, gsplat input
└── PIPELINE.md                       # Detailed technical explanation
```

---

## Requirements

### Local (Mac/Linux) — preprocessing only
- Python 3.10+
- COLMAP: `brew install colmap` (macOS) or `sudo apt install colmap` (Ubuntu)
- Python packages: `pip install opencv-python numpy tqdm`

### GPU (Kaggle/Colab) — training and rendering
- NVIDIA GPU with CUDA 7.0+ (Kaggle T4 x2 works)
- All GPU dependencies are installed by the notebook

---

## Approach 1: 3D Gaussian Splatting

Reconstructs the full 3D scene from a video using COLMAP, then trains a 3D Gaussian Splatting model on a GPU. After training, novel views can be rendered from any camera angle instantly.

### Step 1 — Extract frames (run locally)

```bash
python gaussian_splatting/scripts/01_extract_frames.py \
    --video nvs_example_input_video.mp4 \
    --output-dir data/frames \
    --fps-target 1.0 \
    --max-frames 300 \
    --diff-threshold 2.0
```

This samples 1 frame/second, deduplicates near-identical frames, and saves up to 300 JPEGs to `data/frames/`.

### Step 2 — Run COLMAP (run locally)

```bash
python gaussian_splatting/scripts/02_run_colmap.py \
    --frames-dir data/frames \
    --workspace data/colmap_workspace \
    --gsplat-input data/gsplat_input \
    --matching exhaustive \
    --no-gpu
```

Runs SIFT feature extraction, exhaustive matching, sparse reconstruction, bundle adjustment, and image undistortion. Output goes to `data/gsplat_input/`. Targets: 90%+ images registered, 3D reprojection error < 1.5 px.

### Step 3 — Train and render

**If you have a local CUDA GPU**, install gsplat and run directly:

```bash
pip install gsplat pycolmap torchmetrics[image] tqdm Pillow tyro
python gaussian_splatting/scripts/simple_trainer.py default \
    --data-dir data/gsplat_input \
    --result-dir outputs \
    --max-steps 7000 \
    --disable-viewer
```

**If you do not have a CUDA GPU** (e.g. on macOS), use the Kaggle notebook instead:

1. Zip and upload `data/gsplat_input/` to Kaggle as a dataset named `nvs-gsplat-input`
2. Import `gaussian_splatting/kaggle_train_and_render.ipynb` into Kaggle
3. Enable GPU: Settings → Accelerator → GPU T4 x2
4. Add dataset: right panel → Add data → `nvs-gsplat-input`
5. Run all cells

The notebook installs all dependencies, trains for 7,000 steps (~30–40 min on T4), renders 25 reference frames × 4 novel views each, and saves results to the Kaggle Output tab for download.

### Re-rendering without retraining

If running locally, point the render script at the saved checkpoint directly. If using Kaggle, upload `ckpt_6999_rank0.pt` as a dataset named `nvs-3dgs-checkpoint` — the notebook detects it automatically and skips training.

---

## Approach 2: Diffusion-based NVS

Uses **Zero123++** (`sudo-ai/zero123plus-v1.2`), a pretrained diffusion model that generates 6 novel views from a single input image. No 3D reconstruction or training required.

### Run on Colab or Kaggle

Open `diffusion_nvs/colab_novel_views.ipynb` and run all cells. The notebook:
- Installs `diffusers`, `transformers`, `accelerate`
- Downloads the Zero123++ model weights (~5 GB) from HuggingFace
- For each of 25 selected input frames: generates a 960×640 grid of 6 views, crops to 4 individual 320×320 images
- Saves 100 novel view images total

No local GPU required for setup — model weights download automatically at runtime.

---

## Output Structure

After running the 3DGS pipeline, `outputs/novel_views/` contains:

```
novel_views/
├── frame_00_trainidx0/
│   ├── 00_reference.png       # Training frame rendered by the model
│   ├── novel_01_left.png      # 12° left rotation
│   ├── novel_02_right.png     # 12° right rotation
│   ├── novel_03_up.png        # 12° upward rotation
│   └── novel_04_down.png      # 12° downward rotation
├── frame_01_trainidx9/
│   └── ...
└── ... (25 folders total)
```

---

