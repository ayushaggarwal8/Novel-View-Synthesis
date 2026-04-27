# Novel View Synthesis

Two approaches to generating novel viewpoints of a robot scene from a single input video.

```
Novel-View-Synthesis/
├── gaussian_splatting/   # 3DGS pipeline (requires CUDA GPU)
├── diffusion_nvs/        # Diffusion-based NVS (runs on Mac/MPS)
├── data/                 # Shared: extracted frames, COLMAP output
└── outputs/              # Shared outputs directory
```

---

## Approach 1: 3D Gaussian Splatting

**Located in:** [`gaussian_splatting/`](gaussian_splatting/)

Reconstructs the scene geometry using COLMAP (Structure from Motion) then trains a 3D Gaussian Splatting model to render novel views.

**Requires:** NVIDIA GPU with CUDA. Does not run on macOS.

See [`gaussian_splatting/scripts/`](gaussian_splatting/scripts/) for the full pipeline.

---

## Approach 2: Diffusion-based NVS

**Located in:** [`diffusion_nvs/`](diffusion_nvs/)

Uses a pretrained diffusion model (Stable Zero123) to synthesize novel views directly from input images — no 3D reconstruction required. Runs on Mac (MPS / Apple Silicon).

See [`diffusion_nvs/`](diffusion_nvs/) for setup and usage.

---

## Shared Data

Both approaches use the same extracted frames in `data/frames/`. Run frame extraction once:

```bash
python gaussian_splatting/scripts/01_extract_frames.py \
    --video nvs_example_input_video.mp4 \
    --output-dir data/frames \
    --fps-target 0.5
```
