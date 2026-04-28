#!/usr/bin/env python3
"""
Generate novel views using Zero123++ via HuggingFace diffusers.

Zero123++ generates 6 views simultaneously as a 960x640 grid (2 cols x 3 rows,
each cell 320x320). We keep the first n_views (default 4).

Fixed viewpoints baked into the model (elevation ~+20° above horizon):
  azimuth 30°, 90°, 150°, 210°, 270°, 330°
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


AZIMUTHS = [30, 90, 150, 210, 270, 330]   # baked into Zero123++ weights
ELEVATION = 20                              # approximate, fixed by model


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_memory(device: str) -> None:
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def load_pipeline(device: str, offload: bool):
    try:
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    except ImportError:
        print("ERROR: diffusers not installed.")
        print("  Run: pip install -r diffusion_nvs/requirements.txt")
        sys.exit(1)

    print(f"Loading Zero123++ on {device}...")
    print("  (First run downloads ~5 GB of model weights — takes a few minutes)")

    # MPS and CPU require float32; CUDA can use float16
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=dtype,
        cache_dir="diffusion_nvs/weights",
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    if offload and device != "cpu":
        # Move model layers to device one at a time during forward pass.
        # Reduces peak MPS/CUDA memory at the cost of speed.
        pipe.enable_model_cpu_offload()
        print("  CPU offload enabled (lower memory, slower inference)")
    else:
        pipe = pipe.to(device)

    pipe.enable_attention_slicing()
    return pipe


def preprocess_image(image_path: str, size: int = 320) -> Image.Image:
    """Square-crop, white-background composite, resize to size×size."""
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    crop = min(w, h)
    img = img.crop(((w - crop) // 2, (h - crop) // 2,
                    (w + crop) // 2, (h + crop) // 2))
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg.convert("RGB").resize((size, size), Image.LANCZOS)


def split_grid(grid_img: Image.Image, n_views: int) -> list[Image.Image]:
    """
    Split Zero123++ output grid (2 cols × 3 rows) into individual views.
    Layout (left→right, top→bottom): az 30, 90, 150, 210, 270, 330
    Grid PIL size is (2*view_size, 3*view_size).
    """
    w, h = grid_img.size
    cell_w, cell_h = w // 2, h // 3
    views = []
    for row in range(3):
        for col in range(2):
            box = (col * cell_w, row * cell_h,
                   (col + 1) * cell_w, (row + 1) * cell_h)
            views.append(grid_img.crop(box))
    return views[:n_views]


@torch.no_grad()
def generate_views(pipe, input_image: Image.Image, n_views: int,
                   guidance_scale: float, n_steps: int,
                   view_size: int) -> list[Image.Image]:
    # width = 2 cols × view_size, height = 3 rows × view_size
    result = pipe(
        input_image,
        num_inference_steps=n_steps,
        guidance_scale=guidance_scale,
        width=view_size * 2,
        height=view_size * 3,
    )
    return split_grid(result.images[0], n_views)


def output_paths(stem: str, output_dir: Path, n_views: int) -> list[Path]:
    return [output_dir / f"{stem}_elev+{ELEVATION}_azim+{AZIMUTHS[i]:03d}.png"
            for i in range(n_views)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate novel views using Zero123++ (runs locally on M2/MPS)")
    parser.add_argument("--input-dir", default="diffusion_nvs/inputs",
                        help="Directory of selected input frames")
    parser.add_argument("--output-dir", default="diffusion_nvs/outputs/novel_views",
                        help="Directory to save generated novel views")
    parser.add_argument("--n-views", type=int, default=4,
                        help="Views to keep per image, 1–6 (default: 4)")
    parser.add_argument("--guidance-scale", type=float, default=4.0,
                        help="Classifier-free guidance scale (default: 4.0)")
    parser.add_argument("--steps", type=int, default=75,
                        help="Diffusion inference steps (default: 75)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, mps, or cpu (default: auto)")
    parser.add_argument("--view-size", type=int, default=256,
                        help="Output pixels per view (default: 256). Use 320 for higher quality if you have >24 GB RAM.")
    parser.add_argument("--offload", action="store_true",
                        help="Enable model CPU offload to reduce peak memory usage")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images whose outputs already exist (resume a partial run)")
    return parser.parse_args()


def main():
    args = parse_args()
    args.n_views = max(1, min(args.n_views, 6))

    device = get_device() if args.device == "auto" else args.device
    print(f"Device: {device}")

    if device == "mps":
        # Remove MPS soft memory cap so the OS can manage pressure via swap
        # rather than raising an OOM error. Safe on Apple Silicon unified memory.
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: {input_dir} not found.")
        print("  Run: python diffusion_nvs/scripts/01_select_input_frames.py first")
        sys.exit(1)

    input_images = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    if not input_images:
        print(f"ERROR: No images found in {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing:
        pending = [p for p in input_images
                   if not all(op.exists() for op in output_paths(p.stem, output_dir, args.n_views))]
        skipped = len(input_images) - len(pending)
        if skipped:
            print(f"Skipping {skipped} already-processed images ({len(pending)} remaining)")
        input_images = pending

    if not input_images:
        print("All images already processed.")
        return

    pipe = load_pipeline(device, args.offload)

    selected_azimuths = AZIMUTHS[:args.n_views]
    total = len(input_images) * args.n_views
    print(f"\nAzimuths: {selected_azimuths}°  |  Elevation: +{ELEVATION}°  |  View size: {args.view_size}px")
    print(f"Total: {len(input_images)} images × {args.n_views} views = {total} outputs\n")

    total_generated = 0
    for img_path in tqdm(input_images, desc="Images"):
        stem = img_path.stem
        input_img = preprocess_image(str(img_path))
        input_img.save(output_dir / f"{stem}_input.png")

        views = generate_views(pipe, input_img, args.n_views,
                               args.guidance_scale, args.steps, args.view_size)

        for view, azim in zip(views, selected_azimuths):
            view.save(output_dir / f"{stem}_elev+{ELEVATION}_azim+{azim:03d}.png")
            total_generated += 1

        clear_memory(device)

    print(f"\nGenerated {total_generated} novel views → {output_dir}/")
    print("Next: python diffusion_nvs/scripts/03_make_grid.py")


if __name__ == "__main__":
    main()
