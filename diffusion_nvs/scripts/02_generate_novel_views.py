#!/usr/bin/env python3
"""
Generate novel views using Stable Zero123 via HuggingFace diffusers.

For each input image, generates 4 novel views at different elevation/azimuth angles
that were not present in the original video.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(device: str):
    """Load Stable Zero123 via the diffusers custom_pipeline interface."""
    try:
        from diffusers import DiffusionPipeline
    except ImportError:
        print("ERROR: diffusers not installed. Run: pip install -r diffusion_nvs/requirements.txt")
        sys.exit(1)

    print(f"Loading Stable Zero123 on {device}...")
    print("  (Downloading ~5GB model weights on first run — this takes a few minutes)")

    # MPS doesn't support float16 reliably for all ops; use float32
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-zero123",
        custom_pipeline="stable_zero123",
        torch_dtype=dtype,
        cache_dir="diffusion_nvs/weights",
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    return pipe


def preprocess_image(image_path: str, size: int = 256) -> Image.Image:
    """Load and preprocess image for Zero123: square crop, resize to 256."""
    img = Image.open(image_path).convert("RGBA")

    # Center-crop to square
    w, h = img.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))

    # White background for RGBA
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    img = bg.convert("RGB")

    img = img.resize((size, size), Image.LANCZOS)
    return img


def define_novel_view_angles(n_views: int = 4) -> list[dict]:
    """
    Define novel viewpoint angles relative to the input image.
    Angles chosen to be maximally different from a typical front-facing robot video.

    elevation: degrees up (+) or down (-) from the input camera plane
    azimuth:   degrees left (-) or right (+) rotation around vertical axis
    """
    # These 4 views give good coverage: side views and slight elevation changes
    return [
        {"elevation":  0.0, "azimuth":  90.0},   # 90° right (side view)
        {"elevation":  0.0, "azimuth": -90.0},   # 90° left (other side)
        {"elevation": 30.0, "azimuth":  45.0},   # elevated right
        {"elevation": -20.0, "azimuth": 180.0},  # slightly low, behind
    ][:n_views]


@torch.no_grad()
def generate_views(
    pipe,
    input_image: Image.Image,
    angles: list[dict],
    guidance_scale: float = 3.0,
    n_inference_steps: int = 75,
    device: str = "cpu",
) -> list[Image.Image]:
    """Generate novel views for all specified angles."""
    results = []
    for angle in angles:
        elevation = angle["elevation"]
        azimuth = angle["azimuth"]

        output = pipe(
            input_image,
            elevation=elevation,
            azimuth=azimuth,
            guidance_scale=guidance_scale,
            num_inference_steps=n_inference_steps,
            image_width=256,
            image_height=256,
        )
        results.append(output.images[0])

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate novel views using Stable Zero123")
    parser.add_argument("--input-dir", default="diffusion_nvs/inputs",
                        help="Directory of selected input frames")
    parser.add_argument("--output-dir", default="diffusion_nvs/outputs/novel_views",
                        help="Directory to save generated novel views")
    parser.add_argument("--n-views", type=int, default=4,
                        help="Novel views to generate per input image (default: 4)")
    parser.add_argument("--guidance-scale", type=float, default=3.0,
                        help="Classifier-free guidance scale (default: 3.0)")
    parser.add_argument("--steps", type=int, default=75,
                        help="Diffusion inference steps (default: 75)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, mps, or cpu (default: auto)")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Input image size for the model (default: 256)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device() if args.device == "auto" else args.device
    print(f"Using device: {device}")

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

    print(f"Found {len(input_images)} input images")

    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(device)
    angles = define_novel_view_angles(args.n_views)

    print(f"\nGenerating {args.n_views} novel views per image ({len(input_images)} images total)...")
    print(f"Novel view angles:")
    for i, a in enumerate(angles):
        print(f"  View {i}: elevation={a['elevation']:+.0f}°  azimuth={a['azimuth']:+.0f}°")
    print()

    total_generated = 0
    for img_path in tqdm(input_images, desc="Processing images"):
        img_name = img_path.stem
        input_img = preprocess_image(str(img_path), size=args.image_size)

        # Save preprocessed input alongside outputs for comparison
        input_out = output_dir / f"{img_name}_input.png"
        input_img.save(input_out)

        # Generate novel views
        novel_views = generate_views(
            pipe,
            input_img,
            angles,
            guidance_scale=args.guidance_scale,
            n_inference_steps=args.steps,
            device=device,
        )

        for view_idx, (view_img, angle) in enumerate(zip(novel_views, angles)):
            elev = angle["elevation"]
            azim = angle["azimuth"]
            out_name = f"{img_name}_elev{elev:+.0f}_azim{azim:+.0f}.png"
            view_img.save(output_dir / out_name)
            total_generated += 1

    print(f"\nGenerated {total_generated} novel views → {output_dir}/")
    print("Run: python diffusion_nvs/scripts/03_make_grid.py to create comparison grids")


if __name__ == "__main__":
    main()
