#!/usr/bin/env python3
"""
Create comparison grids: input image + 4 novel views side by side.
Produces one grid image per input frame — useful for the report.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_grid(input_img: Image.Image, novel_views: list[Image.Image],
              labels: list[str], cell_size: int = 256) -> Image.Image:
    """Arrange input + novel views in a 1×5 horizontal strip with labels."""
    n = 1 + len(novel_views)
    grid_w = cell_size * n
    label_h = 30
    grid_h = cell_size + label_h

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    all_imgs = [input_img] + novel_views
    all_labels = ["Input"] + labels

    for i, (img, label) in enumerate(zip(all_imgs, all_labels)):
        img_resized = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(img_resized, (i * cell_size, label_h))
        # Label
        text_x = i * cell_size + cell_size // 2
        draw.text((text_x, 8), label, fill=(0, 0, 0), anchor="mm")

    return grid


def parse_args():
    parser = argparse.ArgumentParser(description="Create novel view comparison grids")
    parser.add_argument("--output-dir", default="diffusion_nvs/outputs/novel_views",
                        help="Directory containing generated novel views")
    parser.add_argument("--grid-dir", default="diffusion_nvs/outputs/grids",
                        help="Output directory for grid images")
    parser.add_argument("--cell-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    grid_dir = Path(args.grid_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)

    # Find all input images
    input_images = sorted(output_dir.glob("*_input.png"))
    if not input_images:
        print(f"No input images found in {output_dir}")
        print("Run 02_generate_novel_views.py first.")
        return

    print(f"Found {len(input_images)} input images, creating grids...")

    for input_path in input_images:
        stem = input_path.stem.replace("_input", "")

        # Find corresponding novel views
        novel_paths = sorted(output_dir.glob(f"{stem}_elev*.png"))
        if not novel_paths:
            print(f"  WARNING: No novel views found for {stem}, skipping")
            continue

        input_img = Image.open(input_path).convert("RGB")
        novel_imgs = [Image.open(p).convert("RGB") for p in novel_paths]

        # Extract angle labels from filenames
        labels = []
        for p in novel_paths:
            name = p.stem  # e.g. input_0000_elev+0_azim+90
            parts = name.split("_")
            elev_part = next((x for x in parts if x.startswith("elev")), "")
            azim_part = next((x for x in parts if x.startswith("azim")), "")
            labels.append(f"{elev_part}\n{azim_part}")

        grid = make_grid(input_img, novel_imgs, labels, cell_size=args.cell_size)
        out_path = grid_dir / f"{stem}_grid.png"
        grid.save(out_path)
        print(f"  Saved: {out_path.name}")

    print(f"\n{len(input_images)} grids saved → {grid_dir}/")


if __name__ == "__main__":
    main()
