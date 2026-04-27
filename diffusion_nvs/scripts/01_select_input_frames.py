#!/usr/bin/env python3
"""
Select representative input frames from data/frames/ for diffusion NVS.

Diffusion models work from individual images, so we pick a diverse subset
of the extracted frames — covering different robot poses and viewpoints.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def compute_frame_diversity(frames_dir: Path, n_select: int) -> list[Path]:
    """
    Greedily select n_select frames that are maximally diverse from each other,
    using mean pixel difference as the diversity metric.
    """
    image_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No images found in {frames_dir}")

    print(f"Found {len(image_paths)} frames, selecting {n_select} diverse ones...")

    # Load all frames as small thumbnails for fast comparison
    thumbs = []
    for p in tqdm(image_paths, desc="Loading thumbnails"):
        img = cv2.imread(str(p))
        thumb = cv2.resize(img, (64, 64)).astype(np.float32)
        thumbs.append(thumb)

    thumbs = np.array(thumbs)  # (N, 64, 64, 3)

    # Greedy max-diversity selection
    selected_indices = [0]  # start with first frame
    for _ in range(n_select - 1):
        min_dists = []
        for i in range(len(thumbs)):
            if i in selected_indices:
                min_dists.append(-1)
                continue
            # Distance to nearest already-selected frame
            dists = [np.mean(np.abs(thumbs[i] - thumbs[s])) for s in selected_indices]
            min_dists.append(min(dists))
        best = int(np.argmax(min_dists))
        selected_indices.append(best)

    selected_indices.sort()
    return [image_paths[i] for i in selected_indices]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select diverse input frames for diffusion NVS")
    parser.add_argument("--frames-dir", default="data/frames",
                        help="Directory of extracted video frames")
    parser.add_argument("--output-dir", default="diffusion_nvs/inputs",
                        help="Output directory for selected frames")
    parser.add_argument("--n-frames", type=int, default=25,
                        help="Number of frames to select (default: 25)")
    return parser.parse_args()


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)

    if not frames_dir.exists():
        print(f"ERROR: {frames_dir} not found.")
        print("  Run: python gaussian_splatting/scripts/01_extract_frames.py first")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    selected = compute_frame_diversity(frames_dir, args.n_frames)

    for i, src in enumerate(selected):
        dst = output_dir / f"input_{i:04d}{src.suffix}"
        shutil.copy2(src, dst)

    print(f"\nSelected {len(selected)} frames → {output_dir}/")
    print("Next: python diffusion_nvs/scripts/02_generate_novel_views.py")


if __name__ == "__main__":
    main()
