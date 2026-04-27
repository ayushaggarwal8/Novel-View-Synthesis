#!/usr/bin/env python3
"""Validate gsplat input data and export scene_info.json for downstream scripts."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_reconstruction(sparse_dir: str):
    try:
        import pycolmap
    except ImportError:
        print("ERROR: pycolmap not installed. Run: pip install pycolmap")
        sys.exit(1)

    try:
        recon = pycolmap.Reconstruction(sparse_dir)
    except Exception as e:
        print(f"ERROR: Failed to load reconstruction from {sparse_dir}: {e}")
        sys.exit(1)

    return recon


def quaternion_to_rotation_matrix(qw, qx, qy, qz) -> np.ndarray:
    """Convert Hamilton quaternion (w,x,y,z) to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    return R


def compute_scene_scale(camtoworlds: np.ndarray) -> float:
    """Max distance from scene center to any camera — used for trajectory sizing."""
    centers = camtoworlds[:, :3, 3]
    mean_center = centers.mean(axis=0)
    distances = np.linalg.norm(centers - mean_center, axis=1)
    return float(distances.max())


def parse_reconstruction(recon) -> tuple[list, np.ndarray]:
    """Extract per-image camera-to-world matrices and metadata."""
    images_info = []
    camtoworlds = []

    for img_id, image in recon.images.items():
        # world-to-camera from pycolmap (cam_from_world is a method in pycolmap >= 3.x)
        pose = image.cam_from_world()
        R = pose.rotation.matrix()
        t = pose.translation

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        c2w = np.linalg.inv(w2c)
        camtoworlds.append(c2w)

        cam = recon.cameras[image.camera_id]
        images_info.append({
            "image_id": img_id,
            "name": image.name,
            "camera_id": image.camera_id,
            "width": cam.width,
            "height": cam.height,
            "cam_from_world_R": R.tolist(),
            "cam_from_world_t": t.tolist(),
        })

    return images_info, np.array(camtoworlds)


def find_sparse_model_dir(gsplat_input_dir: Path) -> Path | None:
    """
    Locate the sparse model directory, handling both COLMAP layouts:
      - COLMAP 3.x / image_undistorter: sparse/0/cameras.bin
      - COLMAP 4.x / image_undistorter: sparse/cameras.bin (no subdirectory)
    Returns the directory containing cameras.bin, or None if not found.
    """
    for candidate in [gsplat_input_dir / "sparse" / "0", gsplat_input_dir / "sparse"]:
        if (candidate / "cameras.bin").exists():
            return candidate
    return None


def validate_directory_structure(gsplat_input_dir: str) -> bool:
    base = Path(gsplat_input_dir)
    ok = True

    if (base / "images").exists():
        print(f"  OK:      {base / 'images'}")
    else:
        print(f"  MISSING: {base / 'images'}")
        ok = False

    sparse_dir = find_sparse_model_dir(base)
    if sparse_dir is not None:
        print(f"  OK:      {sparse_dir}  (sparse model)")
    else:
        print(f"  MISSING: {base / 'sparse'}/[0/]cameras.bin")
        ok = False

    return ok


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate gsplat input data and export scene_info.json")
    parser.add_argument("--gsplat-input", default="data/gsplat_input",
                        help="Path to undistorted gsplat input directory")
    return parser.parse_args()


def main():
    args = parse_args()
    gsplat_input = Path(args.gsplat_input)

    print(f"Validating gsplat input at: {gsplat_input}/")
    print()

    # Check directory structure
    print("Checking directory structure:")
    if not validate_directory_structure(str(gsplat_input)):
        print("\nERROR: Missing required directories.")
        print("  → Run 02_run_colmap.py first.")
        sys.exit(1)

    # Count images
    image_dir = gsplat_input / "images"
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    print(f"\nImages in gsplat input: {len(image_files)}")

    # Load reconstruction (works with both COLMAP 3.x and 4.x layouts)
    sparse_dir = str(find_sparse_model_dir(gsplat_input))
    print(f"\nLoading reconstruction from {sparse_dir} ...")
    recon = load_reconstruction(sparse_dir)

    n_registered = len(recon.images)
    n_points = len(recon.points3D)
    print(f"  Registered images: {n_registered}")
    print(f"  3D points:         {n_points}")

    # Parse camera poses
    images_info, camtoworlds = parse_reconstruction(recon)
    scene_scale = compute_scene_scale(camtoworlds)

    camera_positions = camtoworlds[:, :3, 3]
    scene_center = camera_positions.mean(axis=0).tolist()

    print(f"  Scene scale:       {scene_scale:.4f} (max cam-to-center dist)")
    print(f"  Scene center:      [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")

    # Get intrinsics summary
    cameras_summary = []
    for cam_id, cam in recon.cameras.items():
        cameras_summary.append({
            "camera_id": cam_id,
            "model": str(cam.model),
            "width": cam.width,
            "height": cam.height,
            "params": cam.params.tolist(),
        })

    # Export scene_info.json
    scene_info = {
        "n_images": n_registered,
        "n_points": n_points,
        "scene_scale": scene_scale,
        "scene_center": scene_center,
        "cameras": cameras_summary,
        "registered_images": images_info,
    }

    out_path = gsplat_input / "scene_info.json"
    with open(out_path, "w") as f:
        json.dump(scene_info, f, indent=2)

    print(f"\nExported scene info → {out_path}")
    print(f"\nNext step: python scripts/04_train_gsplat.py --data-dir {args.gsplat_input}")


if __name__ == "__main__":
    main()
