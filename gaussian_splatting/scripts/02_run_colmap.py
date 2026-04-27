#!/usr/bin/env python3
"""Run the full COLMAP SfM pipeline: feature extraction → matching → mapping → undistortion."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], desc: str):
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"CMD:  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\nERROR: '{desc}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def check_colmap():
    if shutil.which("colmap") is None:
        print("ERROR: 'colmap' not found in PATH.")
        print("  Ubuntu:  sudo apt-get install colmap")
        print("  macOS:   brew install colmap")
        sys.exit(1)


def feature_extraction(db_path: str, image_dir: str, use_gpu: bool):
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.max_image_size", "1600",
        "--FeatureExtraction.use_gpu", "1" if use_gpu else "0",
    ]
    run_cmd(cmd, "Feature extraction (SIFT)")


def exhaustive_matching(db_path: str, use_gpu: bool):
    run_cmd([
        "colmap", "exhaustive_matcher",
        "--database_path", db_path,
        "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
        "--FeatureMatching.guided_matching", "1",
    ], "Exhaustive feature matching")


def sequential_matching(db_path: str, use_gpu: bool, overlap: int = 10):
    run_cmd([
        "colmap", "sequential_matcher",
        "--database_path", db_path,
        "--SequentialMatching.overlap", str(overlap),
        "--SequentialMatching.loop_detection", "1",
        "--SequentialMatching.loop_detection_period", "10",
        "--SequentialMatching.loop_detection_num_images", "20",
        "--FeatureMatching.use_gpu", "1" if use_gpu else "0",
        "--FeatureMatching.guided_matching", "1",
    ], f"Sequential feature matching (overlap={overlap})")


def run_mapper(db_path: str, image_dir: str, sparse_dir: str):
    Path(sparse_dir).mkdir(parents=True, exist_ok=True)
    run_cmd([
        "colmap", "mapper",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir,
        "--Mapper.num_threads", "8",
        "--Mapper.init_min_tri_angle", "4",
        "--Mapper.multiple_models", "0",
    ], "Sparse reconstruction (mapper)")


def bundle_adjustment(sparse_model_dir: str):
    run_cmd([
        "colmap", "bundle_adjuster",
        "--input_path", sparse_model_dir,
        "--output_path", sparse_model_dir,
        "--BundleAdjustment.refine_principal_point", "1",
    ], "Bundle adjustment")


def export_txt(sparse_model_dir: str, txt_dir: str):
    Path(txt_dir).mkdir(parents=True, exist_ok=True)
    run_cmd([
        "colmap", "model_converter",
        "--input_path", sparse_model_dir,
        "--output_path", txt_dir,
        "--output_type", "TXT",
    ], "Export sparse model to TXT")


def image_undistorter(image_dir: str, sparse_model_dir: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run_cmd([
        "colmap", "image_undistorter",
        "--image_path", image_dir,
        "--input_path", sparse_model_dir,
        "--output_path", output_dir,
        "--output_type", "COLMAP",
        "--max_image_size", "1600",
    ], "Image undistortion (prepares gsplat input)")



def validate_reconstruction(sparse_model_dir: str, n_input_images: int):
    try:
        import pycolmap
    except ImportError:
        print("WARNING: pycolmap not installed, skipping validation.")
        return

    print(f"\n{'='*60}")
    print("VALIDATION: Checking reconstruction quality")
    print(f"{'='*60}")

    try:
        recon = pycolmap.Reconstruction(sparse_model_dir)
    except Exception as e:
        print(f"ERROR: Could not load reconstruction: {e}")
        sys.exit(1)

    n_registered = len(recon.images)
    n_points = len(recon.points3D)
    pct_registered = 100.0 * n_registered / n_input_images if n_input_images > 0 else 0

    errors = [p.error for p in recon.points3D.values()]
    mean_error = sum(errors) / len(errors) if errors else float("inf")

    print(f"  Registered images:  {n_registered} / {n_input_images} ({pct_registered:.1f}%)")
    print(f"  3D points:          {n_points}")
    print(f"  Mean reprojection error: {mean_error:.3f} px")

    ok = True
    if pct_registered < 80:
        print(f"  WARNING: Only {pct_registered:.1f}% of images registered.")
        print("    → Try increasing --matching-overlap or switching to exhaustive matching.")
        ok = False
    if n_points < 500:
        print(f"  WARNING: Only {n_points} 3D points — reconstruction may be too sparse.")
        ok = False
    if mean_error > 2.0:
        print(f"  WARNING: High reprojection error ({mean_error:.2f} px). Check image quality.")

    if ok:
        print("  Reconstruction looks good!")

    return n_registered, n_points


def count_images(image_dir: str) -> int:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sum(1 for f in Path(image_dir).iterdir() if f.suffix in exts)


def parse_args():
    parser = argparse.ArgumentParser(description="Run full COLMAP SfM pipeline")
    parser.add_argument("--frames-dir", default="data/frames",
                        help="Directory containing extracted frames")
    parser.add_argument("--workspace", default="data/colmap_workspace",
                        help="COLMAP working directory")
    parser.add_argument("--gsplat-input", default="data/gsplat_input",
                        help="Output directory for undistorted gsplat-ready data")
    parser.add_argument("--matching", choices=["auto", "sequential", "exhaustive"],
                        default="auto",
                        help="Matching strategy: auto selects based on image count (default: auto)")
    parser.add_argument("--matching-overlap", type=int, default=10,
                        help="Sequential matching overlap (default: 10)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU for feature extraction/matching")
    return parser.parse_args()


def main():
    args = parse_args()
    check_colmap()

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    image_dir = str(Path(args.frames_dir).resolve())
    db_path = str(workspace / "database.db")
    sparse_dir = str(workspace / "sparse")
    sparse_model_dir = str(workspace / "sparse" / "0")
    txt_dir = str(workspace / "sparse_txt")
    use_gpu = not args.no_gpu

    n_images = count_images(image_dir)
    print(f"Found {n_images} images in {image_dir}")

    if n_images < 10:
        print(f"ERROR: Only {n_images} images found. Need at least 10 for reconstruction.")
        sys.exit(1)

    # 1. Feature extraction
    feature_extraction(db_path, image_dir, use_gpu)

    # 2. Matching — auto-select based on image count
    matching = args.matching
    if matching == "auto":
        matching = "sequential" if n_images >= 100 else "exhaustive"
        print(f"\nAuto-selected '{matching}' matching for {n_images} images.")

    if matching == "sequential":
        sequential_matching(db_path, use_gpu, overlap=args.matching_overlap)
    else:
        exhaustive_matching(db_path, use_gpu)

    # 3. Sparse reconstruction
    run_mapper(db_path, image_dir, sparse_dir)

    if not Path(sparse_model_dir).exists():
        print(f"\nERROR: No reconstruction at {sparse_model_dir}.")
        print("  COLMAP may have failed to initialize. Try:")
        print("  - Increasing --matching-overlap")
        print("  - Using --matching exhaustive")
        print("  - Checking image quality and overlap")
        sys.exit(1)

    # 4. Bundle adjustment
    bundle_adjustment(sparse_model_dir)

    # 5. Export TXT (for debugging)
    export_txt(sparse_model_dir, txt_dir)

    # 6. Validate
    validate_reconstruction(sparse_model_dir, n_images)

    # 7. Undistort images → gsplat input
    image_undistorter(image_dir, sparse_model_dir, args.gsplat_input)

    print(f"\n{'='*60}")
    print("COLMAP pipeline complete!")
    print(f"  gsplat input ready at: {args.gsplat_input}/")
    print(f"  Next step: python scripts/03_prepare_gsplat_data.py --gsplat-input {args.gsplat_input}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
