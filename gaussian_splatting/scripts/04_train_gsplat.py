#!/usr/bin/env python3
"""Train 3D Gaussian Splatting using gsplat's simple_trainer."""

import argparse
import subprocess
import sys
from pathlib import Path


def find_simple_trainer() -> Path | None:
    """Locate gsplat's simple_trainer.py — prefer bundled third_party copy."""
    # Project-local copy (always preferred — version-pinned to installed gsplat)
    project_root = Path(__file__).parent.parent
    local = project_root / "third_party" / "gsplat_examples" / "simple_trainer.py"
    if local.exists():
        return local

    # Fallback: installed package source (dev installs)
    try:
        import gsplat
        gsplat_dir = Path(gsplat.__file__).parent
        for candidate in [
            gsplat_dir.parent / "examples" / "simple_trainer.py",
            gsplat_dir / "examples" / "simple_trainer.py",
        ]:
            if candidate.exists():
                return candidate
    except ImportError:
        pass
    return None


def build_trainer_args(args) -> list[str]:
    cmd = [sys.executable]

    trainer = find_simple_trainer()
    if trainer is None:
        # Fall back: assume gsplat examples installed to PATH or run as module
        print("WARNING: Could not locate simple_trainer.py automatically.")
        print("  Trying: python -m gsplat.examples.simple_trainer")
        cmd += ["-m", "gsplat.examples.simple_trainer"]
    else:
        print(f"Found simple_trainer at: {trainer}")
        cmd.append(str(trainer))

    cmd += [
        "default",
        "--data-dir", args.data_dir,
        "--result-dir", args.result_dir,
        "--max-steps", str(args.max_steps),
        "--sh-degree", str(args.sh_degree),
        "--ssim-lambda", str(args.ssim_lambda),
        "--data-factor", str(args.data_factor),
        "--antialiased",
        "--near-plane", str(args.near_plane),
        "--far-plane", str(args.far_plane),
        "--test-every", str(args.test_every),
        "--eval-steps", *[str(s) for s in args.eval_steps],
        "--save-steps", *[str(s) for s in args.save_steps],
        "--render-traj-path", args.render_traj_path,
        "--tb-every", "100",
        # DefaultStrategy densification settings (gsplat 1.5.x uses --strategy.* prefix)
        "--strategy.refine-start-iter", "500",
        "--strategy.refine-stop-iter", str(min(5000, args.max_steps - 500)),
        "--strategy.refine-every", "100",
        "--strategy.reset-every", "3000",
        "--strategy.grow-grad2d", "0.0002",
        "--strategy.grow-scale3d", "0.01",
        "--strategy.prune-scale3d", "0.1",
    ]

    return cmd


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting with gsplat")
    parser.add_argument("--data-dir", default="data/gsplat_input",
                        help="Path to undistorted COLMAP data (gsplat input)")
    parser.add_argument("--result-dir", default="outputs",
                        help="Output directory for checkpoints and renders")
    parser.add_argument("--max-steps", type=int, default=7000,
                        help="Training iterations (default: 7000)")
    parser.add_argument("--sh-degree", type=int, default=3,
                        help="Spherical harmonics degree (default: 3)")
    parser.add_argument("--ssim-lambda", type=float, default=0.2,
                        help="SSIM loss weight (default: 0.2)")
    parser.add_argument("--data-factor", type=int, default=1,
                        help="Image downsampling factor; use 2 if CUDA OOM (default: 1)")
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=200.0)
    parser.add_argument("--test-every", type=int, default=8,
                        help="Hold out every N-th image for evaluation (default: 8)")
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[3000, 5000, 7000],
                        help="Steps at which to evaluate PSNR/SSIM")
    parser.add_argument("--save-steps", type=int, nargs="+", default=[3000, 7000],
                        help="Steps at which to save checkpoints")
    parser.add_argument("--render-traj-path", default="ellipse",
                        choices=["ellipse", "spiral", "interpolation"],
                        help="Trajectory type for built-in render at end of training")
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("  → Run 02_run_colmap.py and 03_prepare_gsplat_data.py first.")
        sys.exit(1)

    sparse_dir = data_dir / "sparse" / "0"
    if not sparse_dir.exists():
        print(f"ERROR: Sparse model not found at {sparse_dir}")
        print("  → COLMAP undistortion output is missing.")
        sys.exit(1)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gsplat
        print(f"gsplat version: {gsplat.__version__}")
    except ImportError:
        print("ERROR: gsplat not installed.")
        print("  → Run: pip install gsplat")
        sys.exit(1)

    cmd = build_trainer_args(args)

    print(f"\n{'='*60}")
    print("Starting 3DGS training")
    print(f"  Data:       {args.data_dir}")
    print(f"  Output:     {args.result_dir}")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Eval at:    {args.eval_steps}")
    print(f"  Save at:    {args.save_steps}")
    print(f"{'='*60}\n")

    # Ensure simple_trainer.py can import from its sibling datasets/ and utils.py
    trainer = find_simple_trainer()
    env = None
    if trainer is not None:
        import os
        env = os.environ.copy()
        extra_path = str(trainer.parent)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{extra_path}:{existing}" if existing else extra_path

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\nERROR: Training failed with exit code {result.returncode}")
        print("Common fixes:")
        print("  - CUDA OOM: add --data-factor 2 to halve resolution")
        print("  - Low PSNR: try --max-steps 15000")
        sys.exit(result.returncode)

    # Find the latest checkpoint
    ckpt_dir = result_dir / "ckpts"
    if not ckpt_dir.exists():
        ckpt_dir = result_dir / "checkpoints"

    print(f"\nTraining complete! Outputs saved to {result_dir}/")
    print(f"\nNext step: python scripts/05_render_novel_views.py \\")
    print(f"             --gsplat-input {args.data_dir} \\")
    print(f"             --result-dir {args.result_dir} \\")
    print(f"             --output-dir outputs/novel_views")


if __name__ == "__main__":
    main()
