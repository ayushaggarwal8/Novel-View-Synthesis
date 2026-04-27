#!/usr/bin/env python3
"""
Render novel views from a trained 3DGS model.

Generates camera trajectories not present in the training video and renders
images using gsplat's rasterization API.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Trajectory generation helpers
# ---------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def viewmatrix(forward: np.ndarray, up: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Build a camera-to-world matrix from forward/up vectors and position."""
    right = normalize(np.cross(forward, up))
    up_true = normalize(np.cross(right, forward))
    return np.array([
        [right[0],    up_true[0],  -forward[0], pos[0]],
        [right[1],    up_true[1],  -forward[1], pos[1]],
        [right[2],    up_true[2],  -forward[2], pos[2]],
        [0,           0,            0,           1     ],
    ], dtype=np.float32)


def generate_ellipse_orbit(camtoworlds: np.ndarray, n_frames: int = 120,
                            z_variation: float = 0.15) -> np.ndarray:
    """
    Generate an elliptical orbit around the scene center.
    The orbit radius matches the mean training camera distance from center.
    """
    positions = camtoworlds[:, :3, 3]
    center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    mean_radius = radii.mean()

    z_vals = positions[:, 2]
    z_mean = z_vals.mean()
    z_range = (z_vals.max() - z_vals.min()) * z_variation

    traj = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        z = z_mean + z_range * np.sin(2 * np.pi * i / n_frames)

        pos = np.array([
            center[0] + mean_radius * np.cos(theta),
            center[1] + mean_radius * np.sin(theta),
            z,
        ])
        forward = normalize(center - pos)
        up = np.array([0, 0, 1], dtype=np.float32)
        if abs(np.dot(forward, up)) > 0.99:
            up = np.array([0, 1, 0], dtype=np.float32)
        traj.append(viewmatrix(forward, up, pos))

    return np.array(traj, dtype=np.float32)


def generate_interpolated_path(camtoworlds: np.ndarray, n_interp: int = 5) -> np.ndarray:
    """
    Smooth interpolation through training camera positions using linear
    interpolation on positions and SLERP on rotations.
    """
    from scipy.spatial.transform import Rotation, Slerp

    n = len(camtoworlds)
    rotations = Rotation.from_matrix(camtoworlds[:, :3, :3])
    positions = camtoworlds[:, :3, 3]

    t_train = np.linspace(0, 1, n)
    t_interp = np.linspace(0, 1, n * n_interp)

    slerp = Slerp(t_train, rotations)
    interp_rots = slerp(t_interp).as_matrix()
    interp_pos = np.stack([
        np.interp(t_interp, t_train, positions[:, d]) for d in range(3)
    ], axis=1)

    traj = np.eye(4, dtype=np.float32)[None].repeat(len(t_interp), axis=0)
    traj[:, :3, :3] = interp_rots
    traj[:, :3, 3] = interp_pos
    return traj


def generate_spiral_path(camtoworlds: np.ndarray, n_frames: int = 120,
                          n_rots: int = 2, z_rate: float = 0.5) -> np.ndarray:
    """
    Forward-facing spiral path: rotate while zooming in/out.
    Good for tabletop robot scenes.
    """
    positions = camtoworlds[:, :3, 3]
    center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    mean_radius = radii.mean()

    z_vals = positions[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()

    traj = []
    for i in range(n_frames):
        t = i / n_frames
        theta = 2 * np.pi * n_rots * t
        r = mean_radius * (0.7 + 0.3 * np.cos(np.pi * t))  # zoom in/out
        z = z_min + (z_max - z_min) * (0.5 + z_rate * np.sin(np.pi * t))

        pos = np.array([
            center[0] + r * np.cos(theta),
            center[1] + r * np.sin(theta),
            z,
        ])
        forward = normalize(center - pos)
        up = np.array([0, 0, 1], dtype=np.float32)
        if abs(np.dot(forward, up)) > 0.99:
            up = np.array([0, 1, 0], dtype=np.float32)
        traj.append(viewmatrix(forward, up, pos))

    return np.array(traj, dtype=np.float32)


# ---------------------------------------------------------------------------
# Novel view selection
# ---------------------------------------------------------------------------

def optical_axis(c2w: np.ndarray) -> np.ndarray:
    """Z-column of c2w = forward direction of camera in world space."""
    return c2w[:3, 2]


def min_angle_to_training(candidate_c2w: np.ndarray,
                           training_c2ws: np.ndarray) -> float:
    axis_cand = optical_axis(candidate_c2w)
    min_angle = float("inf")
    for train_c2w in training_c2ws:
        axis_train = optical_axis(train_c2w)
        cos_a = np.clip(np.dot(axis_cand, axis_train), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))
        min_angle = min(min_angle, angle)
    return min_angle


def select_novel_views(traj: np.ndarray, training_c2ws: np.ndarray,
                        n_views: int = 4, min_angle_deg: float = 10.0) -> list[int]:
    """
    Select n_views frames from trajectory that are most angular-novel
    relative to training views, while being spread across the trajectory.
    """
    scores = []
    for i, c2w in enumerate(traj):
        angle = min_angle_to_training(c2w, training_c2ws)
        scores.append((i, angle))

    scores.sort(key=lambda x: -x[1])

    # Greedily pick views that are spread across trajectory indices
    selected = []
    used_regions = set()
    region_size = max(1, len(traj) // (n_views * 2))

    for idx, angle in scores:
        region = idx // region_size
        if region not in used_regions:
            selected.append(idx)
            used_regions.add(region)
            if len(selected) >= n_views:
                break

    # If we couldn't find enough truly novel views, relax the spread constraint
    if len(selected) < n_views:
        for idx, angle in scores:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= n_views:
                break

    selected.sort()
    return selected


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def load_splats(result_dir: Path, step: int | None, device: str) -> dict:
    """Load Gaussian splat parameters from checkpoint."""
    ckpt_dir = result_dir / "ckpts"
    if not ckpt_dir.exists():
        ckpt_dir = result_dir / "checkpoints"

    if step is not None:
        candidates = list(ckpt_dir.glob(f"*{step}*.pt")) + list(ckpt_dir.glob(f"ckpt_{step:07d}_rank0.pt"))
    else:
        candidates = sorted(ckpt_dir.glob("*.pt"))

    if not candidates:
        print(f"ERROR: No checkpoint found in {ckpt_dir}")
        sys.exit(1)

    ckpt_path = candidates[-1]
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # gsplat simple_trainer saves: {"splats": {...}, "step": int}
    if isinstance(ckpt, dict) and "splats" in ckpt:
        splats = {k: v.to(device) for k, v in ckpt["splats"].items()}
    elif isinstance(ckpt, dict) and "means" in ckpt:
        splats = {k: v.to(device) for k, v in ckpt.items()}
    else:
        print(f"ERROR: Unrecognized checkpoint format. Keys: {list(ckpt.keys())}")
        sys.exit(1)

    print(f"  Gaussians: {splats['means'].shape[0]:,}")
    return splats


def load_training_poses(gsplat_input: str) -> tuple[np.ndarray, dict]:
    """Load camera-to-world matrices and intrinsics from scene_info.json."""
    scene_info_path = Path(gsplat_input) / "scene_info.json"
    if not scene_info_path.exists():
        print(f"ERROR: {scene_info_path} not found. Run 03_prepare_gsplat_data.py first.")
        sys.exit(1)

    with open(scene_info_path) as f:
        scene_info = json.load(f)

    camtoworlds = []
    for img in scene_info["registered_images"]:
        R = np.array(img["cam_from_world_R"])
        t = np.array(img["cam_from_world_t"])
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c)
        camtoworlds.append(c2w.astype(np.float32))

    camtoworlds = np.array(camtoworlds)

    # Use first camera's intrinsics (all PINHOLE after undistortion)
    cam = scene_info["cameras"][0]
    params = cam["params"]
    # COLMAP PINHOLE: [fx, fy, cx, cy]
    K = np.array([
        [params[0], 0,         params[2]],
        [0,         params[1], params[3]],
        [0,         0,         1        ],
    ], dtype=np.float32)

    intrinsics = {
        "K": K,
        "width": cam["width"],
        "height": cam["height"],
    }

    return camtoworlds, intrinsics


def render_frame(splats: dict, c2w: np.ndarray, K: np.ndarray,
                 width: int, height: int, sh_degree: int, device: str) -> np.ndarray:
    """Render a single frame given camera-to-world pose and intrinsics."""
    from gsplat import rasterization

    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=device)
    viewmat = torch.linalg.inv(c2w_t).unsqueeze(0)  # (1, 4, 4)
    Ks = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3, 3)

    means = splats["means"]                          # (N, 3)
    quats = splats["quats"]                          # (N, 4) w,x,y,z
    scales = torch.exp(splats["scales"])             # stored in log space
    opacities = torch.sigmoid(splats["opacities"])   # stored pre-sigmoid

    # Compute colors from spherical harmonics
    sh0 = splats["sh0"]   # (N, 1, 3)
    shN = splats.get("shN", None)

    if shN is not None:
        sh_coeffs = torch.cat([sh0, shN], dim=1)  # (N, K, 3)
    else:
        sh_coeffs = sh0

    # Camera positions in world space for view-dependent SH
    camera_pos = c2w_t[:3, 3].unsqueeze(0)  # (1, 3)
    dirs = means.unsqueeze(0) - camera_pos.unsqueeze(1)  # (1, N, 3)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

    from gsplat.rendering import spherical_harmonics
    colors = spherical_harmonics(sh_degree, dirs[0], sh_coeffs)
    colors = torch.clamp(colors + 0.5, 0.0, 1.0)  # (N, 3)

    with torch.no_grad():
        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=0.01,
            far_plane=200.0,
            render_mode="RGB",
            packed=False,
            absgrad=False,
        )

    # render_colors: (1, H, W, 3)
    img = render_colors[0].clamp(0, 1).cpu().numpy()
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Render novel views from trained 3DGS model")
    parser.add_argument("--gsplat-input", default="data/gsplat_input",
                        help="Path to gsplat input directory (contains scene_info.json)")
    parser.add_argument("--result-dir", default="outputs",
                        help="Training result directory containing ckpts/")
    parser.add_argument("--output-dir", default="outputs/novel_views",
                        help="Directory to save rendered novel views")
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step to load (default: latest)")
    parser.add_argument("--n-views", type=int, default=4,
                        help="Novel views per trajectory type (default: 4)")
    parser.add_argument("--min-angle", type=float, default=10.0,
                        help="Min angular distance (deg) from any training view to be 'novel' (default: 10)")
    parser.add_argument("--sh-degree", type=int, default=3,
                        help="SH degree used during training (default: 3)")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device (default: cuda)")
    parser.add_argument("--traj-frames", type=int, default=120,
                        help="Frames to generate per trajectory before selecting novel subset (default: 120)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU (will be slow)")
        device = "cpu"

    try:
        import gsplat
    except ImportError:
        print("ERROR: gsplat not installed. Run: pip install gsplat")
        sys.exit(1)

    # Load training poses and intrinsics
    print("Loading training camera poses...")
    training_c2ws, intrinsics = load_training_poses(args.gsplat_input)
    K = intrinsics["K"]
    width = intrinsics["width"]
    height = intrinsics["height"]
    print(f"  {len(training_c2ws)} training cameras  |  {width}x{height}")

    # Load trained Gaussians
    print("\nLoading trained Gaussian splats...")
    result_dir = Path(args.result_dir)
    splats = load_splats(result_dir, args.step, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate trajectories and render
    trajectory_configs = [
        ("ellipse",       generate_ellipse_orbit,      {"n_frames": args.traj_frames}),
        ("interpolated",  generate_interpolated_path,  {"n_interp": 5}),
        ("spiral",        generate_spiral_path,         {"n_frames": args.traj_frames}),
    ]

    total_rendered = 0
    all_novel_angles = []

    for traj_name, traj_fn, traj_kwargs in trajectory_configs:
        print(f"\n--- Trajectory: {traj_name} ---")
        traj = traj_fn(training_c2ws, **traj_kwargs)
        print(f"  Generated {len(traj)} candidate frames")

        novel_indices = select_novel_views(
            traj, training_c2ws, n_views=args.n_views, min_angle_deg=args.min_angle
        )

        traj_out = out_dir / traj_name
        traj_out.mkdir(parents=True, exist_ok=True)

        for rank, frame_idx in enumerate(tqdm(novel_indices, desc=f"Rendering {traj_name}")):
            c2w = traj[frame_idx]
            angle = min_angle_to_training(c2w, training_c2ws)
            all_novel_angles.append(angle)

            img = render_frame(splats, c2w, K, width, height, args.sh_degree, device)
            out_path = traj_out / f"novel_{rank:04d}.png"
            imageio.imwrite(str(out_path), img)

        total_rendered += len(novel_indices)
        print(f"  Saved {len(novel_indices)} novel views → {traj_out}/")

    print(f"\n{'='*60}")
    print(f"Rendering complete: {total_rendered} novel views total")
    print(f"Output directory: {out_dir}/")
    if all_novel_angles:
        print(f"Angular novelty — min: {min(all_novel_angles):.1f}°  "
              f"mean: {np.mean(all_novel_angles):.1f}°  "
              f"max: {max(all_novel_angles):.1f}°")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
