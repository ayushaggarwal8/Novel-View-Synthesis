#!/usr/bin/env python3
"""Extract frames from a video for use with COLMAP + 3D Gaussian Splatting."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from video for NVS pipeline")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output-dir", default="data/frames", help="Output directory for frames")
    parser.add_argument("--fps-target", type=float, default=0.5,
                        help="Target frames per second to extract (default: 0.5)")
    parser.add_argument("--max-frames", type=int, default=150,
                        help="Maximum number of frames to extract (default: 150)")
    parser.add_argument("--resize-max", type=int, default=1600,
                        help="Maximum dimension for resizing (default: 1600)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--diff-threshold", type=float, default=5.0,
                        help="Min mean pixel diff (0-255) to keep a frame; filters near-duplicates (default: 5.0)")
    parser.add_argument("--start-sec", type=float, default=None,
                        help="Start time in seconds (default: beginning of video)")
    parser.add_argument("--end-sec", type=float, default=None,
                        help="End time in seconds (default: end of video)")
    return parser.parse_args()


def resize_if_needed(frame: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def frame_is_novel(frame_gray: np.ndarray, prev_gray: np.ndarray, threshold: float) -> bool:
    """Return True if frame differs enough from the previous kept frame."""
    diff = np.mean(np.abs(frame_gray.astype(np.float32) - prev_gray.astype(np.float32)))
    return diff >= threshold


def extract_frames(video_path: str, output_dir: str, fps_target: float, max_frames: int,
                   resize_max: int, quality: int, diff_threshold: float,
                   start_sec: float | None, end_sec: float | None) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_video_frames / video_fps

    print(f"Video: {video_path}")
    print(f"  Duration: {duration_sec:.1f}s  |  FPS: {video_fps:.2f}  |  Total frames: {total_video_frames}")

    start_frame = int(start_sec * video_fps) if start_sec is not None else 0
    end_frame = int(end_sec * video_fps) if end_sec is not None else total_video_frames
    end_frame = min(end_frame, total_video_frames)

    stride = max(1, round(video_fps / fps_target))
    candidate_count = (end_frame - start_frame) // stride
    print(f"  Sampling every {stride} frames ({fps_target} fps target) → up to {candidate_count} candidates")
    print(f"  Redundancy threshold: {diff_threshold}/255 mean pixel diff")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved_count = 0
    prev_gray = None
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    pbar = tqdm(total=min(candidate_count, max_frames), desc="Extracting frames")

    while frame_idx < end_frame and saved_count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_if_needed(frame, resize_max)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and not frame_is_novel(frame_gray, prev_gray, diff_threshold):
            frame_idx += stride
            continue

        out_path = out_dir / f"frame_{saved_count + 1:05d}.jpg"
        cv2.imwrite(str(out_path), frame, encode_params)
        prev_gray = frame_gray
        saved_count += 1
        pbar.update(1)
        frame_idx += stride

    pbar.close()
    cap.release()

    print(f"\nExtracted {saved_count} frames → {out_dir}/")

    if saved_count < 30:
        print(f"WARNING: Only {saved_count} frames extracted. Consider lowering --diff-threshold or "
              f"increasing --fps-target for better COLMAP reconstruction.")
    elif saved_count > 150:
        print(f"WARNING: {saved_count} frames may slow down COLMAP. Consider lowering --fps-target "
              f"or raising --diff-threshold.")
    else:
        print(f"Frame count looks good for 3DGS ({saved_count} frames).")

    return saved_count


def main():
    args = parse_args()
    extract_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        fps_target=args.fps_target,
        max_frames=args.max_frames,
        resize_max=args.resize_max,
        quality=args.quality,
        diff_threshold=args.diff_threshold,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )


if __name__ == "__main__":
    main()
