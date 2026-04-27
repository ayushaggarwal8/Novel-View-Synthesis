#!/usr/bin/env bash
# End-to-end NVS pipeline: video → novel views
# Run from the project root: bash gaussian_splatting/scripts/run_pipeline.sh <video>
# NOTE: Requires a CUDA GPU. Steps 4-5 will fail on macOS/CPU.

set -euo pipefail

VIDEO="${1:-}"
if [[ -z "$VIDEO" ]]; then
    echo "Usage: bash scripts/run_pipeline.sh <path_to_video.mp4>"
    echo ""
    echo "Optional env vars:"
    echo "  FPS_TARGET=0.5       frames per second to extract"
    echo "  MAX_FRAMES=150       cap on extracted frames"
    echo "  MAX_STEPS=7000       3DGS training iterations"
    echo "  DATA_FACTOR=1        image downscale factor (use 2 if CUDA OOM)"
    exit 1
fi

FPS_TARGET="${FPS_TARGET:-0.5}"
MAX_FRAMES="${MAX_FRAMES:-150}"
MAX_STEPS="${MAX_STEPS:-7000}"
DATA_FACTOR="${DATA_FACTOR:-1}"

FRAMES_DIR="data/frames"
WORKSPACE="data/colmap_workspace"
GSPLAT_INPUT="data/gsplat_input"
RESULT_DIR="outputs"
NOVEL_VIEWS_DIR="outputs/novel_views"

echo "========================================"
echo " Novel View Synthesis Pipeline"
echo "========================================"
echo "  Video:       $VIDEO"
echo "  FPS target:  $FPS_TARGET"
echo "  Max frames:  $MAX_FRAMES"
echo "  Train steps: $MAX_STEPS"
echo "========================================"

echo ""
echo "[1/5] Extracting frames..."
python gaussian_splatting/scripts/01_extract_frames.py \
    --video "$VIDEO" \
    --output-dir "$FRAMES_DIR" \
    --fps-target "$FPS_TARGET" \
    --max-frames "$MAX_FRAMES"

echo ""
echo "[2/5] Running COLMAP SfM..."
python gaussian_splatting/scripts/02_run_colmap.py \
    --frames-dir "$FRAMES_DIR" \
    --workspace "$WORKSPACE" \
    --gsplat-input "$GSPLAT_INPUT"

echo ""
echo "[3/5] Preparing gsplat data..."
python gaussian_splatting/scripts/03_prepare_gsplat_data.py \
    --gsplat-input "$GSPLAT_INPUT"

echo ""
echo "[4/5] Training 3D Gaussian Splatting..."
python gaussian_splatting/scripts/04_train_gsplat.py \
    --data-dir "$GSPLAT_INPUT" \
    --result-dir "$RESULT_DIR" \
    --max-steps "$MAX_STEPS" \
    --data-factor "$DATA_FACTOR"

echo ""
echo "[5/5] Rendering novel views..."
python gaussian_splatting/scripts/05_render_novel_views.py \
    --gsplat-input "$GSPLAT_INPUT" \
    --result-dir "$RESULT_DIR" \
    --output-dir "$NOVEL_VIEWS_DIR"

echo ""
echo "========================================"
echo " Pipeline complete!"
echo " Novel views: $NOVEL_VIEWS_DIR/"
echo "========================================"
