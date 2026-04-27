#!/usr/bin/env bash
# Diffusion NVS pipeline — run from project root
# Usage: bash diffusion_nvs/scripts/run_pipeline.sh
# Requires: data/frames/ to exist (run 01_extract_frames.py first if not)

set -euo pipefail

N_INPUT="${N_INPUT:-25}"         # number of input frames to select
N_VIEWS="${N_VIEWS:-4}"          # novel views per input image
STEPS="${STEPS:-75}"             # diffusion inference steps
GUIDANCE="${GUIDANCE:-3.0}"      # classifier-free guidance scale

echo "========================================"
echo " Diffusion NVS Pipeline (Stable Zero123)"
echo "========================================"
echo "  Input frames:  $N_INPUT"
echo "  Novel views:   $N_VIEWS per image"
echo "  Steps:         $STEPS"
echo "========================================"

# Ensure frames exist
if [ ! -d "data/frames" ] || [ -z "$(ls -A data/frames 2>/dev/null)" ]; then
    echo ""
    echo "No frames found. Extracting from video..."
    python gaussian_splatting/scripts/01_extract_frames.py \
        --video nvs_example_input_video.mp4 \
        --output-dir data/frames \
        --fps-target 0.5
fi

echo ""
echo "[1/3] Selecting diverse input frames..."
python diffusion_nvs/scripts/01_select_input_frames.py \
    --frames-dir data/frames \
    --output-dir diffusion_nvs/inputs \
    --n-frames "$N_INPUT"

echo ""
echo "[2/3] Generating novel views..."
python diffusion_nvs/scripts/02_generate_novel_views.py \
    --input-dir diffusion_nvs/inputs \
    --output-dir diffusion_nvs/outputs/novel_views \
    --n-views "$N_VIEWS" \
    --steps "$STEPS" \
    --guidance-scale "$GUIDANCE"

echo ""
echo "[3/3] Creating comparison grids..."
python diffusion_nvs/scripts/03_make_grid.py \
    --output-dir diffusion_nvs/outputs/novel_views \
    --grid-dir diffusion_nvs/outputs/grids

echo ""
echo "========================================"
echo " Done!"
echo " Novel views: diffusion_nvs/outputs/novel_views/"
echo " Grids:       diffusion_nvs/outputs/grids/"
echo "========================================"
