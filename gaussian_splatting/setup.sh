#!/usr/bin/env bash
# One-shot environment setup for the NVS pipeline.
# Supports: Ubuntu 22.04 + CUDA 12.1, macOS (Apple Silicon / Intel)
#
# Usage:
#   bash setup.sh           # auto-detect platform
#   bash setup.sh --cpu     # force CPU-only (for testing)

set -euo pipefail

CPU_ONLY=false
if [[ "${1:-}" == "--cpu" ]]; then
    CPU_ONLY=true
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "========================================"
echo " Novel View Synthesis — Environment Setup"
echo " OS: $OS  ARCH: $ARCH"
echo "========================================"

# ----------------------------------------
# 1. Install COLMAP
# ----------------------------------------
echo ""
echo "[1/4] Checking COLMAP..."
if command -v colmap &>/dev/null; then
    echo "  COLMAP found: $(colmap --version 2>&1 | head -1)"
else
    if [[ "$OS" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            echo "  Installing COLMAP via Homebrew..."
            brew install colmap
        else
            echo "  ERROR: Homebrew not found. Install from https://brew.sh then run:"
            echo "    brew install colmap"
            exit 1
        fi
    elif [[ "$OS" == "Linux" ]]; then
        echo "  Installing COLMAP via apt..."
        sudo apt-get update -qq && sudo apt-get install -y colmap
    else
        echo "  WARNING: Cannot auto-install COLMAP on $OS."
        echo "  Install manually from: https://colmap.github.io/install.html"
    fi
fi

# ----------------------------------------
# 2. Python environment
# ----------------------------------------
echo ""
echo "[2/4] Setting up Python environment..."

if command -v conda &>/dev/null; then
    ENV_NAME="nvs"
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "  Conda env '${ENV_NAME}' already exists."
    else
        echo "  Creating conda env: ${ENV_NAME} (Python 3.10)"
        conda create -n "$ENV_NAME" python=3.10 -y
    fi
    echo ""
    echo "  NOTE: After setup, activate the environment:"
    echo "    conda activate ${ENV_NAME}"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    echo "  Using system Python: $(python3 --version)"
fi

# ----------------------------------------
# 3. Install PyTorch
# ----------------------------------------
echo ""
echo "[3/4] Installing PyTorch..."

if [[ "$OS" == "Darwin" ]]; then
    # macOS: standard pip install uses CPU + MPS (Metal Performance Shaders)
    echo "  macOS detected — installing PyTorch with MPS support"
    pip3 install torch torchvision torchaudio
elif $CPU_ONLY; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    # Linux: detect CUDA version
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        echo "  Detected CUDA ${CUDA_VER}"

        if [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 1 ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            GSPLAT_INDEX="https://docs.gsplat.studio/whl/pt21cu121"
        elif [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -ge 8 ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            GSPLAT_INDEX="https://docs.gsplat.studio/whl/pt21cu118"
        else
            echo "  Unsupported CUDA version ${CUDA_VER}. Defaulting to CUDA 12.1."
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            GSPLAT_INDEX="https://docs.gsplat.studio/whl/pt21cu121"
        fi
        pip3 install torch torchvision torchaudio --index-url "$TORCH_INDEX"
    else
        echo "  nvcc not found — installing CPU-only PyTorch"
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        GSPLAT_INDEX=""
    fi
fi

# ----------------------------------------
# 4. Install gsplat and dependencies
# ----------------------------------------
echo ""
echo "[4/4] Installing Python dependencies..."

if [[ "$OS" == "Darwin" ]]; then
    # macOS: build gsplat from source (no pre-compiled CUDA wheels)
    echo "  Installing gsplat from source (this may take a few minutes)..."
    pip3 install ninja  # speeds up compilation
    pip3 install git+https://github.com/nerfstudio-project/gsplat.git
elif [[ -n "${GSPLAT_INDEX:-}" ]]; then
    echo "  Installing gsplat pre-compiled wheel..."
    pip3 install gsplat --index-url "$GSPLAT_INDEX"
else
    pip3 install gsplat
fi

# Remaining dependencies
pip3 install \
    "pycolmap>=0.6.0" \
    "numpy<2.0.0" \
    "scipy>=1.11.0" \
    "opencv-python>=4.8.0" \
    "Pillow>=10.0.0" \
    "imageio>=2.31.0" \
    "imageio-ffmpeg>=0.4.9" \
    "torchmetrics[image]>=1.2.0" \
    "viser>=0.1.0" \
    "tqdm>=4.66.0" \
    "tyro>=0.8.8" \
    "PyYAML>=6.0.1" \
    "matplotlib>=3.7.0" \
    "tensorboard>=2.14.0"

# nerfview (optional)
pip3 install git+https://github.com/nerfstudio-project/nerfview.git 2>/dev/null || \
    echo "  nerfview install skipped (optional)"

# ----------------------------------------
# 5. Download gsplat example files (not included in pip wheel)
# ----------------------------------------
echo ""
echo "[5/5] Downloading gsplat example files..."
GSPLAT_VER=$(python3 -c "import gsplat; print(gsplat.__version__)" 2>/dev/null || echo "main")
GSPLAT_TAG="v${GSPLAT_VER}"
BASE_URL="https://raw.githubusercontent.com/nerfstudio-project/gsplat/${GSPLAT_TAG}/examples"

mkdir -p third_party/gsplat_examples/datasets
touch third_party/__init__.py third_party/gsplat_examples/__init__.py

for FILE in simple_trainer.py utils.py; do
    curl -sfL "${BASE_URL}/${FILE}" -o "third_party/gsplat_examples/${FILE}" && echo "  OK: ${FILE}" || echo "  WARN: failed to download ${FILE}"
done
touch third_party/gsplat_examples/datasets/__init__.py  # not in repo, create empty
for FILE in colmap.py normalize.py traj.py; do
    curl -sfL "${BASE_URL}/datasets/${FILE}" -o "third_party/gsplat_examples/datasets/${FILE}" && echo "  OK: datasets/${FILE}" || echo "  WARN: failed to download datasets/${FILE}"
done

# ----------------------------------------
# Verification
# ----------------------------------------
echo ""
echo "Verifying installation..."
python3 - <<'EOF'
import sys
ok = True
for mod in ["torch", "gsplat", "pycolmap", "cv2", "numpy", "scipy", "imageio"]:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  {mod}: {ver}")
    except ImportError as e:
        print(f"  MISSING: {mod} — {e}")
        ok = False

import torch
print(f"  CUDA available:  {torch.cuda.is_available()}")
print(f"  MPS available:   {torch.backends.mps.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

if ok:
    print("\nSetup complete!")
else:
    print("\nSome packages failed to install. Check errors above.")
    sys.exit(1)
EOF

echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " Quick start:"
echo "   bash scripts/run_pipeline.sh nvs_example_input_video.mp4"
echo "========================================"
