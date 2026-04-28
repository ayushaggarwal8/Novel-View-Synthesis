#!/usr/bin/env bash
# Setup for the diffusion NVS pipeline.
# Supports Apple Silicon (M2/MPS) and Linux/CUDA.
# Run from the project root: bash diffusion_nvs/setup.sh

set -euo pipefail

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "========================================"
echo " Diffusion NVS — Environment Setup"
echo " OS: $OS  ARCH: $ARCH"
echo "========================================"

# ----------------------------------------
# 1. Install PyTorch
# ----------------------------------------
echo ""
echo "[1/3] Installing PyTorch..."

if [[ "$OS" == "Darwin" ]]; then
    echo "  macOS — installing PyTorch with MPS (Metal) support"
    pip3 install torch torchvision torchaudio
else
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        echo "  Detected CUDA ${CUDA_VER} — installing CUDA-enabled PyTorch"
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "  No CUDA detected — installing CPU-only PyTorch"
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# ----------------------------------------
# 2. Install diffusion dependencies
# ----------------------------------------
echo ""
echo "[2/3] Installing diffusion dependencies..."
pip3 install -r diffusion_nvs/requirements.txt

# ----------------------------------------
# 3. Verify
# ----------------------------------------
echo ""
echo "[3/3] Verifying installation..."
python3 - <<'EOF'
import sys
ok = True
packages = ["torch", "diffusers", "transformers", "accelerate", "cv2", "PIL", "numpy", "tqdm"]
for mod in packages:
    try:
        import importlib
        m = importlib.import_module(mod)
        print(f"  OK  {mod} {getattr(m, '__version__', '')}")
    except ImportError as e:
        print(f"  MISSING  {mod} — {e}")
        ok = False

import torch
print(f"\n  CUDA available : {torch.cuda.is_available()}")
print(f"  MPS  available : {torch.backends.mps.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")

if not ok:
    print("\nSome packages failed to install — check errors above.")
    sys.exit(1)

print("\nSetup complete!")
print("Run the pipeline with:")
print("  bash diffusion_nvs/scripts/run_pipeline.sh")
EOF
