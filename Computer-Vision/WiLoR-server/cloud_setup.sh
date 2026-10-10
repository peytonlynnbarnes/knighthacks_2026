#!/usr/bin/env bash
# One-time WiLoR server setup on the cloud GPU machine (AMD Developer Cloud, MI300X).
# Pick a machine image that already has ROCm PyTorch, copy this repo's
# Computer-Vision folder over, then run on the cloud machine:
#     bash Computer-Vision/cloud_setup.sh
# It makes ~/wilor-venv (reusing the image's ROCm PyTorch), installs WiLoR-mini,
# downloads the models (~2.5 GB) and checks WiLoR runs on the GPU.
set -euo pipefail
VENV="${VENV:-$HOME/wilor-venv}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# ROCm PyTorch images often keep PyTorch in a conda env that an SSH login doesn't
# activate, so look for the Python that has it (or pass PYTHON=/path/to/python).
find_python() {
    for py in "${PYTHON:-}" python3 python /opt/conda/envs/*/bin/python /opt/conda/bin/python \
              /opt/venv/bin/python /usr/local/bin/python3; do
        [ -n "$py" ] && command -v "$py" >/dev/null 2>&1 || continue
        "$py" -c "import torch, torchvision" >/dev/null 2>&1 && { command -v "$py"; return 0; }
    done
    return 1
}
PY="$(find_python)" || {
    echo "No Python with torch + torchvision found. In a notebook cell run"
    echo "    import sys; print(sys.executable)"
    echo "and re-run with that path:  PYTHON=<that path> bash $0"
    exit 1
}
echo "using $PY"

# The image's PyTorch must see the GPU and come with torchvision, or pip would pull a
# CUDA build from PyPI in their place.
"$PY" - <<'EOF' || exit 1
import sys
try:
    import torch, torchvision
except ImportError as e:
    sys.exit(f"{e}: use a ROCm PyTorch image, or install torch+torchvision from "
             "https://download.pytorch.org/whl/rocm<version> first")
if not torch.cuda.is_available():
    sys.exit("PyTorch sees no GPU (is this the GPU machine / a ROCm build?)")
print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | HIP {torch.version.hip}")
EOF

if [ ! -x "$VENV/bin/python" ]; then
    "$PY" -m venv --system-site-packages "$VENV" \
        || { echo "venv failed: sudo apt install python3-venv, then re-run"; exit 1; }
fi
PIP="$VENV/bin/pip"
"$PIP" install -q --upgrade pip
"$PIP" install -q --no-deps git+https://github.com/warmshao/WiLoR-mini   # its torch<=2.5 pin is too old
"$PIP" install -q smplx==0.1.28 timm einops ultralytics==8.1.34 huggingface_hub scikit-image roma \
    opencv-python-headless dill fastapi uvicorn
"$PIP" install -q --no-build-isolation git+https://github.com/mattloper/chumpy

cd "$HERE"
"$VENV/bin/python" -c "
import torch
from wilor_pub import WiLoRTracker
ms = WiLoRTracker().warm_up(max_crops=4, max_views=2)
print(f'WiLoR ready on {torch.cuda.get_device_name(0)} (warm-up incl. kernel tuning {ms / 1000:.0f} s)')
"
echo
echo "Start the server (keep it running, e.g. inside tmux):"
echo "    cd $HERE && WILOR_TOKEN=<secret> $VENV/bin/python wilor_server.py"