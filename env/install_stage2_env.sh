#!/bin/bash
set -e

echo "======================================"
echo "Stage2 Bridge Environment Installation"
echo "======================================"

PROJECT_DIR=~/work/stage2_bridge
ENV_NAME=fd4mm_stage2

echo "Project dir: $PROJECT_DIR"

cd $PROJECT_DIR

################################
# 1 Create conda environment
################################

echo "Creating conda environment..."

conda create -y -n $ENV_NAME python=3.10

echo "Activating environment..."

source activate $ENV_NAME || conda activate $ENV_NAME

################################
# 2 Install PyTorch + CUDA
################################

echo "Installing PyTorch..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

################################
# 3 Basic dependencies
################################

echo "Installing basic python packages..."

pip install \
numpy \
scipy \
opencv-python \
matplotlib \
tqdm \
scikit-image \
imageio \
imageio-ffmpeg \
einops \
pillow

################################
# 4 Clone SAM2
################################

echo "Installing SAM2..."

mkdir -p external
cd external

if [ ! -d "sam2" ]; then
    git clone https://github.com/facebookresearch/sam2.git
fi

cd sam2

pip install -e .

################################
# 5 Download SAM2 checkpoint
################################

cd $PROJECT_DIR

mkdir -p models
cd models

if [ ! -f "sam2_hiera_small.pt" ]; then
    echo "Downloading SAM2 checkpoint..."
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
fi

################################
# 6 Test installation
################################

echo "Testing RAFT..."

python - <<'PY'
from torchvision.models.optical_flow import raft_large
print("RAFT OK")
PY

echo "Testing SAM2..."

python - <<'PY'
from sam2.build_sam import build_sam2_video_predictor
print("SAM2 OK")
PY

################################
# Done
################################

echo ""
echo "======================================"
echo "Environment installation complete!"
echo "Activate with:"
echo "conda activate $ENV_NAME"
echo "======================================"