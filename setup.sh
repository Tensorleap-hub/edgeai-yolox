#!/usr/bin/env bash

# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################################

# conda environment settings (override with ENV_NAME / PYTHON_VERSION / CUDA_VERSION)
ENV_NAME=${ENV_NAME:-edgeai-yolox}
PYTHON_VERSION=${PYTHON_VERSION:-3.9}
CUDA_VERSION=${CUDA_VERSION:-cu118}
TORCH_VERSION=${TORCH_VERSION:-2.0.1}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.15.2}

######################################################################
# ensure conda is available and activate env
if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but not found on PATH. Please install Miniconda/Anaconda first."
    exit 1
fi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -qw "$ENV_NAME"; then
    echo "Using existing conda env: $ENV_NAME"
else
    echo "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

######################################################################
# system packages (best-effort; skip if apt-get not available)
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get install -y libjpeg-dev zlib1g-dev cmake libffi-dev protobuf-compiler
else
    echo "apt-get not found; please install libjpeg-dev zlib1g-dev cmake libffi-dev protobuf-compiler manually if missing"
fi

######################################################################
# upgrade pip inside the env
python -m pip install --no-input --upgrade pip setuptools

######################################################################
echo "Installing PyTorch with CUDA ($CUDA_VERSION)"
python -m pip install --no-input \
    torch=="${TORCH_VERSION}+${CUDA_VERSION}" \
    torchvision=="${TORCHVISION_VERSION}+${CUDA_VERSION}" \
    -f https://download.pytorch.org/whl/torch_stable.html

echo 'Installing python packages...'
# there is an issue with installing pillow-simd through requirements - force it here
python -m pip uninstall --yes pillow
python -m pip install --no-input -U --force-reinstall pillow-simd
python -m pip install --no-input cython wheel numpy==1.23.0
python -m pip install --no-input torchinfo pycocotools opencv-python

echo "Installing requirements"
python -m pip install --no-input -r requirements.txt

######################################################################
echo "Installing mmcv"
python -m pip install --no-input mmcv-full==1.4.8 -f "https://download.openmmlab.com/mmcv/dist/${CUDA_VERSION}/torch${TORCH_VERSION}/index.html"

######################################################################
# pinned protobuf/onnx versions
python -m pip install --no-input protobuf==3.20.2 onnx==1.13.0

######################################################################
echo 'Installing the python package in editable mode...'
python setup.py develop

echo "Done. Activate with: conda activate $ENV_NAME"
