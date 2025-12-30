#!/usr/bin/env bash
set -e 
ENV_NAME=pie_diabetico
PYTHON_VERSION=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "Verificando entorno Conda: $ENV_NAME"


if conda env list | grep -q "^$ENV_NAME "; then
    echo " El entorno '$ENV_NAME' YA EXISTE, activando ... "
else
    echo "Creando entorno de conde: $ENV_NAME"
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi
conda activate $ENV_NAME

echo "Python version: "
python --version

pip install --upgrade pip setuptools wheel

echo "Instalando PyTorch 2.0.1 + CUDA 11.8"
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

echo "Verificando instalacion"

python -<< EOF 
import torch 
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

#Instalar las demas dependencias 
pip install -e .
echo "Entorno  '$ENV_NAME' Listo "
