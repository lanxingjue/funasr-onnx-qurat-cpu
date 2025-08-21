#!/bin/bash
set -e

echo "🔨 构建FunASR GPU引擎..."

# 检查依赖
python3 -c "import funasr, torch; print('✅ FunASR和PyTorch已安装')"
python3 -c "import torch; assert torch.cuda.is_available(); print('✅ CUDA可用')"

# 安装系统依赖
sudo apt-get update
sudo apt-get install -y build-essential cmake libpython3-dev python3-pybind11

# 构建
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo "✅ 构建完成: build/funasr_gpu_engine"
