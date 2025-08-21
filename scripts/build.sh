#!/bin/bash

# 设置错误时退出 (兼容写法)
if ! set -e 2>/dev/null; then
    # 如果set -e不支持，使用手动错误检查
    echo "注意: 使用手动错误检查模式"
    ERROR_CHECK=true
else
    ERROR_CHECK=false
fi

# 错误检查函数
check_error() {
    if [ $? -ne 0 ]; then
        echo "❌ 错误: $1"
        exit 1
    fi
}

echo "🔨 构建FunASR GPU引擎..."

# 检查依赖
echo "📋 检查依赖..."

# 检查Python和必要包
python3 -c "import sys; print('Python版本:', sys.version)"
check_error "Python3不可用"

python3 -c "import funasr; print('✅ FunASR已安装，版本:', funasr.__version__ if hasattr(funasr, '__version__') else '未知')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ FunASR未安装，开始安装..."
    pip3 install -U funasr
    check_error "FunASR安装失败"
fi

python3 -c "import torch; print('✅ PyTorch已安装，版本:', torch.__version__)" 2>/dev/null
check_error "PyTorch未安装，请手动安装"

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA不可用'; print('✅ CUDA可用，GPU数量:', torch.cuda.device_count())"
check_error "CUDA不可用，请检查PyTorch GPU版本"

# 检查编译工具
echo "🔧 检查编译环境..."

if ! command -v g++ >/dev/null 2>&1; then
    echo "❌ g++编译器未找到"
    echo "Ubuntu/Debian: sudo apt-get install build-essential"
    echo "CentOS/RHEL: sudo yum groupinstall 'Development Tools'"
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "❌ cmake未找到"
    echo "Ubuntu/Debian: sudo apt-get install cmake"
    echo "CentOS/RHEL: sudo yum install cmake"
    exit 1
fi

# 检查系统依赖并安装
echo "📦 检查和安装系统依赖..."

# 检测操作系统
if command -v apt-get >/dev/null 2>&1; then
    PKG_MANAGER="apt-get"
    PKG_UPDATE="sudo apt-get update"
    PKG_INSTALL="sudo apt-get install -y"
elif command -v yum >/dev/null 2>&1; then
    PKG_MANAGER="yum"
    PKG_UPDATE="sudo yum update"
    PKG_INSTALL="sudo yum install -y"
elif command -v dnf >/dev/null 2>&1; then
    PKG_MANAGER="dnf"
    PKG_UPDATE="sudo dnf update"
    PKG_INSTALL="sudo dnf install -y"
else
    echo "⚠️ 未识别的包管理器，请手动安装依赖"
    PKG_MANAGER="manual"
fi

if [ "$PKG_MANAGER" != "manual" ]; then
    echo "使用包管理器: $PKG_MANAGER"
    
    # 更新包列表
    echo "更新包列表..."
    $PKG_UPDATE
    
    # 安装必要依赖
    if [ "$PKG_MANAGER" = "apt-get" ]; then
        $PKG_INSTALL build-essential cmake libpython3-dev python3-pybind11 pkg-config
    elif [ "$PKG_MANAGER" = "yum" ] || [ "$PKG_MANAGER" = "dnf" ]; then
        $PKG_INSTALL gcc gcc-c++ cmake python3-devel python3-pybind11 pkgconfig
    fi
    
    check_error "系统依赖安装失败"
fi

# 检查pybind11
echo "🔗 检查pybind11..."
python3 -c "import pybind11; print('✅ pybind11已安装，版本:', pybind11.__version__)"
if [ $? -ne 0 ]; then
    echo "安装pybind11..."
    pip3 install pybind11
    check_error "pybind11安装失败"
fi

# 创建构建目录
echo "📁 创建构建目录..."
if [ -d "build" ]; then
    echo "清理旧的构建目录..."
    rm -rf build
fi
mkdir build
check_error "创建构建目录失败"

cd build
check_error "进入构建目录失败"

# CMake配置
echo "⚙️ 配置CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-O3 -Wall -Wextra"

check_error "CMake配置失败"

# 检查CPU核数
NPROC=$(nproc 2>/dev/null || echo 4)
echo "🔧 开始编译 (使用 $NPROC 个并发)..."

# 编译
make -j$NPROC
check_error "编译失败"

# 检查编译结果
if [ -f "funasr_gpu_engine" ]; then
    echo "✅ 编译成功！"
    echo "📁 可执行文件: $(pwd)/funasr_gpu_engine"
    
    # 检查可执行文件
    file funasr_gpu_engine
    echo "文件大小: $(du -h funasr_gpu_engine | cut -f1)"
else
    echo "❌ 编译失败 - 可执行文件未生成"
    exit 1
fi

# 回到项目根目录
cd ..

# 创建必要目录
echo "📁 创建运行时目录..."
mkdir -p logs
mkdir -p reports

# 检查音频目录
if [ ! -d "audio_files" ]; then
    echo "⚠️ 音频目录 'audio_files' 不存在"
    echo "请创建音频目录并放入WAV文件:"
    echo "  mkdir audio_files"
    echo "  cp /path/to/your/*.wav audio_files/"
else
    WAV_COUNT=$(ls audio_files/*.wav 2>/dev/null | wc -l)
    echo "✅ 音频目录存在，包含 $WAV_COUNT 个WAV文件"
fi

echo ""
echo "🎉 构建完成！"
echo ""
echo "使用方法:"
echo "  cd build"
echo "  ./funasr_gpu_engine --help"
echo ""
echo "快速测试:"
echo "  ./build/funasr_gpu_engine --test-offline-only --max-files 3"
echo ""
echo "完整测试:"
echo "  ./build/funasr_gpu_engine --max-files 50 --concurrent 4"
