#!/bin/bash

echo "🚀 运行FunASR GPU引擎..."

# 检查可执行文件
if [ ! -f "build/funasr_gpu_engine" ]; then
    echo "❌ 可执行文件不存在"
    echo "请先运行: scripts/build.sh"
    exit 1
fi

# 检查音频文件
if [ ! -d "audio_files" ]; then
    echo "❌ 音频目录 'audio_files' 不存在"
    echo "请创建目录并添加WAV文件:"
    echo "  mkdir audio_files"
    echo "  cp /path/to/your/*.wav audio_files/"
    exit 1
fi

WAV_COUNT=$(ls audio_files/*.wav 2>/dev/null | wc -l)
if [ "$WAV_COUNT" -eq 0 ]; then
    echo "❌ 音频目录中没有WAV文件"
    echo "请添加WAV文件到 audio_files/ 目录"
    exit 1
fi

echo "✅ 找到 $WAV_COUNT 个WAV文件"

# 创建必要目录
mkdir -p logs

# 检查GPU状态
echo "📊 检查GPU状态..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
else
    echo "⚠️ nvidia-smi不可用，无法检查GPU状态"
fi

echo ""
echo "开始运行测试..."

# 运行引擎
cd build
./funasr_gpu_engine --gpu-id 0 --max-files 20 --concurrent 2

echo ""
echo "🎉 测试完成!"
echo "📄 查看详细日志: logs/"
echo "📊 性能报告: funasr_gpu_performance_report.txt"
