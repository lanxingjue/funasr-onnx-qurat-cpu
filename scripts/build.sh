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