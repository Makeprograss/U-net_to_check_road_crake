#!/bin/bash

# Linux启动脚本
# 道路裂缝检测系统

echo "========================================"
echo "  道路裂缝检测系统 - Linux版"
echo "========================================"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null
then
    echo "错误: 未找到Python3，请先安装Python 3.7+"
    exit 1
fi

echo "检查依赖包..."

# 检查并安装gradio
if ! python3 -c "import gradio" 2>/dev/null; then
    echo "安装 Gradio..."
    pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# 检查其他依赖
if ! python3 -c "import torch" 2>/dev/null; then
    echo "错误: 未找到PyTorch，请先安装"
    exit 1
fi

if ! python3 -c "import cv2" 2>/dev/null; then
    echo "安装 OpenCV..."
    pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

echo ""
echo "启动应用..."
echo "访问地址: http://localhost:7860"
echo "按 Ctrl+C 停止服务"
echo ""

# 启动应用
cd "$(dirname "$0")"
python3 app.py