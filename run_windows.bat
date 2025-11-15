@echo off
chcp 65001 >nul
REM Windows启动脚本
REM 道路裂缝检测系统

echo ========================================
echo   道路裂缝检测系统 - Windows版
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python 3.7+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo 检查依赖包...

REM 检查并安装gradio
python -c "import gradio" >nul 2>&1
if %errorlevel% neq 0 (
    echo 安装 Gradio...
    pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
)

REM 检查PyTorch
python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到PyTorch，请先安装
    echo 安装命令: pip install torch torchvision
    pause
    exit /b 1
)

REM 检查OpenCV
python -c "import cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo 安装 OpenCV...
    pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo.
echo 启动应用...
echo 访问地址: http://localhost:7860
echo 按 Ctrl+C 停止服务
echo.

REM 启动应用
cd /d "%~dp0"
python app.py

pause