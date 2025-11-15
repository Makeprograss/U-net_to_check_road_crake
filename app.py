"""
道路裂缝检测 Web 应用
支持 Linux 和 Windows 平台
使用 Gradio 创建交互式界面
"""

import gradio as gr
import torch
import torch.nn as nn
from model import UNet
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

class CrackDetectionApp:
    def __init__(self, model_path='U-net_to_check_road_crake/best_model.pth'):
        """初始化应用"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

        # 加载模型
        self.model = UNet(n_channels=3, n_classes=2)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"成功加载模型: {model_path}")
        else:
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # 数据预处理
        self.normalize = transforms.Normalize(
            mean=[0.15804266, 0.16457426, 0.16825973],
            std=[0.04205786, 0.04393576, 0.04547899]
        )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize
        ])

    def preprocess_image(self, image):
        """预处理输入图像"""
        # 转换为RGB（防止RGBA或灰度图）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 保存原始尺寸
        original_size = image.size  # (width, height)

        # 应用transform
        image_tensor = self.transform(image).unsqueeze(0)  # [1, 3, 224, 224]

        return image_tensor, original_size

    def predict(self, image_tensor):
        """执行预测"""
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)  # [1, 2, 224, 224]
            pred_mask = torch.argmax(output, dim=1)  # [1, 224, 224]

        return pred_mask.cpu().squeeze(0).numpy()  # [224, 224]

    def create_overlay(self, original_image, pred_mask, alpha=0.5):
        """创建裂缝检测叠加图"""
        # 将原图resize到224x224用于叠加
        original_resized = original_image.resize((224, 224))
        original_np = np.array(original_resized)

        # 创建彩色mask（红色表示裂缝）
        color_mask = np.zeros_like(original_np)
        color_mask[pred_mask == 1] = [255, 0, 0]  # 红色标记裂缝

        # 叠加
        overlay = cv2.addWeighted(original_np, 1-alpha, color_mask, alpha, 0)

        return overlay

    def calculate_crack_percentage(self, pred_mask):
        """计算裂缝占比"""
        total_pixels = pred_mask.size
        crack_pixels = np.sum(pred_mask == 1)
        percentage = (crack_pixels / total_pixels) * 100
        return percentage

    def process_image(self, input_image, show_overlay=True, overlay_alpha=0.5):
        """处理上传的图像"""
        if input_image is None:
            return None, None, "请上传图像"

        # 预处理
        image_tensor, original_size = self.preprocess_image(input_image)

        # 预测
        pred_mask = self.predict(image_tensor)

        # 计算裂缝占比
        crack_percentage = self.calculate_crack_percentage(pred_mask)

        # 创建可视化结果
        mask_vis = (pred_mask * 255).astype(np.uint8)  # 转换为0-255

        # 创建叠加图
        if show_overlay:
            overlay_image = self.create_overlay(input_image, pred_mask, overlay_alpha)
        else:
            overlay_image = None

        # 生成报告
        if crack_percentage > 0.1:
            severity = "检测到裂缝"
            if crack_percentage > 10:
                severity += " - 严重"
            elif crack_percentage > 5:
                severity += " - 中等"
            else:
                severity += " - 轻微"
        else:
            severity = "未检测到明显裂缝"

        report = f"""
### 检测结果

- **裂缝占比**: {crack_percentage:.2f}%
- **严重程度**: {severity}
- **图像尺寸**: {original_size[0]} × {original_size[1]}
- **检测分辨率**: 224 × 224
        """

        return mask_vis, overlay_image, report

def create_gradio_interface():
    """创建Gradio界面"""
    # 初始化应用
    app = CrackDetectionApp()

    # 创建界面
    with gr.Blocks(title="道路裂缝检测系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🛣️ 道路裂缝检测系统

            基于 U-Net 深度学习模型的道路裂缝自动检测系统

            **使用方法**：上传道路图片，系统将自动检测并标注裂缝位置
            """
        )

        with gr.Row():
            with gr.Column():
                # 输入
                input_image = gr.Image(type="pil", label="上传道路图片")

                with gr.Accordion("高级设置", open=False):
                    show_overlay = gr.Checkbox(value=True, label="显示叠加图")
                    overlay_alpha = gr.Slider(0, 1, value=0.5, step=0.1, label="叠加透明度")

                predict_btn = gr.Button("🔍 开始检测", variant="primary")

                gr.Examples(
                    examples=[],  # 可以添加示例图片路径
                    inputs=input_image,
                    label="示例图片"
                )

            with gr.Column():
                # 输出
                output_mask = gr.Image(label="检测结果（二值mask）")
                output_overlay = gr.Image(label="叠加可视化")
                output_report = gr.Markdown(label="检测报告")

        # 绑定事件
        predict_btn.click(
            fn=app.process_image,
            inputs=[input_image, show_overlay, overlay_alpha],
            outputs=[output_mask, output_overlay, output_report]
        )

        gr.Markdown(
            """
            ---
            ### 📊 系统信息

            - **模型**: U-Net (Deep Learning)
            - **输入分辨率**: 224×224
            - **类别**: 背景 / 裂缝
            - **评估指标**: IoU (Intersection over Union)

            ### ⚠️ 注意事项

            1. 上传清晰的道路图片以获得最佳效果
            2. 图片会自动缩放到224×224进行检测
            3. 红色区域表示检测到的裂缝
            4. 裂缝占比仅供参考，建议结合人工复核
            """
        )

    return demo

if __name__ == "__main__":
    # 创建界面
    demo = create_gradio_interface()

    # 启动应用
    print("\n" + "="*50)
    print("启动道路裂缝检测系统...")
    print("="*50 + "\n")

    # 在本地启动，支持局域网访问
    demo.launch(
        server_name="0.0.0.0",  # 允许局域网访问
        server_port=7860,        # 端口号
        share=True,             # 设置为True可以生成公网链接
        show_error=True
    )