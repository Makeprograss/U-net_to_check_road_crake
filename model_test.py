import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import UNet
from torchvision import transforms
from PIL import Image
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt

# 使用与训练相同的数据集类
class RoadCrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 获取所有图像文件
        self.images = sorted(glob(os.path.join(image_dir, '*.*')))
        self.masks = sorted(glob(os.path.join(mask_dir, '*.*')))

        assert len(self.images) == len(self.masks), f"图像数量({len(self.images)})与mask数量({len(self.masks)})不匹配"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图像
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 转换为灰度图

        # 应用transforms
        if self.transform:
            image = self.transform(image)
            # mask也需要resize，但不需要normalize
            mask = transforms.Resize((224, 224))(mask)
            mask = transforms.ToTensor()(mask)
            # 将mask转换为类别索引 (0或1)
            mask = (mask > 0.5).long().squeeze(0)  # [H, W]

        return image, mask, img_path

# 计算IoU (Intersection over Union)
def calculate_iou(pred, target, num_classes=2):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())

    return ious  # 返回每个类别的IoU

def test_data_process():
    # 定义数据集的路径
    test_image_path = 'data/test/images'
    test_mask_path = 'data/test/masks'

    normalize = transforms.Normalize(mean=[0.15620959,0.16285059,0.16683851],std=[0.04152462,0.04320607,0.04479365])

    # 定义数据集处理方法变量（与训练时相同）
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = RoadCrackDataset(test_image_path, test_mask_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

    print(f"测试集数据大小: {len(test_dataset)} 张图片")
    return test_loader

def test_model_process(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    total_iou_background = 0.0
    total_iou_crack = 0.0
    test_num = 0

    print("\n开始测试...")
    with torch.no_grad():
        for test_data_x, test_data_y, _ in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 前向传播，输出为 [B, C, H, W]
            output = model(test_data_x)
            # 获取预测结果 [B, H, W]
            pre_mask = torch.argmax(output, dim=1)

            # 计算IoU
            ious = calculate_iou(pre_mask, test_data_y)
            total_iou_background += ious[0] if not np.isnan(ious[0]) else 0
            total_iou_crack += ious[1] if not np.isnan(ious[1]) else 0
            test_num += 1

    # 计算平均IoU
    avg_iou_background = total_iou_background / test_num
    avg_iou_crack = total_iou_crack / test_num
    avg_iou = (avg_iou_background + avg_iou_crack) / 2

    print("\n" + "="*50)
    print("测试结果:")
    print(f"  背景类IoU: {avg_iou_background:.4f}")
    print(f"  裂缝类IoU: {avg_iou_crack:.4f}")
    print(f"  平均IoU: {avg_iou:.4f}")
    print("="*50)

    return avg_iou

def visualize_predictions(model, test_dataloader, num_samples=5, save_dir='test_results'):
    """可视化预测结果"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n生成 {num_samples} 个可视化样本...")

    with torch.no_grad():
        for idx, (image, mask, img_path) in enumerate(test_dataloader):
            if idx >= num_samples:
                break

            image = image.to(device)
            mask = mask.to(device)

            # 预测
            output = model(image)
            pred_mask = torch.argmax(output, dim=1)

            # 转换为numpy用于可视化
            # image shape: [1, C, H, W] -> squeeze(0) -> [C, H, W] -> permute -> [H, W, C]
            image_np = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
            # 反标准化
            mean = np.array([0.15804266, 0.16457426, 0.16825973])
            std = np.array([0.04205786, 0.04393576, 0.04547899])
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)

            # mask shape: [1, H, W] -> squeeze(0) -> [H, W]
            mask_np = mask.cpu().squeeze(0).numpy()
            pred_mask_np = pred_mask.cpu().squeeze(0).numpy()

            # 计算IoU
            ious = calculate_iou(pred_mask, mask)

            # 绘制
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image_np)
            axes[0].set_title('原始图像')
            axes[0].axis('off')

            axes[1].imshow(mask_np, cmap='gray')
            axes[1].set_title('真实标注')
            axes[1].axis('off')

            axes[2].imshow(pred_mask_np, cmap='gray')
            axes[2].set_title(f'预测结果\nIoU: {np.nanmean(ious):.3f}')
            axes[2].axis('off')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f'result_{idx+1}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()

            print(f"  样本 {idx+1}/{num_samples} - IoU: {np.nanmean(ious):.4f} - 已保存到 {save_path}")

    print(f"\n可视化结果已保存到: {save_dir}")

if __name__ == "__main__":
    print("="*50)
    print("U-Net 道路裂缝分割模型测试")
    print("="*50)

    # 加载U-Net模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")

    model = UNet(n_channels=3, n_classes=2)

    # 加载训练好的模型权重
    model_path = 'best_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型: {model_path}")
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        exit(1)

    model = model.to(device)

    # 加载测试数据
    test_dataloader = test_data_process()

    # 执行测试
    avg_iou = test_model_process(model, test_dataloader)

    # 可视化部分预测结果
    visualize_predictions(model, test_dataloader, num_samples=10)

    print("\n测试完成!")