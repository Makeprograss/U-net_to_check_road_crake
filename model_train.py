import argparse
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from model import UNet
from model import DoubleConv
from torchvision import transforms
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import torch.utils.data as Data
import os
from PIL import Image
import numpy as np
from glob import glob

# Dice Loss 实现（对小目标和细节更敏感）
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        """
        Dice Loss for segmentation
        Args:
            smooth: 平滑系数，避免分母为0
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: 模型输出 [B, C, H, W] (C=2，背景和裂缝)
            target: 真实标签 [B, H, W]，值为0或1
        Returns:
            dice_loss: Dice损失值
        """
        # 将预测转换为概率分布（softmax）
        pred = torch.softmax(pred, dim=1)  # [B, C, H, W]

        # 只计算裂缝类别的Dice Loss（类别1）
        pred_crack = pred[:, 1, :, :]  # [B, H, W]
        target_crack = target.float()  # [B, H, W]

        # 展平
        pred_flat = pred_crack.contiguous().view(-1)
        target_flat = target_crack.contiguous().view(-1)

        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        # Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss = 1 - Dice系数
        return 1 - dice

# 组合损失：Dice Loss + Cross Entropy Loss
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        """
        组合Dice Loss和Cross Entropy Loss
        Args:
            dice_weight: Dice Loss的权重
            ce_weight: Cross Entropy Loss的权重
        """
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce

# 自定义数据集类，用于加载图像和对应的mask
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

        return image, mask

#数据加载
def train_val_data_process():

    # 定义数据集的路径
    train_image_path = 'data/train/images'
    train_mask_path = 'data/train/masks'
    val_image_path = 'data/val/images'
    val_mask_path = 'data/val/masks'

    normalize = transforms.Normalize(mean=[0.15620959,0.16285059,0.16683851],std=[0.04152462,0.04320607,0.04479365])

    # 定义数据集处理方法变量（仅用于图像）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # 加载数据集
    train_data = RoadCrackDataset(train_image_path, train_mask_path, transform=train_transform)
    val_data = RoadCrackDataset(val_image_path, val_mask_path, transform=train_transform)

    # 根据GPU数量调整batch size
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    # 每个GPU处理的batch size，总batch size = per_gpu_batch_size * num_gpus
    per_gpu_batch_size = 30
    total_batch_size = per_gpu_batch_size * num_gpus

    train_loader = DataLoader(train_data, batch_size=total_batch_size, shuffle=True,
                             num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=total_batch_size, shuffle=False,
                           num_workers=12, pin_memory=True)
    print(f"Training set data size: {len(train_data)}, Validating set data size: {len(val_data)}")
    print(f"使用 {num_gpus} 个GPU, 总batch size: {total_batch_size} (每GPU: {per_gpu_batch_size})")
    return train_loader, val_loader

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
            ious.append(float('nan'))  # 如果该类别不存在
        else:
            ious.append((intersection / union).item())

    return np.nanmean(ious)  # 返回平均IoU

#模型的相关变量定义,以训练集的训练以及测试集的测试
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 检查可用GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")

    # 将模型放入到主设备
    model = model.to(device)

    # 如果有多个GPU，使用DataParallel进行并行训练
    if num_gpus > 1:
        print(f"使用 DataParallel 在 {num_gpus} 个GPU上并行训练")
        model = nn.DataParallel(model)

    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 学习率调度器：余弦退火策略
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    # 损失函数：使用组合损失（Dice Loss + Cross Entropy Loss）
    # Dice Loss对小目标（裂缝）更敏感，Cross Entropy提供稳定的梯度
    criterion = CombinedLoss(dice_weight=0.6, ce_weight=0.4)
    print("使用组合损失函数: Dice Loss (60%) + Cross Entropy Loss (40%)")
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高IoU
    best_iou = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集IoU列表
    train_iou_all = []
    # 验证集IoU列表
    val_iou_all = []

    since = time.time()
    

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-" * 10)

        # 初始化参数
        train_loss = 0.0
        val_loss = 0.0
        train_iou = 0.0
        val_iou = 0.0
        train_num = 0
        val_num = 0

        # 训练阶段
        train_bar = tqdm(train_dataloader, file=sys.stdout, ncols=100, desc="Training")
        for step, (b_x, b_y) in enumerate(train_bar):
            b_x = b_x.to(device)
            b_y = b_y.to(device, dtype=torch.long)  # [B, H, W]
            model.train()

            # 前向传播，输出为 [B, C, H, W]
            output = model(b_x)
            # 计算损失
            loss = criterion(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 获取预测结果 [B, H, W]
            pre_mask = torch.argmax(output, dim=1)

            # 累加损失和IoU
            train_loss += loss.item() * b_x.size(0)
            train_iou += calculate_iou(pre_mask, b_y) * b_x.size(0)
            train_num += b_x.size(0)

        print()

        # 验证阶段
        val_bar = tqdm(val_dataloader, file=sys.stdout, ncols=100, desc="Validation")
        for step, (b_x, b_y) in enumerate(val_bar):
            b_x = b_x.to(device)
            b_y = b_y.to(device, dtype=torch.long)

            model.eval()

            with torch.no_grad():
                output = model(b_x)
                loss = criterion(output, b_y)
                pre_mask = torch.argmax(output, dim=1)

            val_loss += loss.item() * b_x.size(0)
            val_iou += calculate_iou(pre_mask, b_y) * b_x.size(0)
            val_num += b_x.size(0)

        print()

        # 计算平均值
        train_loss_all.append(train_loss / train_num)
        train_iou_all.append(train_iou / train_num)

        val_loss_all.append(val_loss / val_num)
        val_iou_all.append(val_iou / val_num)

        print('{} Train Loss: {:.4f} Train IoU: {:.4f}'.format(epoch, train_loss_all[-1], train_iou_all[-1]))
        print('{} Val Loss: {:.4f} Val IoU: {:.4f}'.format(epoch, val_loss_all[-1], val_iou_all[-1]))

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print('当前学习率: {:.6f}'.format(current_lr))

        print("\n" + "-" * 50 + "\n")

        # 保存最佳模型
        if val_iou_all[-1] > best_iou:
            best_iou = val_iou_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算耗费时间
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数
    # 加载最高IoU下的模型参数
    model.load_state_dict(best_model_wts)

    # 确保保存模型的目标目录存在
    model_save_path = 'best_model.pth'
    model_save_dir = os.path.dirname(model_save_path)
    if model_save_dir and not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print(f"创建目录: {model_save_dir}")

    # 保存模型（如果使用了DataParallel，需要保存module的state_dict）
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_save_path)
    else:
        torch.save(model.state_dict(), model_save_path)
    print(f"最佳模型已保存到: {model_save_path}, 最佳IoU: {best_iou:.4f}")

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                "train_loss_all": train_loss_all,
                                "val_loss_all": val_loss_all,
                                "train_iou_all": train_iou_all,
                                "val_iou_all": val_iou_all
    })
    # 后续可以用来画图
    return train_process

# 画训练图
def matplot_iou_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_iou_all, 'ro-', label="train IoU")
    plt.plot(train_process["epoch"], train_process.val_iou_all, 'bs-', label="val IoU")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("IoU")
    plt.title("IoU Curve")
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("训练曲线已保存到 training_curves.png")
    plt.show()

if __name__ == "__main__":
    # 设置环境变量以优化多GPU性能
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # 使用所有4个GPU

    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用! 检测到 {torch.cuda.device_count()} 个GPU")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA不可用，将使用CPU训练")

    # 初始化U-Net模型，3个输入通道（RGB），2个输出类别（背景/裂缝）
    model = UNet(n_channels=3, n_classes=2)

    print("\n加载数据集...")
    train_dataloader, val_dataloader = train_val_data_process()

    print("\n开始训练...")
    train_process = train_model_process(model, train_dataloader, val_dataloader, 40)

    print("\n绘制训练曲线...")
    matplot_iou_loss(train_process)

