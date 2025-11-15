"""
U-Net 道路裂缝检测数据集加载器
适配U-Net分割任务
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CrackDataset(Dataset):
    """道路裂缝分割数据集"""

    def __init__(self, image_dir, mask_dir, transform=None, img_size=(224, 224)):
        """
        Args:
            image_dir: 图片文件夹路径
            mask_dir: 掩膜文件夹路径
            transform: 数据增强变换
            img_size: 图片尺寸 (height, width)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size

        # 获取所有图片文件名
        self.images = sorted([f for f in os.listdir(image_dir)
                             if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))])

        if len(self.images) == 0:
            raise ValueError(f"❌ 在 {image_dir} 中没有找到图片文件！")

        print(f"✓ 加载数据集: {len(self.images)} 张图片")
        print(f"  图片路径: {image_dir}")
        print(f"  掩膜路径: {mask_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图片
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 读取掩膜（尝试多种扩展名）
        mask_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            temp_path = os.path.join(self.mask_dir, mask_name + ext)
            if os.path.exists(temp_path):
                mask_path = temp_path
                break

        if mask_path is None:
            raise FileNotFoundError(f"❌ 找不到掩膜文件: {mask_name}")

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"❌ 无法读取图片: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        # 读取掩膜
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"❌ 无法读取掩膜: {mask_path}")
        mask = cv2.resize(mask, self.img_size)

        # 归一化掩膜到 0-1
        mask = (mask > 127).astype(np.float32)  # 二值化

        # 转换为 PIL Image（如果需要使用 torchvision transforms）
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            # 转换为 tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # 掩膜转 tensor
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


if __name__ == "__main__":
    """测试数据加载"""
    import matplotlib.pyplot as plt
    from torchvision import transforms

    # 归一化参数
    normalize = transforms.Normalize(
        mean=[0.15804266, 0.16457426, 0.16825973],
        std=[0.04205786, 0.04393576, 0.04547899]
    )

    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # 测试加载数据集
    base_path = 'data'
    train_img_dir = os.path.join(base_path, "train/images")
    train_mask_dir = os.path.join(base_path, "train/masks")

    print("测试数据集加载...")
    dataset = CrackDataset(
        train_img_dir,
        train_mask_dir,
        transform=transform,
        img_size=(224, 224)
    )

    print(f"\n数据集大小: {len(dataset)}")

    # 获取一个样本
    image, mask = dataset[0]

    print(f"图片形状: {image.shape}")  # [3, 448, 448]
    print(f"掩膜形状: {mask.shape}")   # [1, 448, 448]
    print(f"图片范围: [{image.min():.3f}, {image.max():.3f}]")
    print(f"掩膜范围: [{mask.min():.3f}, {mask.max():.3f}]")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 反归一化显示图片
    img = image.permute(1, 2, 0).numpy()
    img = img * np.array([0.04205786, 0.04393576, 0.04547899]) + np.array([0.15804266, 0.16457426, 0.16825973])
    img = np.clip(img, 0, 1)

    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis('off')

    # 显示掩膜
    mask_np = mask.squeeze().numpy()
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('dataset_test.png', dpi=150)
    print("\n✓ 测试图片已保存: dataset_test.png")
    plt.show()
