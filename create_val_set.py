import os
import shutil
import random

def create_validation_set(train_images_path, train_masks_path, val_images_path, val_masks_path, ratio=1/3):
    """
    从训练集中随机选取指定比例的图片及其标签移动到验证集

    参数:
        train_images_path: 训练集图片路径
        train_masks_path: 训练集标签路径
        val_images_path: 验证集图片路径
        val_masks_path: 验证集标签路径
        ratio: 移动到验证集的比例，默认1/3
    """

    # 创建验证集目录
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_masks_path, exist_ok=True)

    # 获取训练集中所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    all_images = [f for f in os.listdir(train_images_path)
                  if f.lower().endswith(image_extensions)]

    # 计算需要移动的图片数量
    total_images = len(all_images)
    num_to_move = int(total_images * ratio)

    print(f"训练集总图片数: {total_images}")
    print(f"需要移动到验证集的图片数: {num_to_move} (比例: {ratio:.2%})")

    # 随机选择图片
    random.seed(42)  # 设置随机种子以保证可复现性
    selected_images = random.sample(all_images, num_to_move)

    print(f"\n开始移动图片...")

    moved_count = 0
    error_count = 0

    for img_name in selected_images:
        # 获取不带扩展名的文件名
        base_name = os.path.splitext(img_name)[0]

        # 构建源文件路径
        src_image = os.path.join(train_images_path, img_name)

        # 查找对应的mask文件（可能有不同扩展名）
        mask_file = None
        for ext in image_extensions:
            potential_mask = base_name + ext
            mask_path = os.path.join(train_masks_path, potential_mask)
            if os.path.exists(mask_path):
                mask_file = potential_mask
                break

        if mask_file is None:
            print(f"警告: 未找到 {img_name} 对应的mask文件，跳过")
            error_count += 1
            continue

        src_mask = os.path.join(train_masks_path, mask_file)

        # 构建目标文件路径
        dst_image = os.path.join(val_images_path, img_name)
        dst_mask = os.path.join(val_masks_path, mask_file)

        try:
            # 移动文件
            shutil.move(src_image, dst_image)
            shutil.move(src_mask, dst_mask)
            moved_count += 1

            if moved_count % 100 == 0:
                print(f"  已移动 {moved_count}/{num_to_move} 对图片...")

        except Exception as e:
            print(f"错误: 移动 {img_name} 时出错: {e}")
            error_count += 1

    print(f"\n移动完成!")
    print(f"成功移动: {moved_count} 对图片")
    print(f"失败/跳过: {error_count} 对图片")

    return moved_count, error_count


def count_images(folder_path):
    """统计指定文件夹中的图片数量"""
    if not os.path.exists(folder_path):
        return 0

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    count = sum(1 for f in os.listdir(folder_path)
                if f.lower().endswith(image_extensions))
    return count


if __name__ == "__main__":
    # 设置路径
    train_images_path = 'U-net_to_check_road_crake/data/train/images'
    train_masks_path = '/workspace/lyx/U-net_to_check_road_crake/data/train/masks'
    val_images_path = 'U-net_to_check_road_crake/data/val/images'
    val_masks_path = 'U-net_to_check_road_crake/data/val/masks'

    print("=" * 60)
    print("创建验证集 - 从训练集中移动1/3的数据")
    print("=" * 60)

    # 移动前统计
    print("\n移动前统计:")
    train_img_before = count_images(train_images_path)
    train_mask_before = count_images(train_masks_path)
    print(f"训练集 - 图片: {train_img_before}, 标签: {train_mask_before}")

    # 执行移动
    print("\n" + "=" * 60)
    moved, errors = create_validation_set(
        train_images_path,
        train_masks_path,
        val_images_path,
        val_masks_path,
        ratio=1/3
    )

    # 移动后统计
    print("\n" + "=" * 60)
    print("移动后统计:")
    print("=" * 60)

    train_img_after = count_images(train_images_path)
    train_mask_after = count_images(train_masks_path)
    val_img = count_images(val_images_path)
    val_mask = count_images(val_masks_path)

    print(f"\n训练集:")
    print(f"  - 原始图片 (images): {train_img_after} 张")
    print(f"  - 标注图片 (masks):  {train_mask_after} 张")

    print(f"\n验证集:")
    print(f"  - 原始图片 (images): {val_img} 张")
    print(f"  - 标注图片 (masks):  {val_mask} 张")

    print(f"\n变化:")
    print(f"  - 训练集减少: {train_img_before - train_img_after} 对")
    print(f"  - 验证集增加: {val_img} 对")

    print("=" * 60)
    print("验证集创建完成!")
    print("=" * 60)
