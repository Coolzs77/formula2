#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数学符号数据增强脚本
"""

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def augment_math_symbols(symbols_dir, augment_factor=10):
    """
    对数学符号数据集进行数据增强

    参数:
        symbols_dir: 符号数据集目录路径，包含10-14子目录
        augment_factor: 每个原始图像生成的增强样本数量
    """
    print(f"开始数学符号数据增强，增强因子: {augment_factor}")

    # 符号ID对应表
    symbol_names = {
        10: '+',
        11: '-',
        12: '×',
        13: '÷',
        14: '.'
    }

    total_original = 0
    total_generated = 0

    # 遍历每个符号目录
    for symbol_id in range(10, 15):
        symbol_dir = os.path.join(symbols_dir, str(symbol_id))

        if not os.path.exists(symbol_dir):
            print(f"警告: 符号目录 {symbol_dir} 不存在，跳过")
            continue

        # 获取原始图像列表
        image_files = glob(os.path.join(symbol_dir, "*.png"))
        if not image_files:
            print(f"警告: 符号目录 {symbol_dir} 中未找到PNG图像，尝试其他格式")
            image_files = glob(os.path.join(symbol_dir, "*.jpg")) + glob(os.path.join(symbol_dir, "*.jpeg"))

        if not image_files:
            print(f"警告: 符号目录 {symbol_dir} 中未找到图像，跳过")
            continue

        # 过滤掉已经增强过的图像
        original_images = [img for img in image_files if "_aug" not in os.path.basename(img)]

        print(
            f"处理符号 '{symbol_names.get(symbol_id, symbol_id)}' (ID: {symbol_id}) - 原始样本: {len(original_images)}")
        total_original += len(original_images)

        # 为每个原始图像创建增强版本
        for img_path in tqdm(original_images, desc=f"增强符号 {symbol_names.get(symbol_id, symbol_id)}"):
            try:
                # 获取基础文件名（不含扩展名）
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                # 读取原始图像
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"警告: 无法读取图像 {img_path}，跳过")
                    continue

                # 确保图像尺寸为28x28
                if img.shape[0] != 28 or img.shape[1] != 28:
                    img = cv2.resize(img, (28, 28))

                # 生成多个增强样本
                for i in range(augment_factor):
                    # 应用随机增强
                    aug_img = apply_random_augmentations(img)

                    # 保存增强后的图像
                    output_path = os.path.join(symbol_dir, f"{base_name}_aug{i + 1}.png")
                    cv2.imwrite(output_path, aug_img)
                    total_generated += 1

            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")

    print("\n数据增强完成!")
    print(f"原始图像数量: {total_original}")
    print(f"新生成图像数量: {total_generated}")
    print(f"总图像数量: {total_original + total_generated}")
    return total_generated


def apply_random_augmentations(image):
    """应用随机增强变换"""
    # 转换为PIL图像进行处理
    pil_img = Image.fromarray(image)

    # 随机选择应用的增强数量 (1-4)
    num_augmentations = random.randint(1, 4)

    # 定义可能的增强操作列表
    augmentations = [
        rotate_image,
        shift_image,
        change_contrast,
        add_noise,
        elastic_transform,
        add_blur,
        change_brightness
    ]

    # 随机选择并应用增强
    selected_augs = random.sample(augmentations, num_augmentations)
    for aug_func in selected_augs:
        pil_img = aug_func(pil_img)

    # 转换回OpenCV格式
    return np.array(pil_img)


def rotate_image(img):
    """随机旋转图像"""
    angle = random.uniform(-20, 20)  # -20到20度的随机角度
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)


def shift_image(img):
    """随机平移图像"""
    width, height = img.size
    max_shift = width // 10  # 最大平移量为宽度的1/10

    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)

    return img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, shift_x, 0, 1, shift_y),
        resample=Image.BILINEAR,
        fillcolor=255
    )


def change_contrast(img):
    """随机调整对比度"""
    factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Contrast(img).enhance(factor)


def change_brightness(img):
    """随机调整亮度"""
    factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Brightness(img).enhance(factor)


def add_noise(img):
    """添加随机噪声"""
    # 转换为numpy数组
    img_array = np.array(img)

    # 添加高斯噪声
    noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
    noisy_img = cv2.add(img_array, noise)

    # 确保值在有效范围内
    noisy_img = np.clip(noisy_img, 0, 255)

    return Image.fromarray(noisy_img)


def elastic_transform(img, alpha=5, sigma=2):
    """弹性变换 - 模拟手写风格的变化"""
    # 转换为numpy数组
    img_array = np.array(img)

    # 创建位移场
    shape = img_array.shape
    dx = np.random.rand(shape[0], shape[1]) * 2 - 1
    dy = np.random.rand(shape[0], shape[1]) * 2 - 1

    # 高斯模糊位移场
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    # 创建网格
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    # 应用变换
    transformed = cv2.remap(img_array, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)

    return Image.fromarray(transformed)


def add_blur(img):
    """添加随机模糊"""
    radius = random.uniform(0, 0.8)  # 较小的半径以保持符号可辨认
    return img.filter(ImageFilter.GaussianBlur(radius))


def zoom_and_crop(img):
    """随机缩放和裁剪"""
    width, height = img.size
    scale = random.uniform(0.8, 1.2)

    # 缩放
    new_width = int(width * scale)
    new_height = int(height * scale)
    zoomed = img.resize((new_width, new_height), Image.BILINEAR)

    # 裁剪回原来大小
    if new_width > width:
        left = random.randint(0, new_width - width)
        cropped = zoomed.crop((left, 0, left + width, height))
    elif new_height > height:
        top = random.randint(0, new_height - height)
        cropped = zoomed.crop((0, top, width, top + height))
    else:
        # 填充回原来大小
        result = Image.new(zoomed.mode, (width, height), color=255)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        result.paste(zoomed, (left, top))
        cropped = result

    return cropped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数学符号数据集增强工具")
    parser.add_argument("--dir", type=str, default="./data/math_symbols",
                        help="符号数据集目录路径")
    parser.add_argument("--factor", type=int, default=10,
                        help="每个原始图像生成的增强样本数量")

    args = parser.parse_args()

    # 执行数据增强
    augment_math_symbols(args.dir, args.factor)