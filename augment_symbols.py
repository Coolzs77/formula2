#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合数学符号数据增强脚本 - 结合尺寸缩放和数据增强
支持"数字+.jpg"格式的文件名
"""

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import random
import re
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def enhance_math_symbols_dataset(symbols_dir, augment_factor=10, include_scaling=True):
    """
    对数学符号数据集进行综合增强处理

    参数:
        symbols_dir: 符号数据集目录路径
        augment_factor: 每个原始图像生成的增强样本数量
        include_scaling: 是否包含尺寸缩放处理
    """
    print(f"开始数学符号数据综合增强，增强因子: {augment_factor}, 尺寸缩放: {include_scaling}")

    # 符号ID对应表
    symbol_names = {
        10: '+',
        11: '-',
        12: '×',
        13: '÷',
        14: '.'
    }

    # 定义数字+.jpg格式的正则表达式
    digit_plus_pattern = re.compile(r'^(\d+)\+\.jpe?g$', re.IGNORECASE)

    total_original = 0
    total_generated = 0

    # 遍历每个符号目录
    for symbol_id in range(10, 15):
        symbol_dir = os.path.join(symbols_dir, str(symbol_id))

        if not os.path.exists(symbol_dir):
            print(f"警告: 符号目录 {symbol_dir} 不存在，跳过")
            continue

        # 获取所有图像文件
        image_files = glob(os.path.join(symbol_dir, "*.jpg")) + \
                      glob(os.path.join(symbol_dir, "*.jpeg")) + \
                      glob(os.path.join(symbol_dir, "*.png"))

        if not image_files:
            print(f"警告: 符号目录 {symbol_id} 中未找到图像，跳过")
            continue

        # 过滤出原始图像（数字+.jpg格式或没有增强标记的图像）
        original_images = []
        for img in image_files:
            filename = os.path.basename(img)
            # 检查是否为数字+.jpg格式
            if digit_plus_pattern.match(filename):
                original_images.append(img)
            # 否则检查是否为未增强的图像
            elif "_aug" not in filename and "_scaled" not in filename:
                original_images.append(img)

        print(
            f"处理符号 '{symbol_names.get(symbol_id, symbol_id)}' (ID: {symbol_id}) - 原始样本: {len(original_images)}")
        total_original += len(original_images)

        # 为每个原始图像创建增强版本
        for img_path in tqdm(original_images, desc=f"增强符号 {symbol_names.get(symbol_id, symbol_id)}"):
            try:
                # 获取基础文件名
                filename = os.path.basename(img_path)
                match = digit_plus_pattern.match(filename)

                if match:
                    # 如果是数字+.jpg格式，提取数字部分作为基础文件名
                    base_name = match.group(1) + '+'
                else:
                    # 否则按常规方式提取基础文件名（不含扩展名）
                    base_name = os.path.splitext(filename)[0]

                # 读取原始图像
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"警告: 无法读取图像 {img_path}，跳过")
                    continue

                # 确保图像尺寸为28x28
                if img.shape[0] != 28 or img.shape[1] != 28:
                    img = cv2.resize(img, (28, 28))

                # 存储将要处理的图像变体列表（原始 + 缩放变体）
                image_variants = [img]  # 首先添加原始图像

                # 如果启用尺寸缩放，创建缩放变体
                if include_scaling:
                    # 放大版本 (×1.5)
                    scaled_img = cv2.resize(img, None, fx=1.5, fy=1.5)
                    # 居中裁剪回28×28
                    h, w = scaled_img.shape
                    start_h = max(0, (h - 28) // 2)
                    start_w = max(0, (w - 28) // 2)
                    end_h = min(h, start_h + 28)
                    end_w = min(w, start_w + 28)
                    cropped = scaled_img[start_h:end_h, start_w:end_w]
                    # 如需要，调整大小确保28x28
                    if cropped.shape != (28, 28):
                        cropped = cv2.resize(cropped, (28, 28))
                    image_variants.append(cropped)

                    # 缩小版本 (×0.7)
                    small_img = cv2.resize(img, None, fx=0.7, fy=0.7)
                    # 填充回28×28
                    padded = np.full((28, 28), 255, dtype=np.uint8)
                    h, w = small_img.shape
                    start_h = (28 - h) // 2
                    start_w = (28 - w) // 2
                    # 确保索引在有效范围内
                    if h <= 28 and w <= 28:
                        padded[start_h:start_h + h, start_w:start_w + w] = small_img
                        image_variants.append(padded)

                # 为每个变体应用数据增强
                variant_count = 0
                for variant_idx, variant_img in enumerate(image_variants):
                    # 设置变体标识符
                    variant_suffix = ""
                    if include_scaling and variant_idx > 0:
                        variant_suffix = f"_scale{variant_idx}"

                    # 保存原始变体
                    if variant_idx > 0:  # 只保存缩放变体，原始图像已存在
                        output_path = os.path.join(symbol_dir, f"{base_name}{variant_suffix}.png")
                        cv2.imwrite(output_path, variant_img)
                        total_generated += 1
                        variant_count += 1

                    # 对每个变体应用随机增强
                    for aug_idx in range(augment_factor):
                        # 应用随机增强
                        aug_img = apply_random_augmentations(variant_img)

                        # 保存增强后的图像
                        output_path = os.path.join(
                            symbol_dir,
                            f"{base_name}{variant_suffix}_aug{aug_idx + 1}.png"
                        )
                        cv2.imwrite(output_path, aug_img)
                        total_generated += 1
                        variant_count += 1

                print(f"  - 图像 {base_name}: 创建了 {variant_count} 个变体")

            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")

    print("\n数据增强完成!")
    print(f"原始图像数量: {total_original}")
    print(f"新生成图像数量: {total_generated}")
    print(f"总图像数量: {total_original + total_generated}")
    return total_generated


# 其他函数保持不变
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


# 其余函数保持不变...
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
    img_array = np.array(img)
    noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
    noisy_img = cv2.add(img_array, noise)
    noisy_img = np.clip(noisy_img, 0, 255)
    return Image.fromarray(noisy_img)


def elastic_transform(img, alpha=5, sigma=2):
    """弹性变换 - 模拟手写风格的变化"""
    img_array = np.array(img)
    shape = img_array.shape
    dx = np.random.rand(shape[0], shape[1]) * 2 - 1
    dy = np.random.rand(shape[0], shape[1]) * 2 - 1
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    transformed = cv2.remap(img_array, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)
    return Image.fromarray(transformed)


def add_blur(img):
    """添加随机模糊"""
    radius = random.uniform(0, 0.8)  # 较小的半径以保持符号可辨认
    return img.filter(ImageFilter.GaussianBlur(radius))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数学符号数据集综合增强工具")
    parser.add_argument("--dir", type=str, default="./data/math_symbols",
                        help="符号数据集目录路径")
    parser.add_argument("--factor", type=int, default=10,
                        help="每个原始图像生成的增强样本数量")
    parser.add_argument("--no-scaling", action="store_true",
                        help="禁用尺寸缩放处理，只应用随机增强")

    args = parser.parse_args()

    # 执行数据增强
    enhance_math_symbols_dataset(args.dir, args.factor, not args.no_scaling)