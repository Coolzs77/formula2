#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量符号颜色翻转脚本 - 同时保存原文件和翻转文件
"""

import os
import cv2
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path


def batch_invert_symbols():
    """批量翻转符号并保存"""

    print("批量符号颜色翻转器")
    print(f"时间: 2025-07-15 06:48:38 UTC")
    print(f"用户: Coolzs77")

    # 配置
    input_dirs = [
        "data/math_symbols_split"
    ]

    output_base = "data_black_white"

    print(f"输出目录: {output_base}")
    os.makedirs(output_base, exist_ok=True)

    total_processed = 0
    total_copied = 0

    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"跳过不存在的目录: {input_dir}")
            continue

        print(f"\n处理目录: {input_dir}")

        # 创建对应的输出目录
        rel_path = os.path.relpath(input_dir, "data")
        output_dir = os.path.join(output_base, rel_path)

        # 查找所有图像
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob(os.path.join(input_dir, '**', ext), recursive=True))

        print(f"找到 {len(image_files)} 个图像文件")

        for img_path in tqdm(image_files, desc="翻转和保存"):
            try:
                # 计算相对路径和输出路径
                rel_img_path = os.path.relpath(img_path, input_dir)
                output_img_dir = os.path.join(output_dir, os.path.dirname(rel_img_path))
                os.makedirs(output_img_dir, exist_ok=True)

                # 读取图像
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_pil = Image.open(img_path).convert('L')

                # 生成输出文件名
                img_name = Path(img_path).stem
                img_ext = Path(img_path).suffix

                # # 保存原图（带original后缀）
                # original_output = os.path.join(output_img_dir, f"{img_name}_original{img_ext}")
                # img_pil.save(original_output)

                # 判断是否需要翻转
                if np.mean(img) > 127:  # 白底黑字
                    # 翻转颜色
                    inverted = 255 - img
                    inverted_output = os.path.join(output_img_dir, f"{img_name}_inverted{img_ext}")
                    cv2.imwrite(inverted_output, inverted)

                    # 同时保存一个标准名称的翻转版本
                    standard_output = os.path.join(output_img_dir, f"{img_name}{img_ext}")
                    cv2.imwrite(standard_output, inverted)

                    total_processed += 1
                else:  # 已经是黑底白字
                    # 直接复制
                    standard_output = os.path.join(output_img_dir, f"{img_name}{img_ext}")
                    img_pil.save(standard_output)
                    total_copied += 1

            except Exception as e:
                print(f"处理 {img_path} 失败: {e}")

    print(f"\n完成！")
    print(f"翻转处理: {total_processed} 个文件")
    print(f"直接复制: {total_copied} 个文件")
    print(f"输出目录: {output_base}")

    # 创建说明文件
    readme_path = os.path.join(output_base, "README.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("符号数据颜色翻转结果\n")
        f.write("=" * 30 + "\n")
        f.write(f"处理时间: 2025-07-15 06:48:38 UTC\n")
        f.write(f"处理用户: Coolzs77\n\n")
        f.write("文件说明:\n")
        f.write("- *_original.* : 原始文件副本\n")
        f.write("- *_inverted.* : 颜色翻转后的文件\n")
        f.write("- 标准名称文件 : 统一格式的文件（黑底白字）\n\n")
        f.write(f"翻转处理: {total_processed} 个文件\n")
        f.write(f"直接复制: {total_copied} 个文件\n")
        f.write(f"总计: {total_processed + total_copied} 个文件\n")


if __name__ == "__main__":
    batch_invert_symbols()