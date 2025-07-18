#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
符号粗细调整脚本 - 可以对数据集中的指定符号进行加粗或变细处理
"""

import os
import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm


def adjust_symbol_thickness(base_dir, symbol_ids, operation='thin', kernel_size=(2, 2), iterations=1):
    """
    对指定数据集目录下的特定符号ID进行批量粗细调整操作。

    参数:
        base_dir (str): 数据集分割后的根目录
        symbol_ids (list): 需要处理的符号类别ID列表
        operation (str): 操作类型，'thin'表示变细，'thicken'表示加粗
        kernel_size (tuple): 形态学操作核的大小
        iterations (int): 操作重复次数
    """
    assert operation in ['thin', 'thicken'], "操作类型必须是'thin'或'thicken'"

    op_name = "变细" if operation == 'thin' else "加粗"
    print(f"开始对符号ID {symbol_ids} 进行{op_name}操作...")
    print(f"使用核大小: {kernel_size}, 重复次数: {iterations}")

    # 定义形态学操作所使用的"核"
    kernel = np.ones(kernel_size, np.uint8)

    total_processed_files = 0

    # 遍历 train, val, test 三个子目录
    for split in ['train', 'val', 'test']:
        print(f"\n--- 正在处理 {split} 文件夹 ---")
        for symbol_id in symbol_ids:
            symbol_dir = os.path.join(base_dir, split, str(symbol_id))

            if not os.path.exists(symbol_dir):
                print(f"目录不存在，跳过: {symbol_dir}")
                continue

            # 找到该目录下所有的图片文件
            image_files = glob(os.path.join(symbol_dir, "*.png")) + \
                          glob(os.path.join(symbol_dir, "*.jpg")) + \
                          glob(os.path.join(symbol_dir, "*.jpeg"))

            if not image_files:
                continue

            print(f"  正在处理符号ID {symbol_id}，共 {len(image_files)} 个文件...")

            # 使用tqdm显示处理进度
            for img_path in tqdm(image_files, desc=f"符号 {symbol_id}"):
                try:
                    # 以灰度模式读取图像
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue

                    # 确保图像格式正确
                    if np.mean(image) > 127:  # 如果是白底黑字
                        image = 255 - image  # 翻转为黑底白字
                        was_inverted = True
                    else:
                        was_inverted = False

                    # 执行形态学操作
                    if operation == 'thin':
                        processed_image = cv2.erode(image, kernel, iterations=iterations)
                    else:  # thicken
                        processed_image = cv2.dilate(image, kernel, iterations=iterations)

                    # 如果原来是白底黑字，恢复原来的格式
                    if was_inverted:
                        processed_image = 255 - processed_image

                    # 将处理后的图像覆盖保存回原文件
                    cv2.imwrite(img_path, processed_image)
                    total_processed_files += 1

                except Exception as e:
                    print(f"\n处理文件 {img_path} 时出错: {e}")

    print(f"\n操作完成！总共处理了 {total_processed_files} 个文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='调整符号粗细')
    parser.add_argument('--dir', type=str, default='data/data_black_white/math_symbols_split',
                        help='数据集根目录路径')
    parser.add_argument('--ids', type=int, nargs='+', required=True,
                        help='需要处理的符号ID列表，用空格分隔多个ID')
    parser.add_argument('--op', type=str, choices=['thin', 'thicken'], default='thin',
                        help='操作类型: thin(变细)或thicken(加粗)')
    parser.add_argument('--kernel', type=int, nargs=2, default=(2, 2),
                        help='核大小，例如 2 2 表示2x2的核')
    parser.add_argument('--iter', type=int, default=1,
                        help='操作重复次数')
    parser.add_argument('--force', action='store_true',
                        help='强制执行，不提示备份警告')

    args = parser.parse_args()

    if not args.force:
        backup_choice = input("这是一个覆盖性操作，会修改您的原始图片。强烈建议您先备份数据。\n是否继续？ (y/n): ")
        if backup_choice.lower() != 'y':
            print("操作已取消。")
            exit(0)

    adjust_symbol_thickness(
        base_dir=args.dir,
        symbol_ids=args.ids,
        operation=args.op,
        kernel_size=tuple(args.kernel),
        iterations=args.iter
    )