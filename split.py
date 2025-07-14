#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的数据集划分脚本
"""

import os
import shutil
import random
from pathlib import Path


def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    简单划分数据集

    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # 创建输出目录
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # 遍历每个类别目录
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        print(f"处理类别: {class_name}")

        # 获取所有图像文件
        files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
        random.shuffle(files)

        # 计算划分点
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # 划分文件
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        print(f"  训练: {len(train_files)}, 验证: {len(val_files)}, 测试: {len(test_files)}")

        # 创建类别目录并复制文件
        for split, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
            split_class_dir = output_dir / split / class_name
            split_class_dir.mkdir(exist_ok=True)

            for file_path in file_list:
                shutil.copy2(file_path, split_class_dir / file_path.name)


if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)

    # 执行划分
    split_dataset(
        data_dir="./data/math_symbols",  # 原始数据目录
        output_dir="./data/math_symbols_split"  # 输出目录
    )

    print("数据集划分完成!")