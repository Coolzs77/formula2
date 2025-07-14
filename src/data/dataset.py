#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集加载模块
"""

import os
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import torch

from src.models.config import SYMBOL_CLASSES


class MathSymbolDataset(Dataset):
    """
    加载数学符号数据集的类
    """

    def __init__(self, samples, transform=None):
        """
        初始化数据集

        参数:
            samples: 图像路径和标签的列表 [(路径, 标签), ...]
            transform: 图像变换
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 读取图像
        try:
            image = Image.open(img_path).convert('L')  # 转换为灰度图

            # 强制调整图像大小为28x28，确保尺寸一致
            if image.size != (28, 28):
                image = image.resize((28, 28), Image.LANCZOS)

        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            # 创建一个空白图像作为替代
            image = Image.new('L', (28, 28), 255)

        if self.transform:
            image = self.transform(image)

        return image, label
class TransformDataset(Dataset):
    """
    为子数据集应用变换的包装器
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def load_dataset(batch_size=64):
    """
    加载数据集（MNIST + 数学符号）

    参数:
        batch_size: 批次大小

    返回:
        训练数据加载器和测试数据加载器
    """
    # 定义图像变换
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载MNIST数据集 (数字0-9)
    print("加载MNIST数据集...")
    mnist_train = datasets.MNIST('../data/mnist', train=True, download=True, transform=train_transform)
    mnist_test = datasets.MNIST('../data/mnist', train=False, download=True, transform=test_transform)

    train_datasets = [mnist_train]
    test_datasets = [mnist_test]

    # 加载自定义数学符号数据集 (类别10-14)
    print("加载数学符号数据集...")
    math_symbols_dir = '../data/math_symbols'

    if os.path.exists(math_symbols_dir):
        # 类别ID列表
        symbol_ids = [10, 11, 12, 13, 14]  # +, -, ×, ÷, .

        # 收集所有数学符号样本
        all_samples = []

        for symbol_id in symbol_ids:
            symbol_dir = os.path.join(math_symbols_dir, str(symbol_id))
            if not os.path.exists(symbol_dir):
                print(f"警告: 符号目录 {symbol_dir} 不存在")
                continue

            # 获取此符号的所有图像文件
            image_files = glob(os.path.join(symbol_dir, "*.png")) + \
                          glob(os.path.join(symbol_dir, "*.jpg"))

            if not image_files:
                print(f"警告: 符号ID {symbol_id} 目录为空")
                continue

            # 将文件路径和标签添加到样本列表
            symbol_samples = [(img_path, symbol_id) for img_path in image_files]
            all_samples.extend(symbol_samples)

            print(f"加载符号ID {symbol_id} ({SYMBOL_CLASSES[symbol_id]}): {len(symbol_samples)} 个样本")

        if all_samples:
            # 创建数据集
            math_symbol_dataset = MathSymbolDataset(all_samples)

            # 分割为训练集和测试集
            train_size = int(0.8 * len(math_symbol_dataset))
            test_size = len(math_symbol_dataset) - train_size

            math_train, math_test = torch.utils.data.random_split(
                math_symbol_dataset, [train_size, test_size]
            )

            # 应用变换
            math_train = TransformDataset(math_train, train_transform)
            math_test = TransformDataset(math_test, test_transform)

            # 添加到数据集列表
            train_datasets.append(math_train)
            test_datasets.append(math_test)

            print(f"成功加载 {len(math_symbol_dataset)} 个数学符号样本")
            print(f"- 训练集: {train_size} 样本")
            print(f"- 测试集: {test_size} 样本")
        else:
            print("未找到数学符号样本")
    else:
        print(f"数学符号目录 {math_symbols_dir} 不存在")

    # 合并数据集
    combined_train = ConcatDataset(train_datasets)
    combined_test = ConcatDataset(test_datasets)

    print(f"最终数据集: {len(combined_train)} 训练样本, {len(combined_test)} 测试样本")

    # 创建数据加载器
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(combined_test, batch_size=batch_size, num_workers=2)

    return train_loader, test_loader