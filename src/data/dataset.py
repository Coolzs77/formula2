#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集加载模块
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import torch

class MathSymbolDataset(Dataset):
    """
    加载自定义数学符号数据集的类
    """
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        
        参数:
            root_dir: 数据集根目录
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 遍历所有类别目录
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                try:
                    label = int(class_dir)
                    # 遍历该类别下的所有图像
                    for img_file in os.listdir(class_path):
                        if img_file.endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, img_file)
                            self.samples.append((img_path, label))
                except ValueError:
                    print(f"跳过非数字类别目录: {class_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 读取图像
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        
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

def load_dataset(batch_size=64, include_math_symbols=True):
    """
    加载数据集（MNIST + 可选的数学符号）
    
    参数:
        batch_size: 批次大小
        include_math_symbols: 是否包含合成数学符号
        
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
    
    # 加载MNIST数据集
    print("加载MNIST数据集...")
    mnist_train = datasets.MNIST('../data/mnist', train=True, download=True, transform=train_transform)
    mnist_test = datasets.MNIST('../data/mnist', train=False, download=True, transform=test_transform)
    
    train_datasets = [mnist_train]
    test_datasets = [mnist_test]
    
    # 如果指定加载数学符号数据集
    if include_math_symbols:
        math_symbols_dir = '../data/math_symbols'
        
        # 检查数学符号数据集是否存在
        if os.path.exists(math_symbols_dir):
            # 加载数学符号数据集
            math_symbol_dataset = MathSymbolDataset(math_symbols_dir, transform=None)
            
            if len(math_symbol_dataset) > 0:
                # 分割为训练集和测试集
                train_size = int(0.8 * len(math_symbol_dataset))
                test_size = len(math_symbol_dataset) - train_size
                
                math_train, math_test = torch.utils.data.random_split(
                    math_symbol_dataset, [train_size, test_size]
                )
                
                # 创建包含正确变换的训练集和测试集
                math_train = TransformDataset(math_train, train_transform)
                math_test = TransformDataset(math_test, test_transform)
                
                train_datasets.append(math_train)
                test_datasets.append(math_test)
                print(f"加载了 {train_size} 个数学符号训练样本和 {test_size} 个测试样本")
            else:
                print("数学符号数据集为空，将只使用MNIST数据集")
        else:
            print("数学符号数据集不存在，将只使用MNIST数据集")
    
    # 合并数据集
    combined_train = ConcatDataset(train_datasets)
    combined_test = ConcatDataset(test_datasets)
    
    print(f"最终数据集: {len(combined_train)} 训练样本, {len(combined_test)} 测试样本")
    
    # 创建数据加载器
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(combined_test, batch_size=batch_size, num_workers=2)
    
    return train_loader, test_loader