#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集加载模块 - 最终健壮版本
使用动态绝对路径，与执行位置无关
"""

import os
import pickle
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch

# 导入配置，假设config.py在src/models/下
# 为了让这个脚本也能独立运行，我们做一些路径处理
try:
    from src.models.config import SYMBOL_CLASSES
except ImportError:
    # 如果直接运行此脚本，需要手动添加项目根目录到sys.path
    import sys

    # __file__ 是当前脚本的路径
    # os.path.dirname(__file__) 是当前脚本所在的目录 (src/data)
    # os.path.dirname(os.path.dirname(__file__)) 是上级目录 (src)
    # os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 是项目的根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(project_root)
    from src.models.config import SYMBOL_CLASSES

# --- 关键修改：动态获取项目根目录 ---
# 这使得无论从哪里运行脚本，路径都是正确的
# __file__ 指的是当前脚本 (dataset.py) 的路径
# os.path.abspath(__file__) 获取其绝对路径
# 通过三次 os.path.dirname 向上导航到项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MathSymbolDataset(Dataset):
    """数学符号数据集类"""

    def __init__(self, samples, transform=None, target_size=28):
        self.samples = samples
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('L')
            if image.size != (self.target_size, self.target_size):
                image = image.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            image = Image.new('L', (self.target_size, self.target_size), 0)  # 使用黑色背景

        if self.transform:
            image = self.transform(image)
        return image, label


class MnistSplitDataset(Dataset):
    """MNIST划分数据集加载器"""

    def __init__(self, pkl_path, transform=None, target_size=28):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        image = Image.fromarray(image)
        if image.size != (self.target_size, self.target_size):
            image = image.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        if self.transform:
            image = self.transform(image)
        return image, label


def load_datasets(batch_size=64, target_size=28):
    """
    加载所有数据集的最终健壮版本
    """
    print("=" * 60)
    print("加载数据集 - 健壮路径版本")
    print(f"项目根目录: {PROJECT_ROOT}")
    print("=" * 60)

    # 定义图像变换
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 1. 加载MNIST数据 - 使用绝对路径
    print("1. 加载MNIST数据...")
    # os.path.join() 会根据操作系统自动使用正确的路径分隔符（'\' 或 '/'）
    mnist_train_path = os.path.join(PROJECT_ROOT, "data", "mnist_split", "mnist_train.pkl")
    mnist_val_path = os.path.join(PROJECT_ROOT, "data", "mnist_split", "mnist_val.pkl")
    mnist_test_path = os.path.join(PROJECT_ROOT, "data", "mnist_split", "mnist_test.pkl")

    all_datasets = {'train': [], 'val': [], 'test': []}

    if os.path.exists(mnist_train_path):
        all_datasets['train'].append(
            MnistSplitDataset(mnist_train_path, transform=train_transform, target_size=target_size))
        print(f"  ✓ MNIST 训练: {len(all_datasets['train'][-1])} 样本")
    else:
        print(f"  ❌ MNIST训练文件不存在: {mnist_train_path}")

    # (为简洁起见，省略了val和test的加载代码，逻辑与train相同)
    if os.path.exists(mnist_val_path):
        all_datasets['val'].append(
            MnistSplitDataset(mnist_val_path, transform=val_test_transform, target_size=target_size))
        print(f"  ✓ MNIST 验证: {len(all_datasets['val'][-1])} 样本")

    if os.path.exists(mnist_test_path):
        all_datasets['test'].append(
            MnistSplitDataset(mnist_test_path, transform=val_test_transform, target_size=target_size))
        print(f"  ✓ MNIST 测试: {len(all_datasets['test'][-1])} 样本")

    # 2. 加载符号数据 - 使用绝对路径
    print("\n2. 加载符号数据...")
    symbols_base = os.path.join(PROJECT_ROOT, "data", "data_black_white", "math_symbols_split")
    print(f"  符号数据根目录: {symbols_base}")

    if not os.path.exists(symbols_base):
        print(f"  ❌ 符号数据目录不存在")
    else:
        symbol_ids = [10, 11, 12, 13, 14,15,16]
        for split in ['train', 'val', 'test']:
            print(f"  处理 {split} 集符号:")
            split_samples = []
            for symbol_id in symbol_ids:
                symbol_path = os.path.join(symbols_base, split, str(symbol_id))
                if os.path.exists(symbol_path):
                    files = glob(f"{symbol_path}/*.png") + glob(f"{symbol_path}/*.jpg")
                    clean_files = [f for f in files if
                                   not any(x in os.path.basename(f) for x in ['_original', '_inverted'])]
                    if clean_files:
                        symbol_samples = [(img_path, symbol_id) for img_path in clean_files]
                        split_samples.extend(symbol_samples)
                        print(f"    符号 {symbol_id} ({SYMBOL_CLASSES[symbol_id]}): {len(symbol_samples)} 样本")

            if split_samples:
                transform = train_transform if split == 'train' else val_test_transform
                all_datasets[split].append(
                    MathSymbolDataset(split_samples, transform=transform, target_size=target_size))
                print(f"    {split} 符号总计: {len(split_samples)} 样本")

    # 3. 合并并创建加载器
    print("\n3. 创建数据加载器...")
    loaders = {}
    for split in ['train', 'val', 'test']:
        if all_datasets[split]:
            combined_dataset = ConcatDataset(all_datasets[split])
            shuffle = True if split == 'train' else False
            loaders[split] = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
            print(f"  ✓ {split} 加载器创建成功, 总样本数: {len(combined_dataset)}")
        else:
            loaders[split] = None
            print(f"  ❌ {split} 加载器: 无数据")

    print("=" * 60)
    return loaders.get('train'), loaders.get('val'), loaders.get('test')


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_datasets(batch_size=64, target_size=28)