#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的MNIST重新划分脚本
"""

import numpy as np
import gzip
import pickle
import random
from pathlib import Path


def load_mnist_data(data_dir):
    """加载MNIST原始数据"""
    data_dir = Path(data_dir)

    def load_images(filename):
        with gzip.open(data_dir / filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            return data.reshape(-1, 28, 28)

    def load_labels(filename):
        with gzip.open(data_dir / filename, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    # 加载训练集和测试集
    train_images = load_images('train-images-idx3-ubyte.gz')
    train_labels = load_labels('train-labels-idx1-ubyte.gz')
    test_images = load_images('t10k-images-idx3-ubyte.gz')
    test_labels = load_labels('t10k-labels-idx1-ubyte.gz')

    # 合并所有数据
    all_images = np.concatenate([train_images, test_images])
    all_labels = np.concatenate([train_labels, test_labels])

    print(f"总样本数: {len(all_images)}")
    return all_images, all_labels


def split_mnist_simple(data_dir, output_dir, random_seed=42):
    """简单重新划分MNIST"""
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 加载数据
    print("加载MNIST数据...")
    images, labels = load_mnist_data(data_dir)

    # 按类别分组
    class_data = {i: [] for i in range(10)}
    for idx, label in enumerate(labels):
        class_data[label].append(idx)

    # 分层划分
    train_idx, val_idx, test_idx = [], [], []

    for class_id, indices in class_data.items():
        random.shuffle(indices)

        n = len(indices)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

        print(f"数字 {class_id}: 训练={n_train}, 验证={n_val}, 测试={n - n_train - n_val}")

    # 随机打乱
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    # 保存数据
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        'train': (images[train_idx], labels[train_idx]),
        'val': (images[val_idx], labels[val_idx]),
        'test': (images[test_idx], labels[test_idx])
    }

    for split, (data, lbls) in datasets.items():
        with open(output_dir / f'mnist_{split}.pkl', 'wb') as f:
            pickle.dump({'images': data, 'labels': lbls}, f)
        print(f"{split}: {len(data)} 个样本已保存")

    print(f"数据已保存到: {output_dir}")


if __name__ == "__main__":
    split_mnist_simple(
        data_dir="./data/mnist/MNIST/raw",
        output_dir="./data/mnist_split"
    )