#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用划分数据集训练手写公式识别模型
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import argparse

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # 项目根目录
sys.path.append(root_dir)

# 导入必要的模块
from src.models.symbol_recognizer import SymbolRecognizer
from src.models.config import SYMBOL_CLASSES
from src.data.dataset import MathSymbolDataset, TransformDataset


class MnistSplitDataset(Dataset):
    """MNIST划分数据集加载器"""

    def __init__(self, pkl_path, transform=None):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])

        # 将numpy数组转换为PIL图像以便应用变换
        from PIL import Image
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


def load_datasets(batch_size=64):
    """
    加载划分好的数据集(MNIST + 数学符号)

    参数:
        batch_size: 批次大小
    返回:
        训练、验证和测试数据加载器
    """
    print("加载划分后的数据集...")

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

    # 加载MNIST划分数据集
    mnist_train = MnistSplitDataset(os.path.join(root_dir, 'data', 'mnist_split', 'mnist_train.pkl'),
                                    transform=train_transform)
    mnist_val = MnistSplitDataset(os.path.join(root_dir, 'data', 'mnist_split', 'mnist_val.pkl'),
                                  transform=val_test_transform)
    mnist_test = MnistSplitDataset(os.path.join(root_dir, 'data', 'mnist_split', 'mnist_test.pkl'),
                                   transform=val_test_transform)

    print(f"MNIST: {len(mnist_train)} 训练样本, {len(mnist_val)} 验证样本, {len(mnist_test)} 测试样本")

    # 加载数学符号划分数据集
    math_symbol_datasets = {'train': [], 'val': [], 'test': []}

    # 遍历数学符号类别(10-14)
    for symbol_id in range(10, 15):
        for split in ['train', 'val', 'test']:
            symbol_dir = os.path.join(root_dir, 'data', 'math_symbols_split', split, str(symbol_id))
            if not os.path.exists(symbol_dir):
                print(f"警告: 符号目录 {symbol_dir} 不存在")
                continue

            # 获取图像路径和标签
            from glob import glob
            image_files = glob(os.path.join(symbol_dir, "*.png")) + glob(os.path.join(symbol_dir, "*.jpg"))
            samples = [(img_path, symbol_id) for img_path in image_files]

            if samples:
                # 创建数据集
                transform = train_transform if split == 'train' else val_test_transform
                dataset = MathSymbolDataset(samples, transform=transform)
                math_symbol_datasets[split].append(dataset)
                print(f"{split} 集中的符号 {symbol_id} ({SYMBOL_CLASSES[symbol_id]}): {len(samples)} 样本")

    # 合并数据集
    train_datasets = [mnist_train] + math_symbol_datasets['train']
    val_datasets = [mnist_val] + math_symbol_datasets['val']
    test_datasets = [mnist_test] + math_symbol_datasets['test']

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)

    print(f"最终数据集: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本, {len(test_dataset)} 测试样本")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_targets)
    print(example_data.shape)

    fig = plt.figure()
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


    return train_loader, val_loader, test_loader


def train_model(epochs=15, batch_size=64, learning_rate=0.001):
    """
    训练符号识别模型

    参数:
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 记录开始时间
    start_time = time.time()

    # 加载训练、验证和测试数据
    train_loader, val_loader, test_loader = load_datasets(batch_size=batch_size)

    # 创建模型
    print(f"创建CNN模型，类别数: {len(SYMBOL_CLASSES)}")
    model = SymbolRecognizer(num_classes=len(SYMBOL_CLASSES))
    model.to(device)

    # 打印模型信息
    print(model)

    # 计算参数数量
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {param_count:,}")

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # 最佳模型保存路径
    best_model_path = os.path.join(root_dir, 'model', 'best_symbol_model.pth')
    final_model_path = os.path.join(root_dir, 'model', 'symbol_model.pth')
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    # 训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_accuracy = 0.0
    best_epoch = 0

    # 训练循环
    print(f"开始训练，共 {epochs} 轮...")
    for epoch in range(epochs):
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        print(f"Epoch {epoch + 1}/{epochs} - 学习率: {current_lr:.6f}")

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for data, target in progress_bar:
            # 将数据移动到设备
            data, target = data.to(device), target.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            output = model(data)

            # 计算损失
            loss = criterion(output, target)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()

            # 统计训练指标
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            # 更新进度条
            progress_bar.set_postfix(loss=loss.item(), acc=train_correct / train_total)

        # 计算平均训练损失和准确率
        train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
            for data, target in progress_bar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

                # 更新进度条
                progress_bar.set_postfix(loss=loss.item(), acc=val_correct / val_total)

        # 计算平均验证损失和准确率
        val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)

        # 更新学习率
        scheduler.step(val_loss)

        # 打印轮次结果
        print(f'Epoch {epoch + 1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型，验证准确率: {best_accuracy:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), final_model_path)

    # 计算训练总时间
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\n训练完成！")
    print(f"总训练时间: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"最佳验证准确率: {best_accuracy:.4f} (Epoch {best_epoch})")
    print(f"最佳模型已保存为: {best_model_path}")
    print(f"最终模型已保存为: {final_model_path}")

    # 绘制训练曲线
    plot_training_history(history, save_path=os.path.join(root_dir, 'model', 'training_history.png'))

    # 测试最佳模型
    print("\n使用最佳模型在测试集上进行评估...")
    best_model = SymbolRecognizer(num_classes=len(SYMBOL_CLASSES))
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.to(device)

    test_results = evaluate_model(best_model, test_loader, device, criterion)
    print(f"测试集结果 - 损失: {test_results['loss']:.4f}, 准确率: {test_results['accuracy']:.4f}")

    return model, best_accuracy


def evaluate_model(model, data_loader, device, criterion):
    """评估模型性能"""
    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    class_correct = [0] * len(SYMBOL_CLASSES)
    class_total = [0] * len(SYMBOL_CLASSES)

    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="评估"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item() * data.size(0)

            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 统计每个类别的准确率
            for i in range(len(target)):
                label = target[i].item()
                if label in SYMBOL_CLASSES:  # 确保标签在我们的类别映射中
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

    # 计算每个类别的准确率
    class_accuracy = {}
    for i in range(len(SYMBOL_CLASSES)):
        if i in SYMBOL_CLASSES and class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            class_accuracy[SYMBOL_CLASSES[i]] = acc
            print(f"类别 {i} ({SYMBOL_CLASSES[i]}): 准确率 = {acc:.4f} ({class_correct[i]}/{class_total[i]})")

    return {
        'loss': loss / total,
        'accuracy': correct / total,
        'class_accuracy': class_accuracy
    }


def plot_training_history(history, save_path=None):
    """绘制训练历史曲线"""
    plt.figure(figsize=(15, 5))

    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    # 绘制学习率曲线
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()

    if save_path:
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"训练历史图表已保存至: {save_path}")
        except Exception as e:
            print(f"保存图表时出错: {e}")

    # 关闭图形以释放内存
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练手写公式识别模型')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')

    args = parser.parse_args()

    # 训练模型
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )