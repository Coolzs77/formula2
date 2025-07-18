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
from src.data.dataset import load_datasets





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
        # scheduler.step(val_loss)

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
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')

    args = parser.parse_args()

    # 训练模型
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )