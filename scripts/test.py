#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手写数学公式识别 - 测试脚本
用于在测试集上评估模型性能
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入项目模块
from src.models.symbol_recognizer import SymbolRecognizer
from src.models.config import SYMBOL_CLASSES, MODEL_CONFIG
from train import load_datasets


def test_model_on_dataset(model, device, test_loader, criterion, class_names):
    """
    在测试集上评估模型

    参数:
        model: 加载的模型
        device: 计算设备
        test_loader: 测试数据加载器
        criterion: 损失函数
        class_names: 类别名称列表

    返回:
        test_loss, test_acc, predictions, targets
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='测试')

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 统计
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 收集预测结果和真实标签
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # 更新进度条
            avg_loss = test_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            progress_bar.set_postfix({
                '损失': f'{avg_loss:.4f}',
                '准确率': f'{accuracy:.2f}%'
            })

    return test_loss / len(test_loader), 100. * correct / total, all_predictions, all_targets


def calculate_per_class_accuracy(predictions, targets, num_classes):
    """计算每个类别的准确率"""
    per_class_correct = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)

    for pred, target in zip(predictions, targets):
        per_class_total[target] += 1
        if pred == target:
            per_class_correct[target] += 1

    per_class_acc = np.divide(per_class_correct, per_class_total,
                              out=np.zeros_like(per_class_correct),
                              where=per_class_total != 0)

    return per_class_acc


def plot_confusion_matrix(targets, predictions, class_names, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(targets, predictions)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至 {save_path}")

    plt.show()


def plot_per_class_accuracy(per_class_acc, class_names, save_path=None):
    """绘制每个类别的准确率"""
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), per_class_acc)
    plt.xlabel('类别')
    plt.ylabel('准确率')
    plt.title('每个类别的准确率')
    plt.xticks(range(len(class_names)), class_names, rotation=0)
    plt.ylim(0, 1)

    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"每类准确率图表已保存至 {save_path}")

    plt.show()


def plot_diverse_sample_predictions(model, device, test_loader, class_names, save_path=None, samples_per_class=2):
    """
    绘制每个类别的样本预测结果，确保包含所有类别

    参数:
        model: 训练好的模型
        device: 计算设备
        test_loader: 测试数据加载器
        class_names: 类别名称列表
        save_path: 保存路径
        samples_per_class: 每个类别显示的样本数量
    """
    model.eval()

    # 初始化每个类别的样本收集器
    class_samples = {i: [] for i in range(len(class_names))}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = torch.max(probabilities, 1)[0]

            # 将数据移到CPU
            images_cpu = images.cpu()
            labels_cpu = labels.cpu()
            predicted_cpu = predicted.cpu()
            confidence_cpu = confidence.cpu()

            # 为每个类别收集样本
            for i in range(len(images_cpu)):
                true_label = labels_cpu[i].item()
                if len(class_samples[true_label]) < samples_per_class:
                    class_samples[true_label].append({
                        'image': images_cpu[i],
                        'true_label': true_label,
                        'pred_label': predicted_cpu[i].item(),
                        'confidence': confidence_cpu[i].item()
                    })

            # 检查是否所有类别都收集够了
            if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                break

    # 准备绘图数据
    plot_samples = []
    for class_id in range(len(class_names)):
        plot_samples.extend(class_samples[class_id][:samples_per_class])

    if not plot_samples:
        print("没有找到样本数据！")
        return

    # 计算网格布局
    cols = samples_per_class
    rows = len(class_names)

    # 创建图形
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    sample_idx = 0
    for class_id in range(len(class_names)):
        for sample_in_class in range(samples_per_class):
            row = class_id
            col = sample_in_class

            if sample_idx < len(plot_samples):
                sample = plot_samples[sample_idx]

                # 绘制图像
                axes[row, col].imshow(sample['image'].squeeze(), cmap='gray')

                # 设置标题
                true_label = class_names[sample['true_label']]
                pred_label = class_names[sample['pred_label']]
                conf = sample['confidence']

                # 根据预测是否正确设置颜色
                color = 'green' if sample['true_label'] == sample['pred_label'] else 'red'
                title = f'真实: {true_label}\n预测: {pred_label}\n置信度: {conf:.3f}'

                axes[row, col].set_title(title, color=color, fontsize=9)
                axes[row, col].axis('off')

                sample_idx += 1
            else:
                # 如果没有足够的样本，显示空图
                axes[row, col].text(0.5, 0.5, f'类别 {class_names[class_id]}\n无样本',
                                    ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].axis('off')

    # 设置行标签
    for i, class_name in enumerate(class_names):
        fig.text(0.02, 1 - (i + 0.5) / rows, f'类别 {class_name}',
                 rotation=90, va='center', fontsize=12, weight='bold')

    plt.suptitle('每个类别的样本预测结果', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"多样本预测图已保存至 {save_path}")

    plt.show()


def plot_symbol_vs_digit_samples(model, device, test_loader, class_names, save_path=None):
    """
    专门对比数字和符号的预测结果
    """
    model.eval()

    # 分类数字和符号
    digit_classes = list(range(10))  # 0-9
    symbol_classes = list(range(10, len(class_names)))  # 10-14 (+, -, ×, ÷, .)

    digit_samples = []
    symbol_samples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = torch.max(probabilities, 1)[0]

            # 移到CPU
            images_cpu = images.cpu()
            labels_cpu = labels.cpu()
            predicted_cpu = predicted.cpu()
            confidence_cpu = confidence.cpu()

            for i in range(len(images_cpu)):
                true_label = labels_cpu[i].item()
                sample_data = {
                    'image': images_cpu[i],
                    'true_label': true_label,
                    'pred_label': predicted_cpu[i].item(),
                    'confidence': confidence_cpu[i].item()
                }

                if true_label in digit_classes and len(digit_samples) < 10:
                    digit_samples.append(sample_data)
                elif true_label in symbol_classes and len(symbol_samples) < 10:
                    symbol_samples.append(sample_data)

            if len(digit_samples) >= 10 and len(symbol_samples) >= 10:
                break

    # 创建图形
    fig, axes = plt.subplots(2, 10, figsize=(20, 6))

    # 绘制数字样本
    for i, sample in enumerate(digit_samples[:10]):
        axes[0, i].imshow(sample['image'].squeeze(), cmap='gray')
        true_label = class_names[sample['true_label']]
        pred_label = class_names[sample['pred_label']]
        conf = sample['confidence']
        color = 'green' if sample['true_label'] == sample['pred_label'] else 'red'
        title = f'真实: {true_label}\n预测: {pred_label}\n({conf:.3f})'
        axes[0, i].set_title(title, color=color, fontsize=10)
        axes[0, i].axis('off')

    # 绘制符号样本
    for i, sample in enumerate(symbol_samples[:10]):
        axes[1, i].imshow(sample['image'].squeeze(), cmap='gray')
        true_label = class_names[sample['true_label']]
        pred_label = class_names[sample['pred_label']]
        conf = sample['confidence']
        color = 'green' if sample['true_label'] == sample['pred_label'] else 'red'
        title = f'真实: {true_label}\n预测: {pred_label}\n({conf:.3f})'
        axes[1, i].set_title(title, color=color, fontsize=10)
        axes[1, i].axis('off')

    # 添加行标签
    fig.text(0.02, 0.75, '数字样本', rotation=90, va='center', fontsize=14, weight='bold')
    fig.text(0.02, 0.25, '符号样本', rotation=90, va='center', fontsize=14, weight='bold')

    plt.suptitle('数字 vs 符号预测对比', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(left=0.06)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"数字符号对比图已保存至 {save_path}")

    plt.show()


def analyze_class_distribution(test_loader, class_names):
    """
    分析测试集中各类别的分布情况
    """
    class_counts = {i: 0 for i in range(len(class_names))}

    for _, labels in test_loader:
        for label in labels:
            class_counts[label.item()] += 1

    print("\n测试集类别分布:")
    print("-" * 40)
    for class_id, count in class_counts.items():
        print(f"{class_names[class_id]:>3}: {count:>5} 样本")

    # 绘制分布图
    plt.figure(figsize=(12, 6))
    counts = list(class_counts.values())
    plt.bar(range(len(class_names)), counts, color='skyblue')
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.title('数据集中各类别样本分布')
    plt.xticks(range(len(class_names)), class_names)

    # 添加数值标签
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts) * 0.01, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_sample_predictions(model, device, test_loader, class_names, save_path=None, num_samples=20):
    """
    绘制测试集样本的预测结果

    参数:
        model: 训练好的模型
        device: 计算设备
        test_loader: 测试数据加载器
        class_names: 类别名称列表
        save_path: 保存路径
        num_samples: 要显示的样本数量
    """
    model.eval()

    # 获取一批测试数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # 确保数据在正确的设备上
    images = images.to(device)
    labels = labels.to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # 获取预测概率
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = torch.max(probabilities, 1)[0]

    # 转移到CPU用于绘图
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()
    confidence = confidence.cpu()

    # 计算网格布局
    cols = 5
    rows = (num_samples + cols - 1) // cols

    # 创建图形
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        if i >= len(images):
            break

        row = i // cols
        col = i % cols

        # 绘制图像
        axes[row, col].imshow(images[i].squeeze(), cmap='gray')

        # 设置标题
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        conf = confidence[i].item()

        # 根据预测是否正确设置颜色
        color = 'green' if labels[i] == predicted[i] else 'red'
        title = f'真实: {true_label}\n预测: {pred_label}\n置信度: {conf:.3f}'

        axes[row, col].set_title(title, color=color, fontsize=10)
        axes[row, col].axis('off')

    # 隐藏多余的子图
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.suptitle('测试集样本预测结果', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"样本预测图已保存至 {save_path}")

    plt.show()
def plot_error_samples(model, device, test_loader, class_names, save_path=None, num_samples=16):
    """
    绘制预测错误的样本

    参数:
        model: 训练好的模型
        device: 计算设备
        test_loader: 测试数据加载器
        class_names: 类别名称列表
        save_path: 保存路径
        num_samples: 要显示的错误样本数量
    """
    model.eval()

    error_samples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = torch.max(probabilities, 1)[0]

            # 找到预测错误的样本
            incorrect_mask = predicted != labels
            if incorrect_mask.any():
                incorrect_images = images[incorrect_mask].cpu()
                incorrect_labels = labels[incorrect_mask].cpu()
                incorrect_predicted = predicted[incorrect_mask].cpu()
                incorrect_confidence = confidence[incorrect_mask].cpu()

                for i in range(len(incorrect_images)):
                    error_samples.append({
                        'image': incorrect_images[i],
                        'true_label': incorrect_labels[i].item(),
                        'pred_label': incorrect_predicted[i].item(),
                        'confidence': incorrect_confidence[i].item()
                    })

                    if len(error_samples) >= num_samples:
                        break

            if len(error_samples) >= num_samples:
                break

    if not error_samples:
        print("没有找到预测错误的样本！")
        return

    # 计算网格布局
    cols = 4
    rows = (len(error_samples) + cols - 1) // cols

    # 创建图形
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, sample in enumerate(error_samples[:num_samples]):
        row = i // cols
        col = i % cols

        # 绘制图像
        axes[row, col].imshow(sample['image'].squeeze(), cmap='gray')

        # 设置标题
        true_label = class_names[sample['true_label']]
        pred_label = class_names[sample['pred_label']]
        conf = sample['confidence']

        title = f'真实: {true_label}\n预测: {pred_label}\n置信度: {conf:.3f}'
        axes[row, col].set_title(title, color='red', fontsize=10)
        axes[row, col].axis('off')

    # 隐藏多余的子图
    for i in range(len(error_samples), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.suptitle('预测错误的样本', fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"错误样本图已保存至 {save_path}")

    plt.show()


def save_test_results(test_results, save_path):
    """保存测试结果到JSON文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"测试结果已保存至 {save_path}")


def analyze_full_dataset_distribution(batch_size=128, save_path='full_dataset_distribution.png'):
    """
    【最终完整版】
    分析并以表格和图表形式，展示整个数据集中各类别的样本分布情况，
    并标注各类别及数据集的总数。
    """
    print("=" * 80)
    print("开始分析完整数据集（训练、验证、测试）的类别分布...")
    print("=" * 80)

    # 步骤 1: 加载所有数据加载器
    try:
        train_loader, val_loader, test_loader = load_datasets(batch_size=batch_size)
        if not all([train_loader, val_loader, test_loader]):
            raise ValueError("一个或多个数据集加载器为空。")
    except Exception as e:
        print(f"错误：数据加载失败，无法进行分析。请检查您的数据文件。错误详情: {e}")
        return

    # 步骤 2: 初始化计数器
    class_names = [SYMBOL_CLASSES[i] for i in sorted(SYMBOL_CLASSES.keys())]
    num_classes = len(class_names)

    # 为每个数据集和总数创建一个独立的计数器
    train_counts = {i: 0 for i in range(num_classes)}
    val_counts = {i: 0 for i in range(num_classes)}
    test_counts = {i: 0 for i in range(num_classes)}
    total_counts = {i: 0 for i in range(num_classes)}

    loaders = {
        '训练集': (train_loader, train_counts),
        '验证集': (val_loader, val_counts),
        '测试集': (test_loader, test_counts)
    }

    # 步骤 3: 遍历所有数据加载器，分别进行统计
    print("正在统计样本数量...")
    for name, (loader, counts) in loaders.items():
        for _, labels in tqdm(loader, desc=f"统计 {name}"):
            for label in labels:
                label_item = label.item()
                if label_item in counts:
                    counts[label_item] += 1
    print("统计完成！")

    # 步骤 4: 【核心】打印您需要的、包含总数的详细表格
    print("\n" + "=" * 80)
    print("数据集各类别样本分布详细表格")
    print("-" * 80)
    # 打印表头
    print(f"{'类别ID':<8} | {'符号':<6} | {'训练集':<12} | {'验证集':<12} | {'测试集':<12} | {'总计':<12}")
    print("-" * 80)

    # 初始化总计计数器
    total_train, total_val, total_test, grand_total = 0, 0, 0, 0

    # 打印每一行的数据
    for class_id in sorted(SYMBOL_CLASSES.keys()):
        symbol = SYMBOL_CLASSES[class_id]
        train_count = train_counts.get(class_id, 0)
        val_count = val_counts.get(class_id, 0)
        test_count = test_counts.get(class_id, 0)
        class_total = train_count + val_count + test_count

        # 更新总计
        total_train += train_count
        total_val += val_count
        total_test += test_count
        grand_total += class_total

        print(
            f"{class_id:<8} | {symbol:<6} | {train_count:<12} | {val_count:<12} | {test_count:<12} | {class_total:<12}")

    # 打印总计行
    print("-" * 80)
    print(f"{'总计':<8} | {'-':<6} | {total_train:<12} | {total_val:<12} | {total_test:<12} | {grand_total:<12}")
    print("=" * 80)

    # --- 步骤 5: 绘制更直观的“堆叠条形图” ---
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(16, 9))

    # 准备绘图数据
    labels = list(class_names)
    train_data = [train_counts[i] for i in sorted(train_counts.keys())]
    val_data = [val_counts[i] for i in sorted(val_counts.keys())]
    test_data = [test_counts[i] for i in sorted(test_counts.keys())]

    bar_width = 0.6
    indices = np.arange(len(labels))

    bar1 = plt.bar(indices, train_data, bar_width, label='训练集', color='cornflowerblue', edgecolor='grey')
    bar2 = plt.bar(indices, val_data, bar_width, bottom=train_data, label='验证集', color='lightsalmon',
                   edgecolor='grey')
    bottom_for_test = [i + j for i, j in zip(train_data, val_data)]
    bar3 = plt.bar(indices, test_data, bar_width, bottom=bottom_for_test, label='测试集', color='lightgreen',
                   edgecolor='grey')

    # 【核心】在每个堆叠条形图的顶部标注总数
    for i, (train_val, val_val, test_val) in enumerate(zip(train_data, val_data, test_data)):
        total_height = train_val + val_val + test_val
        plt.text(i, total_height, f'{total_height}', ha='center', va='bottom', fontsize=10, weight='bold')

    plt.xlabel('类别', fontsize=12, weight='bold')
    plt.ylabel('样本数量', fontsize=12, weight='bold')
    plt.title('数据集中各类别样本分布（按数据集划分）', fontsize=16, weight='bold')
    plt.xticks(indices, labels, rotation=0, ha="right")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"\n数据集分布图已保存至: {save_path}")

    plt.show()
def test_model(args):
    """测试模型的主函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print("正在加载测试数据...")
    _, _, test_loader = load_datasets(batch_size=args.batch_size)

    # 创建模型
    print("正在创建模型...")
    model = SymbolRecognizer(
        num_classes=MODEL_CONFIG['num_classes'],
        input_channels=MODEL_CONFIG['input_channels']
    )

    # 加载训练好的模型
    model_dir = os.path.join(project_root, args.model_dir)
    model_path = os.path.join(model_dir, args.model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")

    print(f"加载模型: {model_path}")
    # 修复警告: 添加 weights_only=True
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 准备类别名称
    class_names = [SYMBOL_CLASSES[i] for i in range(len(SYMBOL_CLASSES))]

    # 在测试集上评估
    print("开始测试...")
    test_loss, test_acc, predictions, targets = test_model_on_dataset(
        model, device, test_loader, criterion, class_names
    )

    # 计算每个类别的准确率
    per_class_acc = calculate_per_class_accuracy(predictions, targets, len(SYMBOL_CLASSES))

    # 生成分类报告
    class_report = classification_report(
        targets, predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # 打印结果
    print(f"\n{'=' * 50}")
    print(f"测试结果")
    print(f"{'=' * 50}")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"\n每个类别的准确率:")
    for i, acc in enumerate(per_class_acc):
        print(f"{class_names[i]}: {acc:.3f}")

    print(f"\n详细分类报告:")
    print(classification_report(targets, predictions, target_names=class_names, zero_division=0))

    # 保存测试结果
    test_results = {
        'model_path': model_path,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'per_class_accuracy': {class_names[i]: float(acc) for i, acc in enumerate(per_class_acc)},
        'classification_report': class_report,
        'total_samples': len(targets)
    }

    # 创建结果保存目录
    results_dir = os.path.join(model_dir, 'test_results')
    os.makedirs(results_dir, exist_ok=True)

    # 保存测试结果
    results_path = os.path.join(results_dir, f'test_results_{args.model_name.split(".")[0]}.json')
    save_test_results(test_results, results_path)

    # 绘制混淆矩阵
    cm_path = os.path.join(results_dir, f'confusion_matrix_{args.model_name.split(".")[0]}.png')
    plot_confusion_matrix(targets, predictions, class_names, save_path=cm_path)

    # 绘制每个类别的准确率
    acc_path = os.path.join(results_dir, f'per_class_accuracy_{args.model_name.split(".")[0]}.png')
    plot_per_class_accuracy(per_class_acc, class_names, save_path=acc_path)


    # 分析类别分布
    analyze_full_dataset_distribution()

    print("分析测试集类别分布...")
    analyze_class_distribution(test_loader, class_names)

    # 绘制样本预测结果
    if args.plot_samples:
        print("绘制样本预测结果...")
        sample_path = os.path.join(results_dir, f'sample_predictions_{args.model_name.split(".")[0]}.png')
        plot_sample_predictions(model, device, test_loader, class_names, save_path=sample_path,
                                num_samples=args.num_samples)

        # 绘制每个类别的样本
        print("绘制每个类别的样本...")
        diverse_path = os.path.join(results_dir, f'diverse_samples_{args.model_name.split(".")[0]}.png')
        plot_diverse_sample_predictions(model, device, test_loader, class_names, save_path=diverse_path,
                                        samples_per_class=6)

        # 绘制数字vs符号对比
        print("绘制数字vs符号对比...")
        comparison_path = os.path.join(results_dir, f'digit_symbol_comparison_{args.model_name.split(".")[0]}.png')
        plot_symbol_vs_digit_samples(model, device, test_loader, class_names, save_path=comparison_path)

        # 绘制错误样本
        print("绘制预测错误的样本...")
        error_path = os.path.join(results_dir, f'error_samples_{args.model_name.split(".")[0]}.png')
        plot_error_samples(model, device, test_loader, class_names, save_path=error_path,
                           num_samples=args.num_error_samples)

    return test_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='手写数学公式识别模型测试')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--model_dir', type=str, default='model', help='模型目录')
    parser.add_argument('--model_name', type=str, default='best_symbol_model.pth',
                        help='模型文件名')
    parser.add_argument('--plot_samples',default=True, action='store_true', help='是否绘制样本预测图')
    parser.add_argument('--num_samples', type=int, default=20, help='显示的样本数量')
    parser.add_argument('--num_error_samples', type=int, default=16, help='显示的错误样本数量')

    args = parser.parse_args()

    test_model(args)


if __name__ == '__main__':
    main()