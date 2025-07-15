#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手写数学公式识别 - 验证脚本
用于在验证集上评估模型性能，支持多模型对比
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
import json
from glob import glob

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入项目模块
from src.models.symbol_recognizer import SymbolRecognizer
from src.models.config import SYMBOL_CLASSES, MODEL_CONFIG
from train import load_datasets
from test import test_model_on_dataset, calculate_per_class_accuracy


def validate_single_model(model_path, device, val_loader, criterion, class_names):
    """验证单个模型"""
    # 创建模型
    model = SymbolRecognizer(
        num_classes=MODEL_CONFIG['num_classes'],
        input_channels=MODEL_CONFIG['input_channels']
    )

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 评估模型
    val_loss, val_acc, predictions, targets = test_model_on_dataset(
        model, device, val_loader, criterion, class_names
    )

    # 计算每个类别的准确率
    per_class_acc = calculate_per_class_accuracy(predictions, targets, len(SYMBOL_CLASSES))

    return {
        'model_path': model_path,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'per_class_accuracy': per_class_acc,
        'predictions': predictions,
        'targets': targets
    }


def compare_models(model_paths, device, val_loader, criterion, class_names):
    """比较多个模型的性能"""
    results = []

    for model_path in model_paths:
        print(f"\n验证模型: {os.path.basename(model_path)}")
        try:
            result = validate_single_model(model_path, device, val_loader, criterion, class_names)
            results.append(result)
            print(f"验证准确率: {result['val_accuracy']:.2f}%")
        except Exception as e:
            print(f"验证模型 {model_path} 时出错: {e}")

    return results


def plot_model_comparison(results, class_names, save_path=None):
    """绘制模型对比图"""
    if len(results) < 2:
        print("需要至少2个模型才能进行对比")
        return

    # 准备数据
    model_names = [os.path.basename(r['model_path']).replace('.pth', '') for r in results]
    accuracies = [r['val_accuracy'] for r in results]

    # 绘制总体准确率对比
    plt.figure(figsize=(15, 5))

    # 子图1: 总体准确率对比
    plt.subplot(1, 3, 1)
    bars = plt.bar(model_names, accuracies)
    plt.ylabel('验证准确率 (%)')
    plt.title('模型验证准确率对比')
    plt.xticks(rotation=45)

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', va='bottom')

    # 子图2: 每类准确率对比(热力图)
    plt.subplot(1, 3, 2)
    per_class_data = np.array([r['per_class_accuracy'] for r in results])

    import seaborn as sns
    sns.heatmap(per_class_data,
                xticklabels=class_names,
                yticklabels=model_names,
                annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('每类准确率对比')
    plt.xlabel('类别')
    plt.ylabel('模型')

    # 子图3: 验证损失对比
    plt.subplot(1, 3, 3)
    losses = [r['val_loss'] for r in results]
    bars = plt.bar(model_names, losses)
    plt.ylabel('验证损失')
    plt.title('模型验证损失对比')
    plt.xticks(rotation=45)

    # 添加数值标签
    for bar, loss in zip(bars, losses):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                 f'{loss:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型对比图表已保存至 {save_path}")

    plt.show()


def find_models_in_directory(model_dir):
    """在目录中查找所有模型文件"""
    model_patterns = [
        os.path.join(model_dir, '*.pth'),
        os.path.join(model_dir, '**/*.pth')
    ]

    models = []
    for pattern in model_patterns:
        models.extend(glob(pattern, recursive=True))

    # 过滤出常见的模型文件
    valid_models = []
    for model in models:
        basename = os.path.basename(model)
        if any(keyword in basename.lower() for keyword in ['model', 'best', 'final', 'epoch']):
            valid_models.append(model)

    return sorted(valid_models)


def save_validation_results(results, save_path):
    """保存验证结果"""
    # 转换numpy数组为列表以便JSON序列化
    serializable_results = []
    for result in results:
        serializable_result = {
            'model_path': result['model_path'],
            'val_loss': float(result['val_loss']),
            'val_accuracy': float(result['val_accuracy']),
            'per_class_accuracy': [float(acc) for acc in result['per_class_accuracy']]
        }
        serializable_results.append(serializable_result)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"验证结果已保存至 {save_path}")


def validate_models(args):
    """验证模型的主函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载验证数据
    print("正在加载验证数据...")
    _, val_loader, _ = load_datasets(batch_size=args.batch_size)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 准备类别名称
    class_names = [SYMBOL_CLASSES[i] for i in range(len(SYMBOL_CLASSES))]

    # 确定要验证的模型
    model_dir = os.path.join(project_root, args.model_dir)

    if args.model_path:
        # 验证指定的单个模型
        model_paths = [os.path.join(project_root, args.model_path)]
    else:
        # 查找目录中的所有模型
        model_paths = find_models_in_directory(model_dir)
        if not model_paths:
            raise FileNotFoundError(f"在 {model_dir} 中未找到模型文件")

    print(f"找到 {len(model_paths)} 个模型文件:")
    for path in model_paths:
        print(f"  - {path}")

    # 验证模型
    print("\n开始验证...")
    results = compare_models(model_paths, device, val_loader, criterion, class_names)

    if not results:
        print("没有成功验证任何模型")
        return

    # 创建结果保存目录
    results_dir = os.path.join(model_dir, 'validation_results')
    os.makedirs(results_dir, exist_ok=True)

    # 打印结果总结
    print(f"\n{'=' * 60}")
    print(f"验证结果总结")
    print(f"{'=' * 60}")

    for i, result in enumerate(results):
        model_name = os.path.basename(result['model_path'])
        print(f"{i + 1}. {model_name}")
        print(f"   验证准确率: {result['val_accuracy']:.2f}%")
        print(f"   验证损失: {result['val_loss']:.4f}")

    # 找出最佳模型
    best_result = max(results, key=lambda x: x['val_accuracy'])
    print(f"\n最佳模型: {os.path.basename(best_result['model_path'])}")
    print(f"最佳验证准确率: {best_result['val_accuracy']:.2f}%")

    # 保存验证结果
    results_path = os.path.join(results_dir, 'validation_results.json')
    save_validation_results(results, results_path)

    # 绘制对比图表
    if args.plot and len(results) > 1:
        plot_path = os.path.join(results_dir, 'model_comparison.png')
        plot_model_comparison(results, class_names, save_path=plot_path)

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='手写数学公式识别模型验证')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--model_dir', type=str, default='model', help='模型目录')
    parser.add_argument('--model_path', type=str, default=None,
                        help='指定单个模型路径，如果不指定则验证目录中所有模型')
    parser.add_argument('--plot', action='store_true', help='是否绘制对比图表')

    args = parser.parse_args()

    validate_models(args)


if __name__ == '__main__':
    main()