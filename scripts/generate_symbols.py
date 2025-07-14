#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数学符号生成脚本
"""

import os
import sys
import argparse

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入生成器
from src.data.generator import MathSymbolGenerator

def generate_symbol_dataset(samples_per_symbol=1000, output_dir=None):
    """
    生成数学符号数据集
    
    参数:
        samples_per_symbol: 每个符号的样本数量
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.join('data', 'math_symbols')
    
    print(f"开始生成数学符号数据集，每个符号 {samples_per_symbol} 个样本")
    print(f"输出目录: {output_dir}")
    
    # 创建符号生成器
    generator = MathSymbolGenerator(output_dir=output_dir)
    
    # 生成数据集
    generator.generate_dataset(samples_per_symbol=samples_per_symbol)
    
    print(f"数学符号数据集生成完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成数学符号数据集')
    parser.add_argument('--samples', type=int, default=1000, help='每个符号生成的样本数量')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    generate_symbol_dataset(samples_per_symbol=args.samples, output_dir=args.output)