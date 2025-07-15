#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手写公式识别系统入口脚本 - 简化版
"""

import argparse
import os
import tkinter as tk
import sys

def main():
    """
    主程序入口
    """
    parser = argparse.ArgumentParser(description='手写数学公式识别系统')
    parser.add_argument('--train', action='store_true', help='训练新模型')
    parser.add_argument('--ui', action='store_true', help='启动图形用户界面')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='训练批次大小')
    parser.add_argument('--generate_symbols', action='store_true', help='仅生成数学符号数据集')
    
    args = parser.parse_args()
    
    # 如果没有指定参数，默认启动界面
    if not (args.train or args.ui or args.generate_symbols):
        args.ui = True
    

    
    if args.train:
        print("开始训练模型...")
        from scripts.train import train
        train(epochs=args.epochs, batch_size=args.batch_size)
    
    if args.ui:
        print("启动用户界面...")
        root = tk.Tk()
        # 导入UI模块
        from src.ui.app import HandwrittenFormulaRecognitionApp
        app = HandwrittenFormulaRecognitionApp(root)
        root.mainloop()

if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs('./data/mnist', exist_ok=True)
    os.makedirs('./data/math_symbols', exist_ok=True)
    os.makedirs('./model', exist_ok=True)
    
    main()