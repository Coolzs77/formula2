#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型配置
"""

# 定义符号类别映射
SYMBOL_CLASSES = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '-', 12: '×', 13: '÷', 14: '=',
    15: '(', 16: ')', 17: 'x', 18: 'y', 19: '^',
    20: 'sqrt', 21: 'pi', 22: '!', 23: '%', 24: '.'
}

# 模型默认参数
MODEL_CONFIG = {
    'input_channels': 1,
    'hidden_channels': 32,
    'num_classes': len(SYMBOL_CLASSES),
    'dropout_rate': 0.25,
    'learning_rate': 0.001
}