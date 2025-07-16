#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
符号识别模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 添加项目根目录到系统路径，确保可以导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.config import SYMBOL_CLASSES

class ConvBlock(nn.Module):
    """
    卷积块：Conv2d + BatchNorm2d + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """
    残差块：两个卷积层 + 跳跃连接
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out += residual  # 添加残差连接
        return F.relu(out)

class SymbolRecognizer(nn.Module):
    """
    符号识别模型：使用卷积块、残差连接和全局池化
    """
    def __init__(self, num_classes=len(SYMBOL_CLASSES), input_channels=1):
        super(SymbolRecognizer, self).__init__()
        
        # 初始特征提取
        self.initial = ConvBlock(input_channels, 32)
        
        # 第一阶段：32通道，2个残差块
        self.stage1_pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        self.stage1_res1 = ResidualBlock(32)
        self.stage1_res2 = ResidualBlock(32)
        
        # 第二阶段：64通道，2个残差块
        self.stage2_conv = ConvBlock(32, 64, stride=1)
        self.stage2_pool = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        self.stage2_res1 = ResidualBlock(64)
        self.stage2_res2 = ResidualBlock(64)
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化: NxCx7x7 -> NxCx1x1
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 初始特征提取
        x = self.initial(x)  # Nx1x28x28 -> Nx32x28x28
        
        # 第一阶段
        x = self.stage1_pool(x)  # Nx32x28x28 -> Nx32x14x14
        x = self.stage1_res1(x)
        x = self.stage1_res2(x)
        
        # 第二阶段
        x = self.stage2_conv(x)  # Nx32x14x14 -> Nx64x14x14
        x = self.stage2_pool(x)  # Nx64x14x14 -> Nx64x7x7
        x = self.stage2_res1(x)
        x = self.stage2_res2(x)
        
        # 全局池化
        x = self.global_pool(x)  # Nx64x7x7 -> Nx64x1x1
        x = x.view(x.size(0), -1)  # Nx64x1x1 -> Nx64
        
        # 分类
        x = self.classifier(x)  # Nx64 -> Nx<num_classes>
        
        return x