#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像预处理模块 - 处理输入图像以便于识别
"""

import cv2
import numpy as np
from PIL import Image
import os
import sys

def preprocess_image(image_input, target_size=(28, 28), threshold=True, normalize=False):
    """
    对输入的手写公式图像进行预处理

    参数:
        image_input: 输入图像的文件路径或NumPy数组
        target_size: 目标尺寸，默认28x28
        threshold: 是否进行二值化
        normalize: 是否归一化到[0,1]范围

    返回:
        预处理后的二值化图像
    """
    # 检查输入是字符串（文件路径）还是NumPy数组
    if isinstance(image_input, str):
        # 读取图像文件
        original_image = cv2.imread(image_input)

        # 检查图像是否成功加载
        if original_image is None:
            raise Exception(f"无法读取图像: {image_input}")
    else:
        # 如果输入已经是NumPy数组，直接使用
        original_image = image_input

    # 确保图像是3通道的（彩色图像）
    if len(original_image.shape) == 2:  # 如果是灰度图
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    # 转换为灰度图
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊去噪
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # # 自适应二值化
    # if threshold:
    #     binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                      cv2.THRESH_BINARY_INV, 11, 2)
    #
    #     # 进行形态学操作以去除小噪点
    #     kernel = np.ones((2, 2), np.uint8)
    #     processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    # else:
    #     processed_image = blurred_image
    processed_image = blurred_image
    # 调整到目标大小
    if target_size and processed_image.shape[:2] != target_size:
        processed_image = cv2.resize(processed_image, target_size, interpolation=cv2.INTER_AREA)

    # 归一化到[0,1]范围
    if normalize:
        processed_image = processed_image.astype(np.float32) / 255.0

    return processed_image

def segment_symbols(preprocessed_image, min_contour_area=20, padding=2):
    """
    将预处理后的公式图像分割成单个符号

    参数:
        preprocessed_image: 预处理后的二值化图像
        min_contour_area: 最小轮廓面积(过滤噪声)
        padding: 分割后的符号周围填充

    返回:
        符号列表: 包含每个分割出的符号图像及其位置信息
    """
    # 确保图像是二值图像
    if len(preprocessed_image.shape) == 3:
        gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = preprocessed_image

    # 寻找轮廓
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbols = []

    # 处理每个轮廓
    for contour in contours:
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)

        # # 过滤掉太小的轮廓（可能是噪点）
        # if w < 10 or h < 10 or cv2.contourArea(contour) < min_contour_area:
        #     continue

        # 提取符号图像
        symbol_image = gray[y:y+h, x:x+w]

        # 添加到结果列表，包含位置信息
        symbols.append({
            'image': symbol_image,
            'position': (x, y, w, h),
            'normalized_image': normalize_symbol(symbol_image)
        })

    # 按位置排序（从左到右）
    symbols.sort(key=lambda s: s['position'][0])

    return symbols

def normalize_symbol(symbol_image, target_size=(28, 28)):
    """
    将分割出的符号图像归一化为固定大小

    参数:
        symbol_image: 单个符号的图像
        target_size: 归一化的目标尺寸

    返回:
        归一化后的图像
    """
    # 获取原始尺寸
    h, w = symbol_image.shape

    # 计算最大边长
    max_dim = max(w, h)

    # 创建正方形画布（填充为黑色背景）
    square_image = np.zeros((max_dim, max_dim), dtype=np.uint8)

    # 将符号居中放置
    x_offset = (max_dim - w) // 2
    y_offset = (max_dim - h) // 2
    square_image[y_offset:y_offset+h, x_offset:x_offset+w] = symbol_image

    # 调整到目标大小
    normalized_image = cv2.resize(square_image, target_size, interpolation=cv2.INTER_AREA)

    return normalized_image

def process_image(image_input):
    """
    完整的图像处理流水线

    参数:
        image_input: 输入图像路径或NumPy数组

    返回:
        (symbol_list, processed_image)
        symbol_list: 分割后的符号列表
        processed_image: 预处理后的图像
    """
    # 预处理图像（不缩放到28x28，保留原始尺寸以便可视化）
    preprocessed_image = preprocess_image(image_input, target_size=None, normalize=False)

    # 分割符号
    symbols = segment_symbols(preprocessed_image)

    # 返回符号列表和预处理后的图像
    return symbols, preprocessed_image