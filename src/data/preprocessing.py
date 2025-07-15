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


def pad_grayscale(image, target_size, pad_value=0):
    """
    将灰度图填充至指定大小

    参数:
        image: 输入灰度图像 (H x W)
        target_size: 目标尺寸 (宽度, 高度)
        pad_value: 填充值，默认为0（黑色）

    返回:
        填充后的灰度图像 (target_height x target_width)
    """
    h, w = image.shape
    target_w, target_h = target_size

    # 创建空白画布
    padded = np.full((target_h, target_w), pad_value, dtype=np.uint8)

    # 计算放置位置（居中）
    x_offset = (target_w - w) // 2
    y_offset = (target_h - h) // 2

    # 放置原图
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = image

    return padded


def preprocess_image(image_input,source='canvas', normalize=False):
    """
    对输入的手写公式图像进行预处理。

    参数:
        image_input: 输入图像的NumPy数组。
        source:      图像来源, 'file' 或 'canvas'。
        normalize:   是否归一化到[0,1]范围。

    返回:
        预处理后的灰度图像。
    """
    if image_input is None:
        raise ValueError("输入图像不能为空 (image_input is None)")

    # 无论输入是彩图还是灰度图，都统一转换为灰度图进行处理
    if len(image_input.shape) == 3:
        gray_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image_input.copy()  # 如果已经是灰度图，创建一个副本以防意外修改

    # --- 核心逻辑 ---
    # 根据明确的来源参数来决定是否需要反转颜色
    if source == 'file':
        # 计算图像的平均像素值
        mean_pixel_value = np.mean(gray_image)

        # 只有当图像是亮背景（白底黑字）时才进行颜色反转
        if mean_pixel_value > 127:  # 阈值127可以根据需要微调
            # 反转颜色：白(255)->黑(0), 黑(0)->白(255)
            gray_image = 255 - gray_image

    elif source =='canvas':
        # 如果图像来自画布（假定已经是黑底或灰度），则直接使用
        gray_image = gray_image
    # 使用高斯模糊去噪
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)


    processed_image = blurred_image

    cv2.imshow("blurred_image", blurred_image)

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

        # 过滤掉太小的轮廓（可能是噪点）
        if w < 5 or h < 5 or cv2.contourArea(contour) < min_contour_area:
            continue

        # 提取符号图像
        symbol_image = gray[y:y + h, x:x + w]

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
    square_image[y_offset:y_offset + h, x_offset:x_offset + w] = symbol_image

    # 调整到目标大小
    if square_image.shape[:2] >= target_size:
        normalized_image = cv2.resize(square_image, target_size, interpolation=cv2.INTER_AREA)
    else:
        normalized_image = pad_grayscale(square_image, target_size)

    return normalized_image


def process_image(image_input,source):
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
    preprocessed_image = preprocess_image(image_input,source, normalize=False)
    # 分割符号
    symbols = segment_symbols(preprocessed_image)


    # for idx, symbol in enumerate(symbols):
    #     # symbol['normalized_image'] = cv2.bitwise_not(symbol['normalized_image'])
    #     cv2.imshow("normalized_image" + str(idx), symbol['normalized_image'])

    # 返回符号列表和预处理后的图像
    return symbols, preprocessed_image