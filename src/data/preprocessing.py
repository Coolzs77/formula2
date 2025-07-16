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

    # cv2.imshow("blurred_image", blurred_image)

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
    # 过滤掉噪点
    initial_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            initial_boxes.append(cv2.boundingRect(contour))

    # 步骤 2: 【核心】调用新的合并函数
    final_boxes = merge_contours(initial_boxes)
    # final_boxes = initial_boxes
    # 步骤 3: 根据最终的边界框提取符号
    symbols = []
    for (x, y, w, h) in final_boxes:
        symbol_image = preprocessed_image[y:y + h, x:x + w]
        symbols.append({
            'image': symbol_image,
            'position': (x, y, w, h),
            'normalized_image': normalize_symbol(symbol_image)
        })
    # # 处理每个轮廓
    # for contour in contours:
    #     # 获取边界框
    #     x, y, w, h = cv2.boundingRect(contour)
    #
    #     # 过滤掉太小的轮廓（可能是噪点）
    #     if w < 5 or h < 5 or cv2.contourArea(contour) < min_contour_area:
    #         continue
    #
    #     # 提取符号图像
    #     symbol_image = gray[y:y + h, x:x + w]
    #
    #     # 添加到结果列表，包含位置信息
    #     symbols.append({
    #         'image': symbol_image,
    #         'position': (x, y, w, h),
    #         'normalized_image': normalize_symbol(symbol_image)
    #     })
    #
    # 按位置排序（从左到右）
    symbols.sort(key=lambda s: s['position'][0])

    return symbols


def merge_contours(boxes, overlap_threshold=0.5):
    """
    【最终智能版】合并边界框。
    仅当两个框在水平方向上有显著重叠时，才将它们合并。
    """
    if len(boxes) <= 1:
        return boxes

    merged = True
    while merged:
        merged = False
        new_boxes = []
        # 使用一个数组来标记哪些框已经被合并掉了
        is_merged = [False] * len(boxes)

        for i in range(len(boxes)):
            if is_merged[i]:
                continue

            # 创建当前要合并的框
            current_box = list(boxes[i])

            for j in range(i + 1, len(boxes)):
                if is_merged[j]:
                    continue

                next_box = boxes[j]

                # --- 核心判断逻辑 ---
                # 1. 计算两个框在水平方向上的重叠区域长度
                x1_max = current_box[0] + current_box[2]
                x2_max = next_box[0] + next_box[2]
                overlap_x = max(0, min(x1_max, x2_max) - max(current_box[0], next_box[0]))

                # 2. 只有当重叠长度大于任一框宽度的阈值时，才认为是垂直结构
                min_width = min(current_box[2], next_box[2])
                if overlap_x > min_width * overlap_threshold:
                    # 合并两个框
                    x_min = min(current_box[0], next_box[0])
                    y_min = min(current_box[1], next_box[1])
                    x_max = max(current_box[0] + current_box[2], next_box[0] + next_box[2])
                    y_max = max(current_box[1] + current_box[3], next_box[1] + next_box[3])

                    current_box = [x_min, y_min, x_max - x_min, y_max - y_min]

                    # 标记 next_box 已被合并
                    is_merged[j] = True
                    merged = True  # 标记本轮发生了合并

            new_boxes.append(tuple(current_box))

        # 如果本轮发生了合并，就用合并后的新列表进行下一轮检查
        if merged:
            boxes = new_boxes

    # 按x坐标排序，确保最终顺序正确
    new_boxes.sort(key=lambda b: b[0])
    return new_boxes


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