#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数学符号生成器模块
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm

class MathSymbolGenerator:
    """
    用于生成数学符号数据集的类
    """
    def __init__(self, output_dir="../data/math_symbols"):
        """
        初始化生成器
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义要生成的符号
        self.symbols = {
            '+': 10,
            '-': 11,
            '×': 12,
            '÷': 13,
            '=': 14,
            '(': 15,
            ')': 16,
            'x': 17,
            'y': 18,
            '^': 19,
            '√': 20,
            'π': 21,
            '!': 22,
            '%': 23,
            '.': 24
        }
        
        # 创建符号子目录
        for symbol in self.symbols:
            symbol_dir = os.path.join(self.output_dir, str(self.symbols[symbol]))
            os.makedirs(symbol_dir, exist_ok=True)
    
    def generate_dataset(self, samples_per_symbol=1000, image_size=28):
        """
        为每个符号生成指定数量的样本
        
        参数:
            samples_per_symbol: 每个符号的样本数
            image_size: 输出图像大小
        """
        print(f"开始生成数学符号数据集，每个符号 {samples_per_symbol} 个样本...")
        
        # 加载多种字体
        fonts = self._get_available_fonts(16)  # 字体大小范围
        
        # 为每个符号生成样本
        for symbol, label in self.symbols.items():
            symbol_dir = os.path.join(self.output_dir, str(label))
            print(f"生成符号 '{symbol}' (类别 {label}) 的样本...")
            
            for i in tqdm(range(samples_per_symbol)):
                # 创建空白图像
                img = Image.new('L', (image_size, image_size), color=255)
                draw = ImageDraw.Draw(img)
                
                # 随机选择字体和大小
                font_path = random.choice(fonts)
                font_size = random.randint(14, 22)
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except:
                    # 如果字体加载失败，使用默认字体
                    font = ImageFont.load_default()
                
                # 计算文本尺寸以居中放置
                try:
                    text_width, text_height = draw.textsize(symbol, font=font)
                except:
                    # 某些版本的PIL/Pillow可能不支持textsize
                    text_width, text_height = font_size, font_size
                
                # 计算居中位置，添加一点随机偏移
                x = (image_size - text_width) // 2 + random.randint(-3, 3)
                y = (image_size - text_height) // 2 + random.randint(-3, 3)
                x = max(0, min(x, image_size - text_width - 1))
                y = max(0, min(y, image_size - text_height - 1))
                
                # 绘制符号
                draw.text((x, y), symbol, fill=0, font=font)
                
                # 添加随机变换以增强数据多样性
                img = self._apply_random_transformations(img)
                
                # 保存图像
                img_path = os.path.join(symbol_dir, f"{i:05d}.png")
                img.save(img_path)
            
            print(f"完成符号 '{symbol}' 的样本生成，保存在 {symbol_dir}")
    
    def _get_available_fonts(self, size=16):
        """获取系统中可用的字体"""
        # 常见字体路径
        font_paths = []
        
        # Windows字体路径
        windows_font_dir = "C:/Windows/Fonts"
        if os.path.exists(windows_font_dir):
            # 添加一些常见的等宽字体
            for font_name in ['arial.ttf', 'times.ttf', 'cour.ttf', 'calibri.ttf', 'cambria.ttf']:
                font_path = os.path.join(windows_font_dir, font_name)
                if os.path.exists(font_path):
                    font_paths.append(font_path)
        
        # Linux字体路径
        linux_font_dirs = [
            "/usr/share/fonts/truetype",
            "/usr/share/fonts/TTF"
        ]
        for font_dir in linux_font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for file in files:
                        if file.endswith('.ttf'):
                            font_paths.append(os.path.join(root, file))
        
        # macOS字体路径
        mac_font_dir = "/Library/Fonts"
        if os.path.exists(mac_font_dir):
            for file in os.listdir(mac_font_dir):
                if file.endswith('.ttf'):
                    font_paths.append(os.path.join(mac_font_dir, file))
        
        # 如果没有找到字体，添加默认字体
        if not font_paths:
            font_paths = ["default"]
        
        return font_paths
    
    def _apply_random_transformations(self, img):
        """应用随机变换以增强数据多样性"""
        # 转换为NumPy数组
        img_array = np.array(img)
        
        # 随机旋转
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = img_array.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_array = cv2.warpAffine(img_array, M, (w, h), 
                                       flags=cv2.INTER_LINEAR, 
                                       borderMode=cv2.BORDER_CONSTANT, 
                                       borderValue=255)
        
        # 随机缩放
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            h, w = img_array.shape
            new_h, new_w = int(h * scale), int(w * scale)
            img_array = cv2.resize(img_array, (new_w, new_h))
            
            # 调整回原始大小
            if new_h < h:
                pad_h = h - new_h
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                img_array = cv2.copyMakeBorder(
                    img_array, pad_top, pad_bottom, 0, 0, 
                    cv2.BORDER_CONSTANT, value=255
                )
            elif new_h > h:
                crop_h = new_h - h
                crop_top = crop_h // 2
                img_array = img_array[crop_top:crop_top+h, :]
                
            if new_w < w:
                pad_w = w - new_w
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                img_array = cv2.copyMakeBorder(
                    img_array, 0, 0, pad_left, pad_right, 
                    cv2.BORDER_CONSTANT, value=255
                )
            elif new_w > w:
                crop_w = new_w - w
                crop_left = crop_w // 2
                img_array = img_array[:, crop_left:crop_left+w]
        
        # 添加少量噪声
        if random.random() > 0.7:
            noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
            img_array = cv2.add(img_array, noise)
            img_array = np.clip(img_array, 0, 255)
        
        # 随机模糊
        if random.random() > 0.8:
            kernel_size = random.choice([3, 5])
            img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
        
        # 转回PIL图像
        return Image.fromarray(img_array)