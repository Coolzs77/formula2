#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
画布捕获模块 - 从Tkinter画布获取图像
"""

import numpy as np
import cv2
import sys
import os
from PIL import Image, ImageDraw

def capture_canvas_windows(canvas):
    """
    在Windows系统上使用win32gui和win32ui截取画布内容
    
    参数:
        canvas: Tkinter画布对象
        
    返回:
        OpenCV格式的图像数组
    """
    try:
        import win32gui
        import win32ui
        from ctypes import windll
        
        # 获取画布的窗口句柄
        hwnd = canvas.winfo_id()
        
        # 获取窗口的客户区域大小
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width = right - left
        height = bottom - top
        
        # 创建设备上下文
        hdc = win32gui.GetDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hdc)
        save_dc = mfc_dc.CreateCompatibleDC()
        
        # 创建位图对象
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)
        
        # 将窗口内容复制到位图
        windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)
        
        # 将位图数据转换为NumPy数组
        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8)
        img = img.reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
        
        # 清理资源
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hdc)
        
        # 转换为BGR格式
        img = img[..., :3]  # 去除Alpha通道
        img = img[:, :, ::-1]  # 转换RGB为BGR
        
        return img
    
    except ImportError:
        print("无法导入win32gui和win32ui模块，请安装pywin32库")
        return None
    except Exception as e:
        print(f"使用win32gui截图时发生错误: {e}")
        return None

def capture_canvas_pil(canvas):
    """
    使用PIL的ImageGrab尝试捕获画布内容
    
    参数:
        canvas: Tkinter画布对象
        
    返回:
        OpenCV格式的图像数组
    """
    try:
        from PIL import ImageGrab
        
        # 获取画布的坐标
        x = canvas.winfo_rootx()
        y = canvas.winfo_rooty()
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        # 截取画布区域
        img = ImageGrab.grab(bbox=(x, y, x+width, y+height))
        
        # 转换为OpenCV格式
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    except Exception as e:
        print(f"使用PIL截图时发生错误: {e}")
        return None

def capture_canvas_mss(canvas):
    """
    使用mss库截取画布内容
    
    参数:
        canvas: Tkinter画布对象
        
    返回:
        OpenCV格式的图像数组
    """
    try:
        import mss
        
        # 获取画布的坐标
        x = canvas.winfo_rootx()
        y = canvas.winfo_rooty()
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        # 定义捕获区域
        monitor = {"top": y, "left": x, "width": width, "height": height}
        
        # 截取区域
        with mss.mss() as sct:
            img = np.array(sct.grab(monitor))
            
        # 转换格式（mss使用BGRA格式）
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    except ImportError:
        print("无法导入mss模块，请安装: pip install mss")
        return None
    except Exception as e:
        print(f"使用mss截图时发生错误: {e}")
        return None

def canvas_to_binary_image(canvas_image):
    """
    将画布图像转换为二值化图像，确保笔迹为黑色、背景为白色
    
    参数:
        canvas_image: 捕获的画布图像
        
    返回:
        二值化后的图像
    """
    if canvas_image is None:
        return None
        
    # 转为灰度
    gray = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)
    
    # 反转图像确保背景为白色(255)，笔迹为黑色(0)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 转回BGR格式
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def capture_canvas(canvas):
    """
    使用多种方法尝试捕获画布内容，并返回二值化后的图像
    
    参数:
        canvas: Tkinter画布对象
        
    返回:
        二值化后的OpenCV图像
    """
    # 直接从画布内容绘制图像 - 更可靠的方法
    try:
        # 获取画布尺寸
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        # 创建一个与画布相同大小的PIL图像
        pil_image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(pil_image)
        
        # 获取所有项目
        items = canvas.find_all()
        
        # 如果没有绘制任何内容
        if not items:
            print("画布为空")
            return np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 绘制所有线条
        for item in items:
            # 获取线条坐标
            coords = canvas.coords(item)
            # 线条宽度
            width_val = canvas.itemcget(item, 'width')
            # 绘制线段
            for i in range(0, len(coords) - 2, 2):
                draw.line(
                    [coords[i], coords[i + 1], coords[i + 2], coords[i + 3]],
                    fill='black',
                    width=int(float(width_val))
                )
        
        # 转换为OpenCV格式
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 确保笔迹为黑色，背景为白色
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
    except Exception as e:
        print(f"直接绘制画布内容失败: {e}")
        
        # 尝试其他方法
        img = None
        
        # 方法1: 在Windows上使用win32gui/win32ui
        if sys.platform == 'win32':
            img = capture_canvas_windows(canvas)
        
        # 方法2: 如果方法1失败，尝试使用PIL的ImageGrab
        if img is None:
            img = capture_canvas_pil(canvas)
        
        # 方法3: 如果方法2失败，尝试使用mss库
        if img is None:
            img = capture_canvas_mss(canvas)
        
        # 如果所有方法都失败，返回空白图像
        if img is None:
            print("所有截图方法均失败，创建空白图像")
            img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # 将图像转换为二值图像
        return canvas_to_binary_image(img)