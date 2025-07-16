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
        PIL格式的图像
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

        # 确保画布尺寸有效
        if width <= 0 or height <= 0:
            width = max(width, 600)
            height = max(height, 400)
            print(f"画布尺寸调整为: {width}x{height}")

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

        # 转换为RGB格式，去除Alpha通道
        img = img[..., :3]  # 去除Alpha通道
        img = img[:, :, ::-1]  # 转换BGR为RGB

        # 转换为PIL图像
        return Image.fromarray(img)

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
        PIL格式的图像
    """
    try:
        from PIL import ImageGrab

        # 获取画布的坐标
        x = canvas.winfo_rootx()
        y = canvas.winfo_rooty()
        width = canvas.winfo_width()
        height = canvas.winfo_height()

        # 确保尺寸有效
        if width <= 0 or height <= 0:
            width = max(width, 600)
            height = max(height, 400)

        # 截取画布区域
        img = ImageGrab.grab(bbox=(x, y, x + width, y + height))

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
        PIL格式的图像
    """
    try:
        import mss

        # 获取画布的坐标
        x = canvas.winfo_rootx()
        y = canvas.winfo_rooty()
        width = canvas.winfo_width()
        height = canvas.winfo_height()

        # 确保尺寸有效
        if width <= 0 or height <= 0:
            width = max(width, 600)
            height = max(height, 400)

        # 定义捕获区域
        monitor = {"top": y, "left": x, "width": width, "height": height}

        # 截取区域
        with mss.mss() as sct:
            img = np.array(sct.grab(monitor))

        # 转换格式（mss使用BGRA格式转为RGB）
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # 转换为PIL图像
        return Image.fromarray(img)

    except ImportError:
        print("无法导入mss模块，请安装: pip install mss")
        return None
    except Exception as e:
        print(f"使用mss截图时发生错误: {e}")
        return None


def pil_to_mnist_format(pil_image):
    """
    将PIL图像转换为MNIST格式（黑底白字）

    参数:
        pil_image: PIL格式的图像

    返回:
        MNIST格式的PIL灰度图像（黑底白字）
    """
    if pil_image is None:
        return None

    try:
        # 转换为灰度图
        gray_image = pil_image.convert('L')

        # 转换为numpy数组进行处理
        img_array = np.array(gray_image)


        # 需要转换为MNIST格式（黑底白字：背景0，前景255）
        mnist_format = 255 - img_array  # 反转：黑底白字


        # 转换回PIL图像
        return Image.fromarray(mnist_format, mode='L')

    except Exception as e:
        print(f"转换为MNIST格式时出错: {e}")
        return None


def capture_canvas(canvas):
    """
    使用多种方法尝试捕获画布内容，并返回MNIST格式的PIL图像

    参数:
        canvas: Tkinter画布对象

    返回:
        MNIST格式的PIL图像（黑底白字）
    """
    # 方法1: 直接从画布内容绘制图像 - 最可靠的方法
    try:
        # 获取画布尺寸
        width = canvas.winfo_width()
        height = canvas.winfo_height()

        # 确保尺寸有效
        if width <= 0 or height <= 0:
            width = max(width, 600)
            height = max(height, 400)

        # 创建一个与画布相同大小的PIL图像（白底）
        pil_image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(pil_image)

        # 获取所有绘制项目
        items = canvas.find_all()

        # 如果没有绘制任何内容
        if not items:
            print("画布为空")
            # 返回空白的MNIST格式图像（全黑）
            return Image.new('L', (width, height), color=0)

        # 绘制所有线条（黑色笔迹）
        for item in items:
            # 获取线条坐标
            coords = canvas.coords(item)
            if len(coords) < 4:  # 确保有足够的坐标点
                continue

            # 线条宽度
            try:
                width_val = canvas.itemcget(item, 'width')
                width_val = int(float(width_val))
            except:
                width_val = 3  # 默认宽度

            # 绘制线段
            for i in range(0, len(coords) - 2, 2):
                draw.line(
                    [coords[i], coords[i + 1], coords[i + 2], coords[i + 3]],
                    fill='black',  # 黑色笔迹
                    width=width_val
                )

        # 转换为MNIST格式
        mnist_image = pil_to_mnist_format(pil_image)
        if mnist_image is not None:
            return mnist_image

    except Exception as e:
        print(f"直接绘制画布内容失败: {e}")

    # 方法2: 尝试其他截图方法
    img = None

    # 在Windows上使用win32gui/win32ui
    if sys.platform == 'win32':
        img = capture_canvas_windows(canvas)

    # 如果方法1失败，尝试使用PIL的ImageGrab
    if img is None:
        img = capture_canvas_pil(canvas)

    # 如果方法2失败，尝试使用mss库
    if img is None:
        img = capture_canvas_mss(canvas)

    # 如果所有方法都失败，返回空白图像
    if img is None:
        print("所有截图方法均失败，创建空白图像")
        return Image.new('L', (400, 600), color=0)  # MNIST格式的空白图像

    # 将捕获的图像转换为MNIST格式
    return pil_to_mnist_format(img)