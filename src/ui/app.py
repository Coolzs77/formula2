#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手写公式识别系统 - 主应用界面
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageTk
import threading

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入必要的模块
from src.models.symbol_recognizer import SymbolRecognizer
from src.models.config import SYMBOL_CLASSES
from src.data.preprocessing import preprocess_image, process_image
from src.recognition.formula import recognize_formula, generate_visualization, formula_to_latex
from src.ui.canvas import capture_canvas # 使用我们新的函数
from torchvision import transforms     # 必须导入

class HandwrittenFormulaRecognitionApp:
    """手写公式识别应用的主界面类"""

    def __init__(self, root):
        """初始化应用"""
        self.root = root
        self.root.title("手写数学公式识别系统")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)  # 设置最小窗口大小

        # 设置设备(CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 存储画布上一点的坐标
        self.last_x = None
        self.last_y = None

        # 标明是画布还是文件
        self.image_source = None  # 'file' 或 'canvas'


        # 存储图像
        self.original_image = None
        self.processed_image = None
        self.current_display_image = None

        # 加载模型
        self.model = None
        self.model_loaded = False
        self.recognition_results = None

        # 历史记录相关
        self.history_visible = False
        self.history_records = []

        # 创建UI
        self._create_ui()

        # 在后台线程中加载模型
        threading.Thread(target=self._load_model_async).start()

    def _create_ui(self):
        """创建用户界面"""
        # 创建主框架，使用Grid布局管理器
        self.root.columnconfigure(0, weight=1)  # 左侧占1份
        self.root.columnconfigure(1, weight=1)  # 右侧也占1份
        self.root.rowconfigure(0, weight=1)  # 主内容区域自动扩展
        self.root.rowconfigure(1, weight=0)  # 状态栏固定高度

        # 创建左侧和右侧框架
        left_frame = ttk.Frame(self.root, padding=(5, 5, 5, 5))
        left_frame.grid(row=0, column=0, sticky="nsew")

        right_frame = ttk.Frame(self.root, padding=(5, 5, 5, 5))
        right_frame.grid(row=0, column=1, sticky="nsew")

        # 配置左侧框架的网格布局
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)  # 图像区域 - 严格等于一半
        left_frame.rowconfigure(1, weight=1)  # 画布区域 - 严格等于一半

        # 图像显示区域（带标签）
        image_frame = ttk.LabelFrame(left_frame, text="图像显示")
        image_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        # 创建图像显示区域 - 使用Canvas确保完全填满
        self.image_canvas = tk.Canvas(image_frame, bg="black", highlightthickness=0)
        self.image_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 创建画布用于手绘输入（带标签框）
        canvas_frame = ttk.LabelFrame(left_frame, text="手写输入区域")
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # 配置右侧框架
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=0)  # 按钮区域
        right_frame.rowconfigure(1, weight=0)  # 识别结果区域
        right_frame.rowconfigure(2, weight=0)  # 历史记录按钮
        right_frame.rowconfigure(3, weight=1)  # 历史记录区域 (可折叠)

        # 按钮区域
        button_frame = ttk.Frame(right_frame)
        button_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        # 三个按钮并排显示
        self.load_button = ttk.Button(button_frame, text="加载图像", command=self._load_image)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.recognize_button = ttk.Button(button_frame, text="识别公式",
                                           command=self._recognize_formula,
                                           state=tk.DISABLED)
        self.recognize_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.clear_button = ttk.Button(button_frame, text="清除画布", command=self._clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # 识别结果区域
        results_frame = ttk.LabelFrame(right_frame, text="识别结果")
        results_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        results_frame.columnconfigure(0, weight=0)
        results_frame.columnconfigure(1, weight=1)

        # 使用Entry显示结果，参考图像中的样式
        ttk.Label(results_frame, text="公式:", width=8, anchor="w").grid(row=0, column=0, padx=5, pady=10, sticky="w")
        self.formula_var = tk.StringVar()
        self.formula_entry = ttk.Entry(results_frame, textvariable=self.formula_var, font=('Arial', 12),
                                       state="readonly")
        self.formula_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        ttk.Label(results_frame, text="LaTeX:", width=8, anchor="w").grid(row=1, column=0, padx=5, pady=10, sticky="w")
        self.latex_var = tk.StringVar()
        self.latex_entry = ttk.Entry(results_frame, textvariable=self.latex_var, font=('Arial', 12), state="readonly")
        self.latex_entry.grid(row=1, column=1, padx=5, pady=10, sticky="ew")

        ttk.Label(results_frame, text="计算:", width=8, anchor="w").grid(row=2, column=0, padx=5, pady=10, sticky="w")
        self.result_var = tk.StringVar()
        self.result_entry = ttk.Entry(results_frame, textvariable=self.result_var, font=('Arial', 12), state="readonly")
        self.result_entry.grid(row=2, column=1, padx=5, pady=10, sticky="ew")

        # 历史记录按钮
        self.history_button = ttk.Button(right_frame, text="显示/隐藏历史记录", command=self._toggle_history)
        self.history_button.grid(row=2, column=0, sticky="ew", pady=(5, 5))

        # 历史记录区域 - 初始隐藏
        self.history_frame = ttk.LabelFrame(right_frame, text="识别历史")
        # 不立即添加到网格

        # 在历史记录框架中添加文本框
        self.history_text = tk.Text(self.history_frame, wrap=tk.WORD, font=('Arial', 10), height=10)
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 状态栏
        self.status_bar = ttk.Label(self.root, text=f"模型状态: 准备中 | 设备: {self.device}", relief=tk.SUNKEN,
                                    anchor=tk.W)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _on_canvas_resize(self, event):
        """处理画布尺寸变化事件"""
        pass

    def _toggle_history(self):
        """切换历史记录显示/隐藏"""
        if self.history_visible:
            self.history_frame.grid_forget()
            self.history_visible = False
        else:
            self.history_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 5))
            self.history_visible = True
            # 如果有历史记录，刷新显示
            self._refresh_history()

    def _refresh_history(self):
        """刷新历史记录显示"""
        if hasattr(self, 'history_text'):
            self.history_text.config(state=tk.NORMAL)
            self.history_text.delete(1.0, tk.END)

            # 显示所有历史记录
            for timestamp, formula, result in self.history_records:
                self.history_text.insert(tk.END, f"[{timestamp}] {formula} = {result}\n\n")

            self.history_text.config(state=tk.DISABLED)

    def _load_model_async(self):
        """异步加载模型"""
        try:
            self.status_bar.config(text="正在加载模型...")

            # 创建模型实例
            self.model = SymbolRecognizer(num_classes=len(SYMBOL_CLASSES))

            # 尝试加载最佳模型，如果不存在则加载普通模型
            model_path = os.path.join('model', 'best_symbol_model.pth')
            if not os.path.exists(model_path):
                model_path = os.path.join('model', 'symbol_model.pth')

            if os.path.exists(model_path):
                # 使用weights_only=True来避免FutureWarning
                self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                self.model.to(self.device)
                self.model.eval()

                self.model_loaded = True
                self.recognize_button.config(state=tk.NORMAL)
                self.status_bar.config(text=f"模型状态: 已加载 | 设备: {self.device}")
            else:
                messagebox.showerror("错误", "未找到模型文件。请先训练模型或检查模型路径。")
                self.status_bar.config(text=f"模型状态: 加载失败 | 设备: {self.device}")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型时出错: {str(e)}")
            self.status_bar.config(text=f"模型状态: 加载错误 | 设备: {self.device}")

    def _start_draw(self, event):
        """开始手绘"""
        self.last_x = event.x
        self.last_y = event.y

    def _draw(self, event):
        """手绘过程"""
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=5, fill="black", capstyle=tk.ROUND,
                smooth=tk.TRUE, splinesteps=36
            )
        self.last_x = event.x
        self.last_y = event.y

    def _clear_canvas(self):
        """清除画布内容"""
        self.canvas.delete("all")
        self.last_x = None
        self.last_y = None
        self.status_bar.config(text=f"模型状态: 已加载 | 设备: {self.device}")

        # 清除图像变量
        self.original_image = None
        self.processed_image = None
        self.recognition_results = None

        # 清除画布时，重置图像来源
        self.image_source = None

        # 清除结果标签
        self.formula_var.set("")
        self.latex_var.set("")
        self.result_var.set("")

        # 清除图像显示
        self.image_canvas.delete("all")

    def _load_image(self):
        """加载图像文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp")]
        )

        if file_path:
            try:
                self.status_bar.config(text=f"正在处理图像: {os.path.basename(file_path)}")

                # 读取原始图像
                self.original_image = cv2.imread(file_path)
                # 明确设置图像来源为'file'
                self.image_source = 'file'
                # 显示图像
                self._display_image(self.original_image)

                self.status_bar.config(text=f"已加载图像: {os.path.basename(file_path)}")

                # 启用识别按钮
                if self.model_loaded:
                    self.recognize_button.config(state=tk.NORMAL)

            except Exception as e:
                self.status_bar.config(text=f"图像加载错误: {str(e)}")
                messagebox.showerror("错误", f"图像加载失败: {str(e)}")

    def _display_image(self, image):
        """在UI中显示图像"""

        if image is None:
            return

            # 步骤 1: 获取显示区域的实际大小
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()

        # 在程序刚启动时，画布尺寸可能为1，需要提供默认值
        if canvas_width <= 1: canvas_width = 600
        if canvas_height <= 1: canvas_height = 400

        # 步骤 2: 创建一个最终的、与显示区域等大的纯黑色背景
        final_image_bg = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # 步骤 3: 准备要放置的内容
        # 确保内容是3通道的彩色图，以便粘贴
        if len(image.shape) == 2:
            content_to_place = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            content_to_place = image.copy()

        img_h, img_w = content_to_place.shape[:2]

        # 步骤 4: 【核心逻辑】决定是缩小还是直接使用
        if img_h > canvas_height or img_w > canvas_width:
            # 如果图像比画布大，则计算缩放比例并缩小
            scale = min(canvas_height / img_h, canvas_width / img_w)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            content_to_place = cv2.resize(content_to_place, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # else:
        # 如果图像比画布小或相等，则不进行任何缩放，保持其原始尺寸

        # 步骤 5: 计算粘贴位置，以确保内容在黑色背景上居中
        final_content_h, final_content_w = content_to_place.shape[:2]
        x_offset = (canvas_width - final_content_w) // 2
        y_offset = (canvas_height - final_content_h) // 2

        # 步骤 6: 将内容“粘贴”到黑色背景的中央
        final_image_bg[y_offset: y_offset + final_content_h, x_offset: x_offset + final_content_w] = content_to_place

        # 步骤 7: 将最终合成的图像转换为Tkinter格式并显示
        img_rgb = cv2.cvtColor(final_image_bg, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(image=pil_image)

        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.image_canvas.image = photo  # 关键：保持对photo的引用
    def _get_image_from_canvas(self):
        """将画布内容转换为图像"""
        try:
            self.status_bar.config(text="正在获取画布内容...")

            # 使用修复后的capture_canvas函数，直接获得MNIST格式的PIL图像
            pil_image = capture_canvas(self.canvas)

            if pil_image is None:
                self.status_bar.config(text="画布为空，请先绘制内容")
                return None

            # 将PIL图像转换为OpenCV格式用于显示和后续处理
            img_array = np.array(pil_image)

            # 如果是灰度图，转换为3通道用于显示
            if len(img_array.shape) == 2:
                self.original_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            else:
                self.original_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # 显示图像
            self._display_image(self.original_image)
            self.status_bar.config(text="从画布获取图像成功")

            return self.original_image

        except Exception as e:
            self.status_bar.config(text=f"获取画布图像失败: {str(e)}")
            messagebox.showerror("错误", f"获取画布图像失败: {str(e)}")
            print(f"错误详情: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    def _add_to_history(self, formula, result):
        """添加识别结果到历史记录"""
        # 不再尝试直接操作可能不存在的文本控件
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 添加到历史记录列表
        self.history_records.append((timestamp, formula, result))

        # 如果历史记录可见，则刷新显示
        if self.history_visible:
            self._refresh_history()

    def _recognize_formula(self):
        """识别加载的图像或手绘图像中的数学公式"""
        if not self.model_loaded:
            messagebox.showinfo("提示", "模型尚未加载完成，请稍候")
            return

        image_to_process = None
        current_source = self.image_source

        if current_source == 'file':
            image_to_process = self.original_image
        else:  # 如果是来自画布或未指定来源
            # 【新代码】从画布获取图像，并设置来源
            image_to_process = self._get_image_from_canvas()  # 假设此函数返回一个numpy数组
            current_source = 'canvas'

        if image_to_process is None:
            messagebox.showinfo("提示", "请先加载图像或在画布上绘制")
            return

        try:
            self.status_bar.config(text="正在识别公式...")

            # 【核心修改】将图像来源作为参数传递给预处理函数
            preprocessed_image = preprocess_image(
                image_to_process,
                source = current_source  # 传递我们新的状态变量
            )
            # 显示预处理后的图像
            self._display_image(preprocessed_image)

            # 分割符号
            symbols_list, processed_image = process_image(self.original_image,source=current_source)

            # 检查是否成功分割出符号
            if not symbols_list:
                messagebox.showinfo("提示", "未能识别出任何符号，请尝试更清晰的手写或更换图像")
                self.status_bar.config(text="未识别到符号")
                return

            # 识别公式
            formula_string, evaluation_result, recognition_results = recognize_formula(
                self.model, symbols_list, self.device
            )

            # 生成LaTeX
            latex_string = formula_to_latex(recognition_results)

            # 步骤 2: 【核心】创建最终的显示图像
            # 获取显示区域的实际大小
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            if canvas_width <= 1: canvas_width = 600
            if canvas_height <= 1: canvas_height = 400

            # 创建一个最终的、与显示区域等大的黑色背景
            final_display_bg = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # 计算原始图像内容(processed_image)应该缩放到的尺寸和粘贴位置
            content_h, content_w = processed_image.shape[:2]

            scale = min(canvas_width / content_w, canvas_height / content_h, 1.0)
            new_w, new_h = int(content_w * scale), int(content_h * scale)
            # 计算居中所需的偏移量
            x_offset = (canvas_width - new_w) // 2
            y_offset = (canvas_height - new_h) // 2

            # 将处理后的干净内容（黑底白字）缩放并粘贴到黑色背景中央
            resized_content = cv2.resize(processed_image, (new_w, new_h))
            final_display_bg[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = cv2.cvtColor(resized_content,
                                                                                                  cv2.COLOR_GRAY2BGR)

            # 步骤 3: 调用新的可视化函数，在最终图像上绘制标注
            # 我们传递：1. 包含内容的黑色大图, 2. 识别结果, 3. 内容的偏移量
            visualization_result = generate_visualization(
                final_display_bg,
                recognition_results,
                offset=(x_offset, y_offset),
                scale=scale  # <--- 将计算出的scale传递进去

            )
            self._display_image(visualization_result)

            # 显示结果
            result_str = str(evaluation_result) if evaluation_result is not None else "无法计算"
            self.formula_var.set(formula_string)
            self.latex_var.set(latex_string)
            self.result_var.set(result_str)

            # 添加到历史记录
            self._add_to_history(formula_string, result_str)

            self.status_bar.config(text="公式识别完成")

        except Exception as e:
            error_msg = str(e)
            self.status_bar.config(text=f"识别错误: {error_msg[:50]}...")

            # 使用可滚动文本框显示完整错误信息
            error_window = tk.Toplevel(self.root)
            error_window.title("错误详情")
            error_window.geometry("500x300")

            error_text = tk.Text(error_window, wrap=tk.WORD)
            error_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            error_text.insert(tk.END, f"错误类型: {type(e).__name__}\n\n")
            error_text.insert(tk.END, f"错误信息: {error_msg}\n\n")

            import traceback
            error_text.insert(tk.END, "详细堆栈:\n")
            error_text.insert(tk.END, traceback.format_exc())

            # 添加关闭按钮
            close_button = ttk.Button(error_window, text="关闭", command=error_window.destroy)
            close_button.pack(pady=10)

            print(f"错误详情: {str(e)}")
            traceback.print_exc()