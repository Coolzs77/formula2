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
from src.ui.canvas import capture_canvas


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

        # 存储图像
        self.original_image = None
        self.processed_image = None
        self.current_display_image = None

        # 加载模型
        self.model = None
        self.model_loaded = False
        self.recognition_results = None

        # 创建UI
        self._create_ui()

        # 在后台线程中加载模型
        threading.Thread(target=self._load_model_async).start()

    def _create_ui(self):
        """创建用户界面"""
        # 创建主框架，使用Grid布局管理器
        self.root.columnconfigure(0, weight=4)  # 左侧占4份 - 增大比例
        self.root.columnconfigure(1, weight=1)  # 右侧占1份
        self.root.rowconfigure(0, weight=1)  # 主内容区域自动扩展

        # 创建左侧和右侧框架
        left_frame = ttk.Frame(self.root, padding=10)
        left_frame.grid(row=0, column=0, sticky="nsew")

        right_frame = ttk.Frame(self.root, padding=10)
        right_frame.grid(row=0, column=1, sticky="nsew")

        # 配置左侧框架的网格布局
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)  # 图像区域
        left_frame.rowconfigure(1, weight=2)  # 画布区域 - 增大比例

        # 图像显示区域（带滚动条）
        image_frame = ttk.LabelFrame(left_frame, text="图像显示")
        image_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        # 创建带滚动条的画布容器来显示图像
        self.image_canvas = tk.Canvas(image_frame, bg="white", highlightthickness=0)
        self.image_canvas.grid(row=0, column=0, sticky="nsew")

        # 添加滚动条
        image_xscroll = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        image_xscroll.grid(row=1, column=0, sticky="ew")
        image_yscroll = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        image_yscroll.grid(row=0, column=1, sticky="ns")

        self.image_canvas.configure(xscrollcommand=image_xscroll.set, yscrollcommand=image_yscroll.set)

        # 创建图像标签
        self.image_label = ttk.Label(self.image_canvas)
        self.image_canvas.create_window(0, 0, anchor="nw", window=self.image_label)

        # 创建画布用于手绘输入（带标签框）
        canvas_frame = ttk.LabelFrame(left_frame, text="手写输入区域")
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # 配置右侧框架
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=0)  # 按钮区域不扩展
        right_frame.rowconfigure(1, weight=1)  # 结果区域
        right_frame.rowconfigure(2, weight=0)  # 历史区域减小

        # 控制按钮区域
        button_frame = ttk.Frame(right_frame)
        button_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        # 按钮使用更紧凑的布局
        self.load_button = ttk.Button(button_frame, text="加载图像", command=self._load_image)
        self.load_button.pack(side=tk.LEFT, padx=2, pady=5, fill=tk.X, expand=True)

        self.recognize_button = ttk.Button(button_frame, text="识别公式",
                                           command=self._recognize_formula,
                                           state=tk.DISABLED)
        self.recognize_button.pack(side=tk.LEFT, padx=2, pady=5, fill=tk.X, expand=True)

        self.clear_button = ttk.Button(button_frame, text="清除画布", command=self._clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=2, pady=5, fill=tk.X, expand=True)

        # 识别结果显示（减小高度）
        results_frame = ttk.LabelFrame(right_frame, text="识别结果")
        results_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 5))
        results_frame.columnconfigure(0, weight=0)  # 标签列不扩展
        results_frame.columnconfigure(1, weight=1)  # 内容列自动扩展

        # 使用紧凑布局显示结果
        ttk.Label(results_frame, text="公式:", anchor=tk.W).grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.formula_label = scrolledtext.ScrolledText(results_frame, height=2, wrap=tk.WORD)
        self.formula_label.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.formula_label.config(state=tk.DISABLED)  # 设置为只读

        ttk.Label(results_frame, text="LaTeX:", anchor=tk.W).grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.latex_label = scrolledtext.ScrolledText(results_frame, height=2, wrap=tk.WORD)
        self.latex_label.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.latex_label.config(state=tk.DISABLED)  # 设置为只读

        ttk.Label(results_frame, text="计算:", anchor=tk.W).grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.result_label = scrolledtext.ScrolledText(results_frame, height=2, wrap=tk.WORD)
        self.result_label.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        self.result_label.config(state=tk.DISABLED)  # 设置为只读

        # 折叠历史记录区域
        history_button = ttk.Button(right_frame, text="显示/隐藏历史记录", command=self._toggle_history)
        history_button.grid(row=2, column=0, sticky="ew", pady=(5, 0))

        # 添加历史记录区域（初始隐藏）
        self.history_frame = ttk.LabelFrame(right_frame, text="识别历史")
        # 不使用grid添加到布局中，而是在需要时添加
        self.history_frame.columnconfigure(0, weight=1)
        self.history_frame.rowconfigure(0, weight=1)

        self.history_text = scrolledtext.ScrolledText(self.history_frame, height=6, wrap=tk.WORD)
        self.history_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.history_text.config(state=tk.DISABLED)  # 设置为只读
        self.history_visible = False  # 历史记录初始隐藏

        # 状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _toggle_history(self):
        """切换历史记录区域的显示/隐藏状态"""
        if self.history_visible:
            self.history_frame.grid_forget()
            self.history_visible = False
        else:
            self.history_frame.grid(row=3, column=0, sticky="nsew", pady=(5, 0))
            self.history_visible = True

    def _on_canvas_resize(self, event):
        """处理画布尺寸变化事件"""
        # 可用于调整画布内容
        pass

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
                self.status_bar.config(text=f"模型加载完成 (设备: {self.device})")
            else:
                messagebox.showerror("错误", "未找到模型文件。请先训练模型或检查模型路径。")
                self.status_bar.config(text="模型加载失败")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型时出错: {str(e)}")
            self.status_bar.config(text="模型加载失败")

    def _start_draw(self, event):
        """开始手绘"""
        self.last_x = event.x
        self.last_y = event.y

    def _draw(self, event):
        """手绘过程"""
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=3, fill="black", capstyle=tk.ROUND,
                smooth=tk.TRUE, splinesteps=36
            )
        self.last_x = event.x
        self.last_y = event.y

    def _clear_canvas(self):
        """清除画布内容"""
        self.canvas.delete("all")
        self.last_x = None
        self.last_y = None
        self.status_bar.config(text="画布已清除")

        # 清除图像变量
        self.original_image = None
        self.processed_image = None
        self.recognition_results = None

        # 清除结果标签
        self._set_text(self.formula_label, "")
        self._set_text(self.latex_label, "")
        self._set_text(self.result_label, "")

    def _load_image(self):
        """加载图像文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp")]
        )

        if file_path:
            try:
                self.status_bar.config(text=f"正在处理图像: {file_path}")

                # 读取原始图像
                self.original_image = cv2.imread(file_path)

                # 显示图像
                self._display_image(self.original_image)

                self.status_bar.config(text=f"图像已加载: {file_path}")

                # 启用识别按钮
                if self.model_loaded:
                    self.recognize_button.config(state=tk.NORMAL)

            except Exception as e:
                self.status_bar.config(text=f"图像加载错误: {str(e)}")
                messagebox.showerror("错误", f"图像加载失败: {str(e)}")

    def _display_image(self, image):
        """在UI中显示图像"""
        if image is not None:
            # 获取显示区域大小
            display_width = self.image_canvas.winfo_width()
            display_height = self.image_canvas.winfo_height()

            # 如果尺寸太小（初始化时），使用默认值
            if display_width < 50:
                display_width = 600
            if display_height < 50:
                display_height = 400

            # 获取图像尺寸
            height, width = image.shape[:2]

            # 如果图像太大，按比例缩小以适应显示区域
            if width > display_width or height > display_height:
                scale_ratio = min(display_width / width, display_height / height)
                new_width = int(width * scale_ratio)
                new_height = int(height * scale_ratio)
                display_image = cv2.resize(image, (new_width, new_height))
            else:
                display_image = image.copy()

            # 如果是浮点类型数组，转换为uint8
            if display_image.dtype == np.float32 or display_image.dtype == np.float64:
                display_image = (display_image * 255).astype(np.uint8)

            # 转换颜色通道
            if len(display_image.shape) == 2:  # 灰度图像
                display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2RGB)
            else:  # BGR图像
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

            # 将OpenCV图像转换为PIL格式
            pil_image = Image.fromarray(display_image)

            # 转换为PhotoImage格式
            photo = ImageTk.PhotoImage(image=pil_image)

            # 更新图像标签
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # 保持引用，防止垃圾回收

            # 配置滚动区域
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))

            # 保存当前显示的图像
            self.current_display_image = display_image

    def _get_image_from_canvas(self):
        """将画布内容转换为图像"""
        try:
            self.status_bar.config(text="正在获取画布内容...")

            # 获取画布尺寸
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()

            # 创建一个与画布相同大小的PIL图像
            pil_image = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(pil_image)

            # 获取所有项目
            items = self.canvas.find_all()

            # 如果没有绘制任何内容
            if not items:
                self.status_bar.config(text="画布为空，请先绘制内容")
                return

            # 绘制所有线条
            for item in items:
                # 获取线条坐标
                coords = self.canvas.coords(item)
                # 线条宽度
                width_val = self.canvas.itemcget(item, 'width')
                # 绘制线段
                for i in range(0, len(coords) - 2, 2):
                    draw.line(
                        [coords[i], coords[i + 1], coords[i + 2], coords[i + 3]],
                        fill='black',
                        width=int(float(width_val))
                    )

            # 转换为OpenCV格式
            self.original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # 确保笔迹为黑色，背景为白色
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            self.original_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            self._display_image(self.original_image)
            self.status_bar.config(text="从画布获取图像成功")

        except Exception as e:
            self.status_bar.config(text=f"获取画布图像失败: {str(e)}")
            messagebox.showerror("错误", f"获取画布图像失败: {str(e)}")
            print(f"错误详情: {str(e)}")
            import traceback
            traceback.print_exc()

    def _set_text(self, text_widget, content):
        """设置文本框内容"""
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)

    def _add_to_history(self, formula, latex, result):
        """添加识别结果到历史记录"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.history_text.config(state=tk.NORMAL)
        self.history_text.insert(tk.END, f"[{timestamp}]\n")
        self.history_text.insert(tk.END, f"公式: {formula}\n")
        self.history_text.insert(tk.END, f"结果: {result}\n")
        self.history_text.insert(tk.END, "-" * 30 + "\n")
        self.history_text.see(tk.END)  # 滚动到底部
        self.history_text.config(state=tk.DISABLED)

    def _recognize_formula(self):
        """识别加载的图像或手绘图像中的数学公式"""
        if not self.model_loaded:
            messagebox.showinfo("提示", "模型尚未加载完成，请稍候")
            return

        if self.original_image is None:
            # 如果没有加载图像，使用画布内容
            self._get_image_from_canvas()

        if self.original_image is None:
            messagebox.showinfo("提示", "请先加载图像或在画布上绘制")
            return

        try:
            self.status_bar.config(text="正在识别公式...")

            # 进行图像预处理
            preprocessed_image = preprocess_image(self.original_image)

            # 显示预处理后的图像
            self._display_image(preprocessed_image)

            # 分割符号
            symbols_list, processed_image = process_image(self.original_image)

            # 识别公式
            formula_string, evaluation_result, recognition_results = recognize_formula(
                self.model, symbols_list, self.device
            )

            # 生成LaTeX
            latex_string = formula_to_latex(recognition_results)

            # 生成可视化结果
            visualization_result = generate_visualization(self.original_image, recognition_results)
            self._display_image(visualization_result)

            # 显示结果
            result_str = str(evaluation_result) if evaluation_result is not None else "无法计算"
            self._set_text(self.formula_label, formula_string)
            self._set_text(self.latex_label, latex_string)
            self._set_text(self.result_label, result_str)

            # 添加到历史记录
            self._add_to_history(formula_string, latex_string, result_str)

            self.status_bar.config(text="公式识别完成")

        except Exception as e:
            self.status_bar.config(text=f"识别错误: {str(e)}")
            messagebox.showerror("错误", f"公式识别失败: {str(e)}")
            print(f"错误详情: {str(e)}")
            import traceback
            traceback.print_exc()