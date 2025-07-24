#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手写公式识别系统 - 优化版主应用界面
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
import time

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 导入我们的色彩方案
from src.ui.colors import COLOR_SCHEME, FONTS, apply_style

# 导入必要的模块
from src.models.symbol_recognizer import SymbolRecognizer
from src.models.config import SYMBOL_CLASSES, MODEL_CONFIG
from src.data.preprocessing import preprocess_image, process_image
from src.recognition.formula import recognize_formula, generate_visualization, formula_to_latex
from src.ui.canvas import capture_canvas


class HandwrittenFormulaRecognitionApp:
    """手写公式识别应用的优化主界面类"""

    def __init__(self, root):
        """初始化应用"""
        self.root = root
        self.root.title("手写数学公式识别系统")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # 设置背景色
        self.root.configure(bg=COLOR_SCHEME['background'])

        # 设置设备(CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 绘图变量
        self.last_x = None
        self.last_y = None
        self.brush_width = 5
        self.brush_color = "#000000"
        self.tool_var = tk.StringVar(value="pen")  # 默认工具是画笔

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

        # 应用样式
        self.style = ttk.Style()
        apply_style(self.style)

        # 创建UI
        self._create_ui()

        # 在后台线程中加载模型
        threading.Thread(target=self._load_model_async).start()

    def _create_ui(self):
        """创建用户界面"""
        # 设置主框架布局
        self.root.columnconfigure(0, weight=8)  # 左侧占比更大
        self.root.columnconfigure(1, weight=5)  # 右侧占比适中
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        # 创建左侧和右侧框架
        left_frame = ttk.Frame(self.root, padding=(10, 10, 5, 10))
        left_frame.grid(row=0, column=0, sticky="nsew")

        self.right_frame = ttk.Frame(self.root, padding=(5, 10, 10, 10))
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # 配置左侧框架的网格布局
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)

        # 图像显示区域
        image_frame = ttk.LabelFrame(left_frame, text="图像显示", padding=(5, 5))
        image_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        # 使用黑色背景画布
        self.image_canvas = tk.Canvas(
            image_frame,
            bg="black",
            highlightthickness=0
        )
        self.image_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 手写输入区域
        self.canvas_frame = ttk.LabelFrame(left_frame, text="手写输入区域", padding=(5, 5))
        self.canvas_frame.grid(row=1, column=0, sticky="nsew")
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.rowconfigure(1, weight=1)  # 工具栏在索引0，画布在索引1

        # 创建画布工具栏
        self._create_canvas_toolbar()

        # 创建画布 - 设置为白色背景
        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg=COLOR_SCHEME['canvas_bg'],
            highlightthickness=1,
            highlightbackground=COLOR_SCHEME['border']
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._end_draw)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # 配置右侧框架
        self.right_frame.columnconfigure(0, weight=1)
        self.right_frame.rowconfigure(0, weight=0)  # 按钮区域
        self.right_frame.rowconfigure(1, weight=0)  # 状态区域
        self.right_frame.rowconfigure(2, weight=0)  # 识别结果区域
        self.right_frame.rowconfigure(3, weight=0)  # 历史记录按钮
        self.right_frame.rowconfigure(4, weight=1)  # 历史记录区域

        # 按钮区域
        button_frame = ttk.Frame(self.right_frame)
        button_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        # 美化按钮设计
        self.load_button = ttk.Button(
            button_frame,
            text="📂 加载图像",
            command=self._load_image
        )
        self.load_button.grid(row=0, column=0, padx=5, pady=10, sticky="ew")

        self.recognize_button = ttk.Button(
            button_frame,
            text="🔍 识别公式",
            command=self._recognize_formula,
            state=tk.DISABLED
        )
        self.recognize_button.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        self.clear_button = ttk.Button(
            button_frame,
            text="🗑️ 清除画布",
            command=self._clear_canvas
        )
        self.clear_button.grid(row=0, column=2, padx=5, pady=10, sticky="ew")

        # 添加反馈元素
        self._add_feedback_elements()

        # 创建优化的结果显示区域
        self._create_results_area()

        # 历史记录按钮
        self.history_button = ttk.Button(
            self.right_frame,
            text="📜 显示/隐藏历史记录",
            command=self._toggle_history
        )
        self.history_button.grid(row=3, column=0, sticky="ew", pady=(10, 5))

        # 历史记录区域 - 初始隐藏
        self.history_frame = ttk.LabelFrame(self.right_frame, text="识别历史")
        # 不立即添加到网格

        # 优化历史记录显示
        self.history_text = tk.Text(
            self.history_frame,
            wrap=tk.WORD,
            font=FONTS['mono'],
            bg=COLOR_SCHEME['card_bg'],
            fg=COLOR_SCHEME['text_primary'],
            borderwidth=1,
            relief="solid",
            height=10,
            width=1
        )
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 状态栏
        self.status_bar = ttk.Label(
            self.root,
            text=f"模型状态: 准备中 | 设备: {self.device}",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(10, 2)
        )
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _create_canvas_toolbar(self):
        """创建画布工具栏"""
        # 工具栏框架
        self.toolbar_frame = ttk.Frame(self.canvas_frame)
        self.toolbar_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=2)

        # 笔刷大小
        ttk.Label(self.toolbar_frame, text="笔触:", font=FONTS['small']).pack(side=tk.LEFT, padx=(5, 2))
        self.brush_size = tk.IntVar(value=5)
        brush_combo = ttk.Combobox(self.toolbar_frame, textvariable=self.brush_size,
                                   values=[1,2,3,4,5,6,7,8,9], width=3, state="readonly")
        brush_combo.pack(side=tk.LEFT, padx=2)
        brush_combo.bind("<<ComboboxSelected>>", self._update_brush)

        # 颜色选择
        ttk.Label(self.toolbar_frame, text="颜色:", font=FONTS['small']).pack(side=tk.LEFT, padx=(10, 2))
        self.brush_color = tk.StringVar(value="#000000")
        colors = ["#000000", "#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

        for color in colors:
            btn = tk.Button(self.toolbar_frame, bg=color, width=2, height=1,
                            command=lambda c=color: self._set_brush_color(c))
            btn.pack(side=tk.LEFT, padx=1)

        # 分割线
        ttk.Separator(self.toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # 工具按钮 - 使用图标或文本
        self.tool_var = tk.StringVar(value="pen")

        # 画笔工具
        pen_btn = ttk.Button(self.toolbar_frame, text="✏️画笔", width=8,
                             command=lambda: self._set_tool("pen"))
        pen_btn.pack(side=tk.LEFT, padx=2)

        # 橡皮擦
        eraser_btn = ttk.Button(self.toolbar_frame, text="🧹橡皮", width=8,
                                command=lambda: self._set_tool("eraser"))
        eraser_btn.pack(side=tk.LEFT, padx=2)

        # 分割线
        ttk.Separator(self.toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        # 清除按钮(移动到工具栏)
        clear_btn = ttk.Button(self.toolbar_frame, text="🗑️清除", width=8,
                               command=self._clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=2)

        # 提示标签 - 显示当前模式
        self.tool_label = ttk.Label(self.toolbar_frame, text="模式: 画笔", font=FONTS['small'])
        self.tool_label.pack(side=tk.RIGHT, padx=5)

    def _add_feedback_elements(self):
        """添加交互反馈元素 - 删除进度条版本"""
        # 状态指示区域 (在按钮下方)
        status_frame = ttk.Frame(self.right_frame)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5), padx=5)

        # 状态标签
        status_label_title = ttk.Label(
            status_frame,
            text="识别状态:",
            font=FONTS['main_bold']
        )
        status_label_title.pack(side=tk.LEFT, padx=5)

        # 状态指示器（彩色方块）
        self.status_indicator_canvas = tk.Canvas(
            status_frame,
            width=20,
            height=20,
            bg=COLOR_SCHEME['background'],
            highlightthickness=0
        )
        self.status_indicator_canvas.pack(side=tk.LEFT, padx=5)
        self.status_indicator_rect = self.status_indicator_canvas.create_rectangle(
            0, 0, 20, 20,
            fill=COLOR_SCHEME['success'],
            outline=""
        )

        # 状态提示文本
        self.status_label = ttk.Label(
            status_frame,
            text="就绪",
            font=FONTS['main']
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        # 添加提示信息功能(tooltip)
        self._add_tooltips()

    def _add_tooltips(self):
        """为组件添加悬停提示 - 改进版，防止闪烁"""
        tooltips = {
            self.load_button: "从文件加载图像",
            self.recognize_button: "识别手写公式",
            self.clear_button: "清除当前画布内容",
        }

        # 创建一个全局的tooltip窗口和计时器
        self.tooltip_window = None
        self.tooltip_timer = None

        def show_tooltip(widget, text):
            """延迟显示tooltip，并确保位置正确"""

            def _show():
                nonlocal widget, text
                # 如果已经有tooltip，先销毁
                hide_tooltip()

                # 计算更好的位置 - 在按钮下方中央
                x = widget.winfo_rootx() + widget.winfo_width() // 2
                y = widget.winfo_rooty() + widget.winfo_height() + 5

                # 创建tooltip窗口
                self.tooltip_window = tk.Toplevel(widget)
                self.tooltip_window.wm_overrideredirect(True)  # 无边框窗口

                # 使tooltip窗口位于所有窗口之上
                self.tooltip_window.attributes('-topmost', True)

                # 添加tooltip内容
                label = ttk.Label(
                    self.tooltip_window,
                    text=text,
                    background=COLOR_SCHEME['accent'],
                    foreground="white",
                    relief="solid",
                    borderwidth=1,
                    font=FONTS['small'],
                    padding=(5, 2)
                )
                label.pack()

                # 调整位置，使tooltip在按钮下方居中
                tooltip_width = label.winfo_reqwidth()
                self.tooltip_window.wm_geometry(f"+{x - tooltip_width // 2}+{y}")

            # 清除之前的计时器（如果有）
            if self.tooltip_timer:
                self.root.after_cancel(self.tooltip_timer)

            # 设置延迟显示tooltip (300ms延迟)
            self.tooltip_timer = self.root.after(300, _show)

        def hide_tooltip():
            """隐藏tooltip并清理资源"""
            # 清除计时器
            if self.tooltip_timer:
                self.root.after_cancel(self.tooltip_timer)
                self.tooltip_timer = None

            # 销毁tooltip窗口
            if self.tooltip_window:
                self.tooltip_window.destroy()
                self.tooltip_window = None

        # 为每个按钮绑定事件
        for widget, text in tooltips.items():
            widget.bind("<Enter>", lambda event, w=widget, t=text: show_tooltip(w, t))
            widget.bind("<Leave>", lambda event: hide_tooltip())
    def _create_results_area(self):
        """创建优化的结果显示区域"""
        # 使用卡片式设计
        results_frame = ttk.LabelFrame(self.right_frame, text="识别结果")
        results_frame.grid(row=2, column=0, sticky="ew", pady=(5, 10), padx=8)

        # 公式显示区域
        formula_frame = ttk.Frame(results_frame)
        formula_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        # 使用图标和更清晰的布局
        formula_icon = tk.Label(formula_frame, text="🔢", font=('Arial', 14),
                                bg=COLOR_SCHEME['background'], fg=COLOR_SCHEME['primary'])
        formula_icon.pack(side=tk.LEFT, padx=(0, 5))

        formula_label = ttk.Label(formula_frame, text="识别公式:",
                                  font=FONTS['main_bold'], foreground=COLOR_SCHEME['primary'])
        formula_label.pack(side=tk.LEFT, padx=5)

        # 公式显示 - 使用Entry增强可读性
        self.formula_var = tk.StringVar()
        self.formula_entry = ttk.Entry(formula_frame, textvariable=self.formula_var,
                                       font=FONTS['mono'], style="Result.TEntry",
                                       width=30, state="readonly")
        self.formula_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        # LaTeX显示区域
        latex_frame = ttk.Frame(results_frame)
        latex_frame.pack(fill=tk.X, padx=10, pady=5)

        latex_icon = tk.Label(latex_frame, text="📐", font=('Arial', 14),
                              bg=COLOR_SCHEME['background'], fg=COLOR_SCHEME['primary'])
        latex_icon.pack(side=tk.LEFT, padx=(0, 5))

        latex_label = ttk.Label(latex_frame, text="LaTeX格式:",
                                font=FONTS['main_bold'], foreground=COLOR_SCHEME['primary'])
        latex_label.pack(side=tk.LEFT, padx=5)

        self.latex_var = tk.StringVar()
        self.latex_entry = ttk.Entry(latex_frame, textvariable=self.latex_var,
                                     font=FONTS['mono'], style="Result.TEntry",
                                     width=30, state="readonly")
        self.latex_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        # 计算结果区域
        result_frame = ttk.Frame(results_frame)
        result_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        result_icon = tk.Label(result_frame, text="🧮", font=('Arial', 14),
                               bg=COLOR_SCHEME['background'], fg=COLOR_SCHEME['primary'])
        result_icon.pack(side=tk.LEFT, padx=(0, 5))

        result_label = ttk.Label(result_frame, text="计算结果:",
                                 font=FONTS['main_bold'], foreground=COLOR_SCHEME['primary'])
        result_label.pack(side=tk.LEFT, padx=5)

        # 计算结果 - 使用高亮显示
        self.result_var = tk.StringVar()
        self.result_entry = ttk.Entry(result_frame, textvariable=self.result_var,
                                      font=FONTS['result'], style="Result.TEntry",
                                      width=30, state="readonly")
        self.result_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        # 复制按钮区域
        copy_frame = ttk.Frame(results_frame)
        copy_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        # 添加复制按钮
        copy_formula_btn = ttk.Button(copy_frame, text="复制公式", width=10,
                                      command=lambda: self._copy_to_clipboard(self.formula_var.get()))
        copy_formula_btn.pack(side=tk.LEFT, padx=5)

        copy_latex_btn = ttk.Button(copy_frame, text="复制LaTeX", width=10,
                                    command=lambda: self._copy_to_clipboard(self.latex_var.get()))
        copy_latex_btn.pack(side=tk.LEFT, padx=5)

        copy_result_btn = ttk.Button(copy_frame, text="复制结果", width=10,
                                     command=lambda: self._copy_to_clipboard(self.result_var.get()))
        copy_result_btn.pack(side=tk.LEFT, padx=5)

    def _on_canvas_resize(self, event):
        """处理画布尺寸变化事件"""
        # 可以在这里添加画布重绘或其他处理逻辑
        pass

    def _start_draw(self, event):
        """开始手绘"""
        self.last_x = event.x
        self.last_y = event.y
        # 设置图像来源为canvas
        self.image_source = 'canvas'

    def _draw(self, event):
        """手绘过程，支持画笔和橡皮擦模式"""
        if self.last_x and self.last_y:
            if self.tool_var.get() == "pen":
                # 画笔模式
                self.canvas.create_line(
                    self.last_x, self.last_y, event.x, event.y,
                    width=self.brush_size.get(),
                    fill=self.brush_color.get(),
                    capstyle=tk.ROUND,
                    smooth=tk.TRUE,
                    splinesteps=36,
                    tags="drawing"
                )
            else:
                # 橡皮擦模式 - 创建白色圆形擦除现有内容
                eraser_size = self.brush_size.get() * 2
                x1 = event.x - eraser_size
                y1 = event.y - eraser_size
                x2 = event.x + eraser_size
                y2 = event.y + eraser_size

                # 查找并删除在橡皮擦区域内的所有绘图对象
                overlapping = self.canvas.find_overlapping(x1, y1, x2, y2)
                for item_id in overlapping:
                    if "drawing" in self.canvas.gettags(item_id):
                        self.canvas.delete(item_id)

                # 显示橡皮擦效果(临时)
                eraser = self.canvas.create_oval(
                    x1, y1, x2, y2,
                    outline=COLOR_SCHEME['accent'],
                    width=1
                )
                self.canvas.after(50, lambda: self.canvas.delete(eraser))

        self.last_x = event.x
        self.last_y = event.y

    def _update_brush(self, event=None):
        """更新画笔大小"""
        self.brush_width = self.brush_size.get()

    def _set_brush_color(self, color):
        """设置画笔颜色"""
        self.brush_color.set(color)

    def _set_tool(self, tool):
        """设置当前工具"""
        self.tool_var.set(tool)
        if tool == "pen":
            self.tool_label.config(text="模式: 画笔")
            self.canvas.config(cursor="pencil")
        else:  # eraser
            self.tool_label.config(text="模式: 橡皮")
            self.canvas.config(cursor="circle")

    def _load_model_async(self):
        """异步加载模型"""
        try:
            self.status_bar.config(text="正在加载模型...")

            # 创建模型实例
            self.model = SymbolRecognizer(num_classes=len(SYMBOL_CLASSES))

            # 尝试加载最佳模型，如果不存在则加载普通模型
            model_path = os.path.join(project_root, 'model', 'best_symbol_model.pth')
            if not os.path.exists(model_path):
                model_path = os.path.join(project_root, 'model', 'symbol_model.pth')

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

        # 步骤 4: 决定是缩小还是直接使用
        if img_h > canvas_height or img_w > canvas_width:
            # 如果图像比画布大，则计算缩放比例并缩小
            scale = min(canvas_height / img_h, canvas_width / img_w)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            content_to_place = cv2.resize(content_to_place, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 步骤 5: 计算粘贴位置，以确保内容在黑色背景上居中
        final_content_h, final_content_w = content_to_place.shape[:2]
        x_offset = (canvas_width - final_content_w) // 2
        y_offset = (canvas_height - final_content_h) // 2

        # 步骤 6: 将内容"粘贴"到黑色背景的中央
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

    def _clear_canvas(self):
        """清除画布内容"""
        self.canvas.delete("all")
        self.last_x = None
        self.last_y = None
        self.status_bar.config(text=f"模型状态: {'已加载' if self.model_loaded else '准备中'} | 设备: {self.device}")

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

    def _toggle_history(self):
        """切换历史记录显示/隐藏"""
        if self.history_visible:
            self.history_frame.grid_forget()
            self.history_visible = False
        else:
            self.history_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 5))
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

    def _show_processing_feedback(self, message="处理中..."):
        """显示处理反馈 - 不使用进度条"""
        # 更新状态文本
        self.status_label.config(text=message)

        # 更改状态指示器颜色为处理中(橙色)
        self.status_indicator_canvas.itemconfig(
            self.status_indicator_rect,
            fill=COLOR_SCHEME['warning']
        )

        # 禁用识别按钮
        self.recognize_button.config(state=tk.DISABLED)

        # 更新UI
        self.root.update_idletasks()
        self.root.update()

    def _hide_processing_feedback(self, message="完成"):
        """隐藏处理反馈 - 不使用进度条"""
        # 更新状态文本
        self.status_label.config(text=message)

        # 将状态指示器恢复为成功颜色(绿色)
        self.status_indicator_canvas.itemconfig(
            self.status_indicator_rect,
            fill=COLOR_SCHEME['success']
        )

        # 如果模型已加载，启用识别按钮
        if self.model_loaded:
            self.recognize_button.config(state=tk.NORMAL)

    def _recognize_formula(self):
        """识别加载的图像或手绘图像中的数学公式 - 增强版"""
        if not self.model_loaded:
            messagebox.showinfo("提示", "模型尚未加载完成，请稍候")
            return

        # 显示处理反馈
        self._show_processing_feedback("正在识别公式...")

        try:
            image = None
            current_source = self.image_source

            if current_source == 'file':
                image = self.original_image
            else:  # 从画布获取
                image = self._get_image_from_canvas()
                current_source = 'canvas'

            if image is None:
                self._hide_processing_feedback("就绪")
                messagebox.showinfo("提示", "请先加载图像或在画布上绘制")
                return

            # 分割符号
            symbols_list, processed_image = process_image(image, source=current_source)

            # 检查是否成功分割出符号
            if not symbols_list:
                self._hide_processing_feedback("未识别到符号")
                messagebox.showinfo("提示", "未能识别出任何符号，请尝试更清晰的手写或更换图像")
                return

            # 识别公式
            formula_string, evaluation_result, recognition_results = recognize_formula(
                self.model, symbols_list, self.device
            )

            # 生成LaTeX
            latex_string = formula_to_latex(recognition_results)

            # 创建可视化
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            if canvas_width <= 1: canvas_width = 600
            if canvas_height <= 1: canvas_height = 400

            # 创建背景
            final_display_bg = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # 计算缩放因子
            content_h, content_w = processed_image.shape[:2]
            scale = min(canvas_width / content_w, canvas_height / content_h, 1.0)
            new_w, new_h = int(content_w * scale), int(content_h * scale)

            # 计算偏移量使内容居中
            x_offset = (canvas_width - new_w) // 2
            y_offset = (canvas_height - new_h) // 2

            # 处理与粘贴内容
            resized_content = cv2.resize(processed_image, (new_w, new_h))
            final_display_bg[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = cv2.cvtColor(resized_content,
                                                                                                  cv2.COLOR_GRAY2BGR)

            # 生成可视化
            visualization_result = generate_visualization(
                final_display_bg,
                recognition_results,
                offset=(x_offset, y_offset),
                scale=scale
            )
            self._display_image(visualization_result)

            # 显示结果
            result_str = str(evaluation_result) if evaluation_result is not None else "无法计算"
            self.formula_var.set(formula_string)
            self.latex_var.set(latex_string)
            self.result_var.set(result_str)

            # 添加到历史记录
            self._add_to_history(formula_string, result_str)

            # 隐藏处理反馈
            self._hide_processing_feedback("识别完成")

        except Exception as e:
            self._hide_processing_feedback("识别失败")
            # 显示详细错误信息
            self._show_error_details(e)

    def _end_draw(self, event):
        """结束绘图操作"""
        # 保存绘图状态
        self.last_x = None
        self.last_y = None

    def _copy_to_clipboard(self, text):
        """复制文本到剪贴板"""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

        # 显示提示消息
        self.status_label.config(text="已复制到剪贴板")
        self.status_label.after(1500, lambda: self.status_label.config(text="就绪"))

    def _show_error_details(self, error):
        """显示美化的错误详情对话框"""
        import traceback

        error_win = tk.Toplevel(self.root)
        error_win.title("错误详情")
        error_win.geometry("600x400")
        error_win.configure(bg=COLOR_SCHEME['background'])

        # 添加图标和标题
        header = ttk.Frame(error_win)
        header.pack(fill=tk.X, padx=20, pady=10)

        # 错误图标
        error_icon = tk.Label(header, text="❌", font=('Arial', 36),
                              fg=COLOR_SCHEME['error'],
                              bg=COLOR_SCHEME['background'])
        error_icon.pack(side=tk.LEFT, padx=(0, 10))

        # 错误信息
        error_info = ttk.Frame(header)
        error_info.pack(side=tk.LEFT, fill=tk.BOTH)

        ttk.Label(error_info, text="发生错误",
                  font=FONTS['title'],
                  foreground=COLOR_SCHEME['error']).pack(anchor="w")

        ttk.Label(error_info, text=f"类型: {type(error).__name__}",
                  font=FONTS['main_bold']).pack(anchor="w", pady=(5, 0))

        # 错误详情区域
        detail_frame = ttk.LabelFrame(error_win, text="详细信息")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 使用文本区域显示堆栈跟踪
        error_text = tk.Text(detail_frame, wrap=tk.WORD, font=FONTS['mono'],
                             width=70, height=10)
        error_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(detail_frame, orient="vertical",
                                  command=error_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        error_text.configure(yscrollcommand=scrollbar.set)

        # 插入错误信息
        error_text.insert(tk.END, f"{str(error)}\n\n")
        error_text.insert(tk.END, traceback.format_exc())
        error_text.config(state="disabled")  # 设为只读

        # 添加关闭按钮
        close_btn = ttk.Button(error_win, text="关闭",
                               command=error_win.destroy,
                               width=15)
        close_btn.pack(pady=15)

        # 居中显示
        error_win.update_idletasks()
        width = error_win.winfo_width()
        height = error_win.winfo_height()
        x = (error_win.winfo_screenwidth() // 2) - (width // 2)
        y = (error_win.winfo_screenheight() // 2) - (height // 2)
        error_win.geometry('{}x{}+{}+{}'.format(width, height, x, y))

        # 设置为模态窗口
        error_win.transient(self.root)
        error_win.grab_set()
        self.root.wait_window(error_win)


# 如果直接运行此文件，则创建应用实例
if __name__ == "__main__":
    root = tk.Tk()
    app = HandwrittenFormulaRecognitionApp(root)
    root.mainloop()