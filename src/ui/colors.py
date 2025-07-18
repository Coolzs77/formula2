#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手写公式识别系统 - 色彩方案模块
"""
import sys

# 深蓝主题配色方案 - 专业且舒适的色调
COLOR_SCHEME = {
    'primary': '#1a365d',  # 深蓝色(主色)
    'secondary': '#2a4365',  # 次要深蓝色
    'accent': '#4299e1',  # 亮蓝色(强调色)
    'accent_hover': '#63b3ed',  # 亮蓝色悬停效果

    'background': '#f8fafc',  # 浅灰背景色
    'card_bg': '#ffffff',  # 卡片背景色
    'canvas_bg': '#ffffff',  # 画布背景色

    'text_primary': '#1a202c',  # 主要文本色
    'text_secondary': '#4a5568',  # 次要文本色
    'text_disabled': '#a0aec0',  # 禁用文本色

    'success': '#48bb78',  # 成功色(绿色)
    'warning': '#ed8936',  # 警告色(橙色)
    'error': '#e53e3e',  # 错误色(红色)

    'border': '#e2e8f0',  # 边框色
    'divider': '#edf2f7',  # 分隔线

    'shadow': '#00000029'  # 阴影色
}

# 字体方案
FONTS = {
    # 主要字体
    'main': ('Microsoft YaHei UI', 10) if 'win' in sys.platform else ('Helvetica Neue', 10),
    'main_bold': ('Microsoft YaHei UI', 10, 'bold') if 'win' in sys.platform else ('Helvetica Neue', 10, 'bold'),

    # 标题字体
    'title': ('Microsoft YaHei UI', 12, 'bold') if 'win' in sys.platform else ('Helvetica Neue', 12, 'bold'),

    # 等宽字体(用于公式显示)
    'mono': ('Consolas', 11) if 'win' in sys.platform else ('Menlo', 11),

    # 结果显示字体
    'result': ('Consolas', 12, 'bold') if 'win' in sys.platform else ('Menlo', 12, 'bold'),

    # 按钮字体
    'button': ('Microsoft YaHei UI', 9) if 'win' in sys.platform else ('Helvetica Neue', 9),

    # 小文本字体
    'small': ('Microsoft YaHei UI', 8) if 'win' in sys.platform else ('Helvetica Neue', 8),
}


# 组件样式设置
def apply_style(style):
    """应用自定义ttk样式"""
    # 全局配置
    style.configure('.',
                    background=COLOR_SCHEME['background'],
                    foreground=COLOR_SCHEME['text_primary'],
                    font=FONTS['main']
                    )

    # 按钮样式
    style.configure('TButton',
                    background=COLOR_SCHEME['accent'],
                    foreground='black',
                    font=FONTS['button'],
                    padding=(10, 5)
                    )

    # 按钮悬停和按下效果
    style.map('TButton',
              background=[('active', COLOR_SCHEME['accent_hover']),
                          ('pressed', COLOR_SCHEME['secondary'])],
              foreground=[('active', 'green'),
                          ('pressed', 'green')]
              )

    # 标签样式
    style.configure('TLabel',
                    background=COLOR_SCHEME['background'],
                    foreground=COLOR_SCHEME['text_primary'],
                    font=FONTS['main']
                    )

    # 标题标签
    style.configure('Title.TLabel',
                    font=FONTS['title'],
                    foreground=COLOR_SCHEME['primary']
                    )

    # 框架样式
    style.configure('Card.TFrame',
                    background=COLOR_SCHEME['card_bg'],
                    borderwidth=1,
                    relief='solid'
                    )

    # LabelFrame样式
    style.configure('TLabelframe',
                    background=COLOR_SCHEME['card_bg'],
                    foreground=COLOR_SCHEME['text_primary'],
                    bordercolor=COLOR_SCHEME['border'],
                    font=FONTS['main_bold']
                    )

    style.configure('TLabelframe.Label',
                    background=COLOR_SCHEME['background'],
                    foreground=COLOR_SCHEME['primary'],
                    font=FONTS['main_bold']
                    )

    # 结果输入框样式
    style.configure('Result.TEntry',
                    font=FONTS['result'],
                    foreground=COLOR_SCHEME['primary'],
                    fieldbackground=COLOR_SCHEME['card_bg'],
                    bordercolor=COLOR_SCHEME['accent']
                    )

