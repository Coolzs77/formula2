# 手写数学公式识别系统使用说明

**项目名称**: Hand-written Formula Recognition System  
**更新时间**: 2025-07-19 16:17:19 北京时间  
**版本**: v2.4

## 1. 项目概述

本项目是一个基于深度学习的手写数学公式识别系统，能够识别手写的数字(0-9)和数学符号(+、-、×、÷、.)以及左右括号()，并进行公式识别和计算。系统使用残差卷积神经网络(ResNet-based CNN)，支持从文件加载图像和画布手写输入两种识别方式。

### 1.1 系统特性
- **12层残差神经网络**: 使用ConvBlock和ResidualBlock构建
- **智能符号分割**: 支持垂直结构符号(如除号÷)的正确识别
- **双输入支持**: 文件加载和实时手写输入
- **实时计算**: 支持基本数学运算和符号运算
- **可视化界面**: 直观的GUI界面，支持历史记录查看
- **数据增强**: 支持多种数据增强技术提升模型性能

### 1.2 支持的符号类别
| 类别ID | 符号 | 类型 | 描述 |
|:------:|:----:|:----:|:----:|
| 0-9 | 0,1,2,3,4,5,6,7,8,9 | 数字 | 基本数字字符 |
| 10 | + | 运算符 | 加法符号 |
| 11 | - | 运算符 | 减法符号 |
| 12 | × | 运算符 | 乘法符号 |
| 13 | ÷ | 运算符 | 除法符号 |
| 14 | . | 符号 | 小数点 |
| 15 | ( | 符号 | 左括号 |
| 16 | ) | 符号 | 右括号 |

**总计**: 17个类别

## 2. 系统要求

### 2.1 硬件要求
- **内存**: 最低4GB RAM，推荐8GB或以上
- **存储**: 约3GB可用磁盘空间
- **显卡**: 可选CUDA支持的GPU(用于加速训练)

### 2.2 软件要求
- **Python**: 3.7+ (推荐Python 3.8或更高版本)
- **操作系统**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+

### 2.3 依赖库
```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.3.0
Pillow>=8.0.0
sympy>=1.8.0
tqdm>=4.62.0
scikit-learn>=1.0.0
seaborn>=0.11.0
```


## 3. 目录结构

```
formula2/
├── data/                       # 数据目录
│   ├── data_black_white/       # 统一格式的数据集(黑底白字)
│   │   └── math_symbols_split/ # 处理后的符号数据
│   │       ├── train/          # 训练集
│   │       ├── val/            # 验证集
│   │       └── test/           # 测试集
│   ├── math_symbols/           # 原始符号数据
│   ├── math_symbols_split/     # 划分后的符号数据
│   ├── mnist/                  # MNIST数据集(自动下载)
│   └── mnist_split/            # 划分后的MNIST数据
├── model/                      # 模型保存目录
│   ├── best_symbol_model.pth   # 最佳模型
│   ├── symbol_model.pth        # 最终模型
│   └── training_history.png    # 训练曲线图
├── src/                        # 源代码目录
│   ├── data/                   # 数据处理模块
│   │   ├── dataset.py          # 数据集加载
│   │   └── preprocessing.py    # 图像预处理
│   ├── models/                 # 模型定义
│   │   ├── config.py           # 模型配置
│   │   └── symbol_recognizer.py # 识别模型
│   ├── recognition/            # 识别逻辑
│   │   └── formula.py          # 公式处理
│   └── ui/                     # 用户界面
│       ├── app.py              # 主应用界面
│       └── canvas.py           # 画布处理
|       └── colors.py           # 界面组间颜色设置
├── scripts/                    # 脚本目录
│   ├── train.py                # 训练脚本
│   ├── test.py                 # 测试脚本
│   └── valid.py                # 验证脚本
├── test photos/                # 包含可供作为“加载图像”的测试图片
│   ├── t1.png                   # 测试图片1
│   ├── t2.png                   # 测试图片2
│   └── t3.png                   # 测试图片3
├── augment_symbols.py          # 数据增强脚本
├── adjust_symbol_thickness.py  # 符号粗细调整脚本
├── revert.py                   # 图像颜色处理脚本
├── sign_split.py               # 符号数据划分脚本
├── mnist_split.py              # MNIST数据划分脚本
├── main.py                     # 主程序入口
└── README.md                  # 本文档
```
### 5 数据集结构
```
data/
├── data_black_white/           # 统一格式数据集(最终使用的数据集)
│   └── math_symbols_split/     # 处理后的符号数据
│       ├── train/              # 训练集(70%)
│       ├── val/                # 验证集(15%)
│       └── test/               # 测试集(15%)
├── math_symbols/               # 原始符号数据
│   ├── 10/                     # +号数据
│   ├── 11/                     # -号数据
│   ├── 12/                     # ×号数据
│   ├── 13/                     # ÷号数据
│   └── 14/                     # .号数据
│   └── 15/                     # (号数据
│   └── 16/                     # )号数据
├── math_symbols_split/         # 划分后的原始符号数据
├── mnist/                      # MNIST数据集
└── mnist_split/                # 划分后的MNIST数据
    ├── mnist_train.pkl
    ├── mnist_val.pkl
    └── mnist_test.pkl
```
## 6. 快速开始

### 6.2 训练模型
```bash
python scripts/train.py --epochs 15 --batch_size 64 --lr 0.001
```

### 6.3 启动图形界面
```bash
python main.py
```

## 7. 详细使用指南

### 7.1 图形界面使用

#### 启动界面
```bash
python main.py --ui
```

#### 界面布局
- **左侧上方**: 图像显示区域，显示加载的图像或预处理结果
- **左侧下方**: 手写输入区域，支持鼠标绘制，画笔工具栏
- **右侧上方**: 操作按钮(加载图像、识别公式、清除画布)，提供状态显示
- **右侧中间**: 识别结果显示(公式、LaTeX、计算结果)，支持复制结果
- **右侧下方**: 历史记录区域(可折叠)

#### 使用步骤

**方法一：文件识别**
1. 点击"加载图像"按钮
2. 选择包含手写公式的图像文件(.png, .jpg, .jpeg, .bmp)
3. 点击"识别公式"按钮
4. 查看识别结果

**方法二：手写识别**
1. 在左下方画布区域用鼠标手写数学公式
2. 点击"识别公式"按钮
3. 查看识别结果
4. 使用"清除画布"可重新开始

### 7.2 命令行使用

#### 主程序参数
```bash
python main.py [选项]

选项:
  --train              训练新模型
  --ui                 启动图形用户界面
  --epochs EPOCHS      训练轮数 (默认: 15)
  --batch_size SIZE    训练批次大小 (默认: 64)
```

#### 训练脚本
```bash
python scripts/train.py [选项]

选项:
  --epochs EPOCHS      训练轮数 (默认: 15)
  --batch_size SIZE    批次大小 (默认: 64)
  --lr LEARNING_RATE   学习率 (默认: 0.001)
```

#### 测试脚本
```bash
python scripts/test.py [选项]

选项:
  --batch_size SIZE    批次大小 (默认: 64)
  --model_dir DIR      模型目录 (默认: model)
  --model_name NAME    模型文件名 (默认: best_symbol_model.pth)
  --plot_samples       绘制样本预测图
  --num_samples NUM    显示的样本数量 (默认: 20)
```

#### 验证脚本
```bash
python scripts/valid.py [选项]

选项:
  --batch_size SIZE    批次大小 (默认: 64)
  --model_dir DIR      模型目录 (默认: model)
  --model_path PATH    指定单个模型路径
  --plot               绘制对比图表
```

### 7.3 数据处理工具详解

#### 数据增强工具
```bash
python augment_symbols.py [选项]

选项:
  --dir DIR            符号数据集目录路径 (默认: ./data/math_symbols)
  --factor FACTOR      每个原始图像生成的增强样本数量 (默认: 10)
  --no-scaling         禁用尺寸缩放处理，只应用随机增强
```

增强效果包括：
- **随机旋转**: -15°到15°范围内旋转
- **随机平移**: 最大平移量为图像宽度的10%
- **对比度调整**: 0.7到1.3倍对比度变化
- **亮度调整**: 0.7到1.3倍亮度变化
- **高斯噪声**: 标准差为5的随机噪声
- **弹性变换**: 模拟手写风格变化
- **随机模糊**: 半径0到0.8的高斯模糊

#### 图像处理工具
```bash
python revert.py
```

处理功能：
- 自动检测图像颜色模式(白底黑字 vs 黑底白字)
- 智能颜色翻转，统一为黑底白字格式
- 批量处理整个数据集
- 生成处理报告和统计信息
- 保持原始文件，同时生成标准格式文件
