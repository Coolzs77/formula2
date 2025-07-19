# 手写数学公式识别系统使用说明

**项目名称**: Hand-written Formula Recognition System  
**更新时间**: 2025-07-19 16:17:19 北京时间  
**用户**: Coolzs77  
**版本**: v2.2

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

## 3. 安装步骤

### 3.1 克隆项目
```bash
git clone https://github.com/Coolzs77/formula2.git
cd formula2
```

### 3.2 创建虚拟环境(推荐)
```bash
# 使用conda
conda create -n formula_env python=3.8
conda activate formula_env

# 或使用venv
python -m venv formula_env
# Windows
formula_env\Scripts\activate
# Linux/macOS
source formula_env/bin/activate
```

### 3.3 安装依赖
```bash
pip install -r requirements.txt
```

### 3.4 验证安装
```bash
python main.py --help
```

## 4. 目录结构

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
├── augment_symbols.py          # 数据增强脚本
├── adjust_symbol_thickness.py  # 符号粗细调整脚本
├── revert.py                   # 图像颜色处理脚本
├── sign_split.py               # 符号数据划分脚本
├── mnist_split.py              # MNIST数据划分脚本
├── main.py                     # 主程序入口
└── 使用说明.md                  # 本文档
```

## 5. 数据准备与预处理

### 5.1 数据集
MNIST数据集以及Kaggle上手写数学符号数据集https://www.kaggle.com/datasets/clarencezhao/handwritten-math-symbol-dataset?resource=download，https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols/code
### 5.2 MNIST数据划分
将MNIST数据集重新划分为训练集、验证集、测试集：

```bash
python mnist_split.py
```

该脚本会：
- 加载MNIST原始数据
- 按类别进行分层划分(70%训练，15%验证，15%测试)
- 保存到`data/mnist_split/`目录

### 5.3 符号数据划分
将数学符号数据集进行划分：

```bash
python sign_split.py
```

该脚本会：
- 处理`data/math_symbols/`中的原始符号数据
- 按比例划分(70%训练，15%验证，15%测试)
- 保存到`data/math_symbols_split/`目录

### 5.4 图像颜色标准化
统一图像格式为黑底白字：

```bash
python revert.py
```

该脚本会：
- 检测图像是否为白底黑字
- 自动翻转颜色为黑底白字
- 保存处理后的图像到`data/data_black_white/`目录
- 生成处理报告`README.txt`

### 5.5 数据增强
使用数据增强技术扩充训练数据：

```bash
python augment_symbols.py --dir ./data/math_symbols --factor 10 --no-scaling
```

参数说明：
- `--dir`: 符号数据集目录路径
- `--factor`: 每个原始图像生成的增强样本数量
- `--no-scaling`: 禁用尺寸缩放处理，只应用随机增强

数据增强包括：
- 随机旋转(-20°到20°)
- 随机缩放(0.8到1.2倍)
- 高斯噪声添加
- 随机模糊
- 弹性变换
### 5.6 符号像素粗细调整
- 对左右括号原始数据集进行符号像素加粗

## 6. 快速开始

### 6.1 数据准备流程
```bash
# 1. 划分MNIST数据
python mnist_split.py

# 2. 划分符号数据
python sign_split.py

# 3. 标准化图像格式
python revert.py

# 4. 数据增强(可选)
python augment_symbols.py --dir ./data/math_symbols --factor 5
```

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

## 8. 网络架构详解

### 8.1 模型结构
- **总参数量**: 214,735 (约21.5万参数)
- **网络深度**: 12层
- **输入尺寸**: N×1×28×28
- **输出尺寸**: N×15

### 8.2 详细架构
```
输入层 (N×1×28×28)
    ↓
Conv Block 1 (1→32通道) + MaxPool (28×28→14×14)
    ↓
Residual Block 1 (32通道)
    ↓
Residual Block 2 (32通道)
    ↓
Conv Block 2 (32→64通道) + MaxPool (14×14→7×7)
    ↓
Residual Block 3 (64通道)
    ↓
Residual Block 4 (64通道)
    ↓
Global Average Pooling (7×7→1×1)
    ↓
Classifier (64→128→15)
```

### 8.3 关键特性
- **残差连接**: 4个跳跃连接，解决梯度消失问题
- **批归一化**: 每个卷积层后都有BatchNorm
- **Dropout正则化**: 分类器中使用p=0.5和p=0.3的Dropout
- **全局平均池化**: 替代传统全连接层，减少过拟合

## 9. 训练与评估

### 9.1 训练配置
```python
# 推荐训练参数
epochs = 15
batch_size = 64
learning_rate = 0.001
optimizer = "AdamW"
scheduler = "ReduceLROnPlateau"
```

### 9.2 数据增强配置
```python
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### 9.3 性能指标
| 指标 | 目标值 | 备注 |
|:----:|:------:|:----:|
| 训练准确率 | >95% | 充分训练后 |
| 验证准确率 | >90% | 泛化能力 |
| 测试准确率 | >88% | 最终性能 |
| 推理速度 | <10ms | CPU单张图像 |
| 模型大小 | ~0.8MB | 压缩后 |

## 10. 数据集管理

### 10.1 数据集结构
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

### 10.2 数据质量控制
- **颜色统一**: 所有图像统一为黑底白字格式
- **尺寸标准**: 28×28像素，居中对齐
- **类别平衡**: 各类别样本数量尽量均衡
- **质量检查**: 自动过滤损坏或异常图像

## 11. 高级功能

### 11.1 符号分割优化
系统实现了智能的符号分割算法，特别针对除号(÷)等垂直结构符号进行了优化：

```python
def merge_contours(boxes, overlap_threshold=0.5):
    """
    智能合并边界框，专门处理垂直结构符号
    仅当两个框在水平方向上有显著重叠时，才将它们合并
    """
    # 实现细节见 src/data/preprocessing.py
```

### 11.2 多来源图像处理
系统支持不同来源的图像处理：

```python
def preprocess_image(image_input, source='canvas', normalize=False):
    """
    source参数：
    - 'file': 文件加载的图像，自动检测并翻转颜色
    - 'canvas': 画布手写的图像，直接处理
    """
```

### 11.3 公式计算支持
使用sympy库进行符号运算：
- 支持基本四则运算
- 支持带变量的表达式
- 自动LaTeX格式转换
### 11.4 括号与1的识别
- 归一化了符号方向，针对数字1和括号区分
- 高宽比大于3的可能是数字1
### 11.5 边缘信息增强
- 略微调整增大了符号检测时的识别框内填充

## 12. 故障排除

### 12.1 常见问题

**Q: 模型加载失败**
```
错误: 未找到模型文件。请先训练模型或检查模型路径
```
解决方案：
- 确认已运行训练命令：`python scripts/train.py`
- 检查`model/`目录下是否存在`.pth`模型文件
- 验证模型路径配置是否正确

**Q: 数据集路径错误**
```
错误: 数据集目录不存在
```
解决方案：
- 运行数据准备脚本：`python mnist_split.py` 和 `python sign_split.py`
- 确认`data/`目录结构正确
- 检查数据集是否已下载和处理

**Q: 数据增强失败**
```
错误: 无法读取图像文件
```
解决方案：
- 确认图像文件格式正确(.png, .jpg, .jpeg)
- 检查文件权限和路径
- 验证图像文件未损坏

**Q: CUDA相关错误**
```
错误: CUDA out of memory
```
解决方案：
- 减小batch_size参数：`--batch_size 32`
- 使用CPU训练：`export CUDA_VISIBLE_DEVICES=-1`
- 检查CUDA和PyTorch版本兼容性

### 12.2 调试模式
启用详细日志输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 13. 性能优化

### 13.1 训练优化
- 使用数据增强提高泛化能力
- 调整学习率调度策略
- 实验不同的优化器组合
- 使用预训练权重初始化

### 13.2 数据预处理优化
- 批量处理图像转换
- 缓存预处理结果
- 并行数据加载
- 内存映射文件读取

### 13.3 推理优化
- 模型量化：`torch.quantization`
- ONNX导出：支持跨平台部署
- 批量处理：同时处理多张图像
- 缓存优化：避免重复加载模型

## 14. 扩展开发

### 14.1 添加新符号
1. 准备新符号的训练数据
2. 修改`src/models/config.py`中的`SYMBOL_CLASSES`
3. 运行数据处理脚本统一格式
4. 重新训练模型

### 14.2 集成到其他项目
```python
from src.models.symbol_recognizer import SymbolRecognizer
from src.recognition.formula import recognize_formula

# 加载模型
model = SymbolRecognizer(num_classes=15)
model.load_state_dict(torch.load('model/best_symbol_model.pth'))

# 识别公式
formula, result, details = recognize_formula(model, symbols, device)
```

### 14.3 数据增强定制
可以修改`augment_symbols.py`中的增强参数：
```python
# 自定义增强参数
rotation_range = (-20, 20)      # 旋转角度范围
noise_std = 5                   # 噪声标准差
blur_radius = (0, 0.8)          # 模糊半径范围
```

## 15. 贡献指南

### 15.1 代码规范
- 使用Python PEP8编码规范
- 添加详细的文档字符串
- 提供单元测试
- 遵循项目的目录结构

### 15.2 提交流程
1. Fork项目仓库
2. 创建功能分支
3. 提交代码并添加测试
4. 创建Pull Request
5. 代码审查通过后合并

## 16. 许可证与致谢

### 16.1 许可证
本项目使用MIT许可证，详见LICENSE文件。

### 16.2 致谢
- PyTorch团队提供深度学习框架
- MNIST数据集提供基础数字数据
- 相关开源项目的启发和参考

## 17. 版本更新记录
### v2.2 (2025-07-19)
- 添加了对左右括号的识别
- 优化了ui界面
- 
### v2.1 (2025-07-16)
- 更新项目目录结构
- 移除generator.py依赖
- 添加数据增强功能
- 优化数据预处理流程
- 完善图像格式标准化

### v2.0 (2025-07-16)
- 完全重构项目架构
- 优化符号分割算法
- 添加智能图像预处理
- 改进GUI界面设计
- 增加模型验证和测试工具

### v1.0 (2025-07-15)
- 初始版本发布
- 基础CNN模型实现
- 简单GUI界面
- 支持基本符号识别

---

**最后更新**: 2025-07-19 16:17:19 北京时间  
**维护者**: Coolzs77  
**项目地址**: https://github.com/Coolzs77/formula2
