#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公式识别逻辑 - 处理符号识别结果，形成公式并计算
"""

import os
import sys
import torch
import numpy as np
import cv2
from torchvision import transforms

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.config import SYMBOL_CLASSES

try:
    import sympy
    from sympy import symbols, sympify
    from sympy.parsing.sympy_parser import parse_expr
except ImportError:
    print("警告: sympy库未安装，公式计算功能将受限")
    sympy = None

def recognize_symbols(model, symbols_list, device):
    """
    识别一组符号
    
    参数:
        model: 加载的模型
        symbols_list: 符号图像列表
        device: 计算设备(CPU/GPU)
        
    返回:
        识别结果列表，每个元素为(符号类别ID, 符号, 置信度)
    """
    model.eval()
    results = []
    
    # 预处理变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    for symbol in symbols_list:
        # 获取归一化图像
        normalized_image = symbol['normalized_image']
        
        # 应用变换
        image_tensor = transform(normalized_image).unsqueeze(0).to(device)
        
        # 前向传播
        with torch.no_grad():
            output = model(image_tensor)
            
            # 获取预测结果
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            class_id = predicted.item()
            conf = confidence.item()
            symbol_str = SYMBOL_CLASSES.get(class_id, "unknown")
            
            # 保存结果和位置信息
            results.append({
                'symbol': symbol_str,
                'position': symbol['position'],
                'class_id': class_id,
                'confidence': conf
            })
    
    return results

def formula_to_string(recognition_results):
    """
    将识别结果转换为公式字符串
    
    参数:
        recognition_results: 符号识别结果列表
        
    返回:
        公式字符串
    """
    formula = ""
    
    for result in recognition_results:
        symbol = result['symbol']
        # 特殊符号转换
        if symbol == 'pi':
            formula += 'π'
        elif symbol == 'sqrt':
            formula += '√'
        else:
            formula += symbol
            
    return formula

def formula_to_latex(recognition_results):
    """
    将识别结果转换为LaTeX字符串
    
    参数:
        recognition_results: 符号识别结果列表
        
    返回:
        LaTeX格式的公式
    """
    latex = ""
    
    previous_symbol = None
    skip_next = False
    
    for i, result in enumerate(recognition_results):
        if skip_next:
            skip_next = False
            continue
            
        symbol = result['symbol']
        
        # 特殊符号转换为LaTeX格式
        if symbol == 'pi':
            latex += '\\pi '
        elif symbol == 'sqrt':
            latex += '\\sqrt{'
            # 寻找下一个符号作为根号内的内容
            if i + 1 < len(recognition_results):
                next_symbol = recognition_results[i + 1]['symbol']
                latex += next_symbol + '}'
                skip_next = True
            else:
                latex += '}'
        elif symbol == '×':
            latex += '\\times '
        elif symbol == '÷':
            latex += '\\div '
        elif symbol == '^':
            latex += '^{'
            # 寻找下一个符号作为指数
            if i + 1 < len(recognition_results):
                next_symbol = recognition_results[i + 1]['symbol']
                latex += next_symbol + '}'
                skip_next = True
            else:
                latex += '}'
        else:
            latex += symbol + ' '
            
        previous_symbol = symbol
    
    return latex

def evaluate_formula(formula_string):
    """
    计算公式结果
    
    参数:
        formula_string: 公式字符串
        
    返回:
        计算结果或错误信息
    """
    if sympy is None:
        return "无法计算：缺少sympy库"
        
    try:
        # 替换常见符号为Python可解析的形式
        calc_string = formula_string.replace('×', '*')
        calc_string = calc_string.replace('÷', '/')
        calc_string = calc_string.replace('π', 'pi')
        calc_string = calc_string.replace('√', 'sqrt')
        
        # 检查是否需要符号计算(包含x或y变量)
        if 'x' in calc_string or 'y' in calc_string:
            x, y = sympy.symbols('x y')
            try:
                expr = parse_expr(calc_string)
                return expr
            except:
                return "无法解析表达式"
        else:
            # 直接计算数值结果
            try:
                result = eval(calc_string)
                return result
            except:
                try:
                    # 尝试使用sympy计算
                    expr = parse_expr(calc_string)
                    return expr
                except:
                    return "无法计算"
    except Exception as e:
        return f"计算错误: {str(e)}"

def recognize_formula(model, symbols_list, device):
    """
    识别公式并计算结果
    
    参数:
        model: 加载的模型
        symbols_list: 分割后的符号列表
        device: 计算设备
        
    返回:
        (formula_string, evaluation_result, recognition_results)
        formula_string: 公式字符串
        evaluation_result: 计算结果
        recognition_results: 原始识别结果
    """
    # 识别符号
    recognition_results = recognize_symbols(model, symbols_list, device)
    
    # 转换为公式字符串
    formula_string = formula_to_string(recognition_results)
    
    # 计算结果
    try:
        evaluation_result = evaluate_formula(formula_string)
    except Exception as e:
        evaluation_result = f"计算错误: {str(e)}"
        
    return formula_string, evaluation_result, recognition_results



def generate_visualization(base_image, recognition_results, offset=(0, 0), scale=1.0):
    """
    【最终修正版】在给定的基础图像上，根据偏移量和缩放比例绘制标注。

    参数:
        base_image:          一个已经包含居中内容的、与显示区域等大的背景图。
        recognition_results: 识别结果列表。
        offset:              内容在base_image上的偏移量 (x_offset, y_offset)。
        scale:               内容被缩放的比例。
    """
    vis_image = base_image.copy()
    offset_x, offset_y = offset

    for result in recognition_results:
        # 获取相对于原始、未缩放内容的坐标 (x, y, w, h)
        x, y, w, h = result['position']
        symbol = result.get('symbol', '?')
        conf = result.get('confidence', 0) * 100

        # 【核心修正】
        # 1. 先将原始坐标和尺寸按比例缩放
        # 2. 再加上偏移量，得到在最终大图上的绝对坐标
        final_x = int(x * scale) + offset_x
        final_y = int(y * scale) + offset_y
        final_w = int(w * scale)
        final_h = int(h * scale)

        # 使用计算出的最终坐标绘制绿色边界框
        cv2.rectangle(vis_image, (final_x, final_y), (final_x + final_w, final_y + final_h), (0, 255, 0), 2)

        # 使用最终坐标添加红色识别结果标签
        symbol_display_map = {'×': 'x', '÷': '/'}
        display_symbol = symbol_display_map.get(symbol, symbol)
        label = f"{display_symbol} ({conf:.1f}%)"

        label_y_pos = final_y - 10 if final_y > 20 else final_y + final_h + 20
        cv2.putText(vis_image, label, (final_x, label_y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return vis_image