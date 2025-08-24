#!/usr/bin/env python3
"""
创建测试数据集用于验证训练和监控链路
生成简单的文本数据用于LLM训练测试
"""

import os
import json
import random
from pathlib import Path

def generate_simple_text_data():
    """生成简单的文本数据用于测试"""
    
    # 简单的文本样本
    templates = [
        "今天天气很好，适合出门散步。",
        "机器学习是人工智能的一个重要分支。", 
        "深度学习模型需要大量的训练数据。",
        "SwanLab是一个优秀的实验监控平台。",
        "Python是数据科学领域的热门语言。",
        "自然语言处理技术发展迅速。",
        "大语言模型改变了AI的发展方向。",
        "训练神经网络需要仔细调整超参数。",
        "开源项目促进了AI技术的普及。",
        "实验监控对于机器学习项目很重要。"
    ]
    
    # 生成训练数据
    train_data = []
    for i in range(100):  # 生成100个训练样本
        text = random.choice(templates)
        # 简单的输入-输出对
        train_data.append({
            "input": f"请继续这句话：{text[:10]}",  # 取前10个字符作为输入
            "output": text,  # 完整句子作为输出
            "id": i
        })
    
    return train_data

def save_data(data, output_dir):
    """保存数据到指定目录"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON格式
    train_file = output_path / "train.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 测试数据已保存到: {train_file}")
    print(f"✓ 生成了 {len(data)} 个训练样本")
    
    return str(train_file)

if __name__ == "__main__":
    # 生成测试数据
    print("正在生成测试数据...")
    test_data = generate_simple_text_data()
    
    # 保存数据
    data_file = save_data(test_data, "data/test")
    
    # 显示数据样本
    print(f"\n数据样本预览:")
    for i, sample in enumerate(test_data[:3]):
        print(f"样本 {i+1}:")
        print(f"  输入: {sample['input']}")  
        print(f"  输出: {sample['output']}")
        print()