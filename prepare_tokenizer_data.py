#!/usr/bin/env python3
"""
准备tokenizer训练数据
创建示例数据集用于训练支持图像占位符的tokenizer
"""

import json
import random
from pathlib import Path
from typing import List


def create_sample_dataset(output_path: str, num_samples: int = 10000):
    """
    创建示例训练数据集
    
    Args:
        output_path: 输出文件路径
        num_samples: 生成样本数量
    """
    
    # 基础文本模板
    text_templates = [
        # 中文模板
        "这是一段普通的文本。",
        "机器学习是人工智能的重要分支。",
        "深度学习模型在图像识别任务中表现出色。",
        "自然语言处理技术正在快速发展。",
        "计算机视觉可以帮助我们理解图像内容。",
        "多模态学习结合了文本和图像信息。",
        "Transformer架构革命性地改变了NLP领域。",
        "预训练模型为下游任务提供了强大的基础。",
        
        # 英文模板
        "This is a sample text for tokenizer training.",
        "Machine learning algorithms are powerful tools.",
        "Deep neural networks can learn complex patterns.",
        "Natural language processing enables human-computer interaction.",
        "Computer vision helps machines understand visual content.",
        "Multimodal learning combines different types of data.",
        "Large language models show impressive capabilities.",
        "Pre-trained models serve as foundation for many tasks.",
        
        # 技术相关
        "数据预处理是机器学习pipeline的重要步骤。",
        "特征工程可以显著提升模型性能。",
        "模型评估需要使用合适的指标。",
        "超参数调优是模型优化的关键环节。",
        "数据增强技术可以提高模型的泛化能力。",
        "正则化方法有助于防止过拟合。",
        "集成学习通过组合多个模型来提升效果。",
        "迁移学习利用预训练模型的知识。",
    ]
    
    # 图像相关模板
    image_templates = [
        # 中文图像描述
        "请看这张图片：{image}",
        "这张图片展示了：{image}",
        "图片内容如下：{image}",
        "如图所示：{image}",
        "根据图片 {image} 可以看出",
        "图像 {image} 显示了重要信息",
        "参考图片：{image}",
        "下面是相关图片：{image}",
        
        # 英文图像描述  
        "Here is the image: {image}",
        "Look at this picture: {image}",
        "The image shows: {image}",
        "Based on the image {image}",
        "Referring to image {image}",
        "This picture {image} demonstrates",
        "See the following image: {image}",
        "The visual content: {image}",
        
        # 对话式
        "用户：展示图片 {image}",
        "用户：请分析这张图 {image}",
        "助手：根据图片 {image}，我可以看到",
        "User: Show me the image {image}",
        "User: Analyze this picture {image}",
        "Assistant: Based on the image {image}, I can see",
        
        # 指令式
        "描述图片：{image}",
        "分析图像内容：{image}",
        "识别图片中的对象：{image}",
        "Describe the image: {image}",
        "Analyze the visual content: {image}",
        "Identify objects in {image}",
    ]
    
    # 多图像模板
    multi_image_templates = [
        "比较这两张图片：{image1} 和 {image2}",
        "第一张图 {image1}，第二张图 {image2}",
        "图片序列：{image1} {image2} {image3}",
        "Compare these images: {image1} and {image2}",
        "First image {image1}, second image {image2}",
        "Image sequence: {image1} {image2} {image3}",
    ]
    
    def generate_image_placeholder(length: int = None) -> str:
        """生成指定长度的图像占位符"""
        if length is None:
            # 常见的图像token长度
            lengths = [1, 64, 196, 256, 384, 576]
            length = random.choice(lengths)
        return "@" * length
    
    samples = []
    
    # 生成纯文本样本 (30%)
    for _ in range(int(num_samples * 0.3)):
        text = random.choice(text_templates)
        samples.append(text)
    
    # 生成单图像样本 (50%)
    for _ in range(int(num_samples * 0.5)):
        template = random.choice(image_templates)
        image_placeholder = generate_image_placeholder()
        text = template.format(image=image_placeholder)
        samples.append(text)
    
    # 生成多图像样本 (20%)
    for _ in range(int(num_samples * 0.2)):
        template = random.choice(multi_image_templates)
        placeholders = {
            'image1': generate_image_placeholder(),
            'image2': generate_image_placeholder(), 
            'image3': generate_image_placeholder(),
        }
        text = template.format(**placeholders)
        samples.append(text)
    
    # 随机打乱
    random.shuffle(samples)
    
    # 保存为文本文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(sample + '\n')
    
    print(f"✅ 已生成 {len(samples)} 条训练样本到 {output_path}")
    
    # 生成统计信息
    stats = {
        'total_samples': len(samples),
        'avg_length': sum(len(s) for s in samples) / len(samples),
        'samples_with_images': sum(1 for s in samples if '@' in s),
        'total_at_symbols': sum(s.count('@') for s in samples),
    }
    
    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"📊 数据统计已保存到 {stats_path}")
    print(f"   - 总样本数: {stats['total_samples']}")
    print(f"   - 平均长度: {stats['avg_length']:.1f} 字符")
    print(f"   - 包含图像的样本: {stats['samples_with_images']}")
    print(f"   - 总@符号数: {stats['total_at_symbols']}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成tokenizer训练数据")
    parser.add_argument("--output", default="data/tokenizer_train.txt",
                       help="输出文件路径 (默认: data/tokenizer_train.txt)")
    parser.add_argument("--num-samples", type=int, default=10000,
                       help="生成样本数量 (默认: 10000)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("准备Tokenizer训练数据")
    print("="*60)
    
    create_sample_dataset(args.output, args.num_samples)
    
    print(f"\n使用方法:")
    print(f"python train_tokenizer.py --data {args.output} --output tokenizers/my_tokenizer")


if __name__ == "__main__":
    main()