#!/usr/bin/env python3
"""
自定义Tokenizer训练脚本 - 支持图像占位符"@"
使用Hugging Face tokenizers库训练BPE tokenizer
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Iterator
from tqdm import tqdm

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, normalizers
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


def load_text_data(data_paths: List[str], max_files: int = None) -> Iterator[str]:
    """
    加载训练文本数据
    
    Args:
        data_paths: 数据文件或目录路径列表
        max_files: 最大处理文件数量限制
        
    Yields:
        str: 文本行
    """
    file_count = 0
    
    for data_path in data_paths:
        path = Path(data_path)
        
        if path.is_file():
            files = [path]
        elif path.is_dir():
            # 支持多种文本格式
            files = []
            for ext in ['*.txt', '*.json', '*.jsonl']:
                files.extend(path.glob(f"**/{ext}"))
        else:
            print(f"警告: 路径 {data_path} 不存在，跳过")
            continue
            
        for file_path in files[:max_files] if max_files else files:
            print(f"处理文件: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix == '.jsonl':
                        # JSONL格式，每行一个JSON对象
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                # 假设文本字段为'text'
                                if 'text' in data:
                                    yield data['text']
                            except json.JSONDecodeError:
                                continue
                    elif file_path.suffix == '.json':
                        # JSON格式，可能是数组或对象
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'text' in item:
                                    yield item['text']
                                elif isinstance(item, str):
                                    yield item
                        elif isinstance(data, dict) and 'text' in data:
                            yield data['text']
                    else:
                        # 纯文本格式
                        for line in f:
                            line = line.strip()
                            if line:  # 跳过空行
                                yield line
                                
                file_count += 1
                if max_files and file_count >= max_files:
                    break
                    
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
                continue


def create_image_placeholder_text(num_images: int = 1000) -> List[str]:
    """
    创建包含图像占位符的示例文本，帮助tokenizer学习"@"符号的使用
    
    Args:
        num_images: 生成的图像样本数量
        
    Returns:
        List[str]: 包含图像占位符的文本示例
    """
    templates = [
        "这是一张图片：{image_token}",
        "请看这张图：{image_token}",
        "图片内容：{image_token}",
        "如图所示：{image_token}",
        "{image_token} 这张图片展示了...",
        "根据图片 {image_token} 可以看出...",
        "参考图像：{image_token}",
        "{image_token}（图片说明）",
        "Here is an image: {image_token}",
        "Look at this picture: {image_token}",
        "The image {image_token} shows...",
        "Based on {image_token}, we can see...",
    ]
    
    image_examples = []
    
    for i in range(num_images):
        # 生成不同长度的@占位符（模拟不同图像token长度）
        lengths = [1, 196, 256, 384]  # 常见的图像patch数量
        length = lengths[i % len(lengths)]
        image_token = "@" * length
        
        template = templates[i % len(templates)]
        text = template.format(image_token=image_token)
        image_examples.append(text)
    
    return image_examples


def train_tokenizer(
    data_paths: List[str],
    output_dir: str,
    vocab_size: int = 30000,
    min_frequency: int = 2,
    max_files: int = None,
    include_image_examples: bool = True
):
    """
    训练支持图像占位符的BPE tokenizer
    
    Args:
        data_paths: 训练数据路径列表
        output_dir: 输出目录
        vocab_size: 词汇表大小
        min_frequency: token最小频率
        max_files: 最大处理文件数量
        include_image_examples: 是否包含图像占位符示例
    """
    
    print("="*60)
    print("开始训练自定义Tokenizer")
    print("="*60)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 定义特殊token
    special_tokens = [
        "<unk>",    # 未知token
        "<bos>",    # 序列开始
        "<eos>",    # 序列结束
        "<pad>",    # 填充token
        "@",        # 图像占位符基础token
        "<image>",  # 备用图像token
    ]
    
    print(f"特殊token: {special_tokens}")
    print(f"目标词汇表大小: {vocab_size}")
    print(f"最小频率: {min_frequency}")
    
    # 创建BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # 设置预处理器 - 处理空白和标点
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),           # 按空白分割
        pre_tokenizers.Punctuation(behavior="isolated"),  # 隔离标点符号
    ])
    
    # 设置标准化器 - 处理文本规范化
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),        # Unicode标准化
        normalizers.Lowercase(),  # 转小写（可选）
        normalizers.StripAccents()  # 去除重音符号
    ])
    
    # 配置BPE训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        continuing_subword_prefix="##",  # 子词前缀
        end_of_word_suffix="</w>",       # 词结束后缀
    )
    
    # 准备训练数据
    print("\n准备训练数据...")
    
    # 收集所有文本数据
    all_texts = []
    
    # 加载用户数据
    print("加载训练文本...")
    for text in tqdm(load_text_data(data_paths, max_files), desc="加载文本"):
        all_texts.append(text)
    
    print(f"加载了 {len(all_texts)} 条文本数据")
    
    # 添加图像占位符示例
    if include_image_examples:
        print("生成图像占位符示例...")
        image_examples = create_image_placeholder_text(1000)
        all_texts.extend(image_examples)
        print(f"添加了 {len(image_examples)} 条图像示例")
    
    print(f"总计 {len(all_texts)} 条训练数据")
    
    # 开始训练
    print("\n开始训练tokenizer...")
    tokenizer.train_from_iterator(all_texts, trainer=trainer)
    
    # 设置后处理器
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    
    # 保存tokenizer
    tokenizer_path = output_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer已保存到: {tokenizer_path}")
    
    # 创建HuggingFace兼容的tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    
    # 保存HuggingFace格式
    hf_path = output_path / "hf_tokenizer"
    hf_tokenizer.save_pretrained(str(hf_path))
    print(f"HuggingFace tokenizer已保存到: {hf_path}")
    
    # 测试tokenizer
    print("\n="*60)
    print("测试tokenizer")
    print("="*60)
    
    test_texts = [
        "这是一个测试句子。",
        "这是一张图片：@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",  # 196个@
        "Hello world! This is an image: @@@@@@@@@@@@@@",
        "请看这张图：@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 这张图很有趣。",
    ]
    
    for text in test_texts:
        tokens = hf_tokenizer.tokenize(text)
        ids = hf_tokenizer.encode(text)
        decoded = hf_tokenizer.decode(ids)
        
        print(f"\n原文: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"Token数量: {len(tokens)}")
        print(f"Token IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")
        print(f"解码: {decoded[:50]}{'...' if len(decoded) > 50 else ''}")
        
        # 特别检查@符号的处理
        if "@" in text:
            at_count = text.count("@")
            print(f"@ 符号数量: {at_count}")
            at_tokens = [token for token in tokens if "@" in token]
            print(f"包含@的tokens: {at_tokens[:10]}{'...' if len(at_tokens) > 10 else ''}")
    
    # 保存配置信息
    config = {
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "training_data_files": len(all_texts),
        "tokenizer_path": str(tokenizer_path),
        "hf_tokenizer_path": str(hf_path),
    }
    
    config_path = output_path / "training_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n训练配置已保存到: {config_path}")
    print("\n✅ Tokenizer训练完成！")
    
    return hf_tokenizer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="训练支持图像占位符的自定义tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 从文本文件训练:
  python train_tokenizer.py --data corpus.txt --output tokenizers/my_tokenizer
  
  # 从多个目录训练:
  python train_tokenizer.py --data data/train data/valid --output tokenizers/my_tokenizer
  
  # 自定义参数:
  python train_tokenizer.py --data corpus.txt --output tokenizers/my_tokenizer --vocab-size 50000 --min-freq 3
        """
    )
    
    parser.add_argument("--data", nargs="+", required=True,
                       help="训练数据文件或目录路径（支持.txt, .json, .jsonl）")
    parser.add_argument("--output", required=True,
                       help="输出目录路径")
    parser.add_argument("--vocab-size", type=int, default=30000,
                       help="词汇表大小 (默认: 30000)")
    parser.add_argument("--min-freq", type=int, default=2,
                       help="token最小频率 (默认: 2)")
    parser.add_argument("--max-files", type=int, default=None,
                       help="最大处理文件数量限制 (默认: 无限制)")
    parser.add_argument("--no-image-examples", action="store_true",
                       help="不包含图像占位符示例")
    
    args = parser.parse_args()
    
    # 训练tokenizer
    train_tokenizer(
        data_paths=args.data,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        max_files=args.max_files,
        include_image_examples=not args.no_image_examples
    )


if __name__ == "__main__":
    main()