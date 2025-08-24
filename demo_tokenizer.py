#!/usr/bin/env python3
"""
自定义Tokenizer使用示例
演示如何加载和使用训练好的tokenizer处理图像占位符
"""

from pathlib import Path
from transformers import PreTrainedTokenizerFast


def load_custom_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """
    加载自定义训练的tokenizer
    
    Args:
        tokenizer_path: tokenizer路径
        
    Returns:
        PreTrainedTokenizerFast: 加载的tokenizer
    """
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def demo_tokenizer_usage(tokenizer_path: str):
    """
    演示tokenizer的使用方法
    
    Args:
        tokenizer_path: tokenizer路径
    """
    
    print("="*60)
    print("自定义Tokenizer使用演示")
    print("="*60)
    
    # 加载tokenizer
    try:
        tokenizer = load_custom_tokenizer(tokenizer_path)
        print(f"✅ 成功加载tokenizer: {tokenizer_path}")
        print(f"词汇表大小: {tokenizer.vocab_size}")
        print(f"特殊token: {tokenizer.special_tokens_map}")
    except Exception as e:
        print(f"❌ 加载tokenizer失败: {e}")
        return
    
    # 测试文本样例
    test_cases = [
        # 纯文本
        "你好，世界！这是一个测试句子。",
        "Hello, world! This is a test sentence.",
        
        # 包含单个图像占位符
        "这是一张图片：@",
        "Here is an image: @",
        
        # 包含196个@的图像占位符（CLIP ViT patch数量）
        "请看这张图：" + "@" * 196,
        "Look at this picture: " + "@" * 196,
        
        # 包含多个图像的复杂文本
        f"第一张图：{'@' * 196} 第二张图：{'@' * 196} 请比较这两张图片的差异。",
        f"Image 1: {'@' * 196} Image 2: {'@' * 196} Compare these two images.",
        
        # 对话式文本
        f"用户：展示一下这张图 {'@' * 196}\n助手：这张图片显示了...",
        
        # 指令式文本
        f"<bos>描述图片内容：{'@' * 196}<eos>",
    ]
    
    print(f"\n{'='*60}")
    print("Token化测试")
    print(f"{'='*60}")
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n--- 测试案例 {i} ---")
        
        # 显示原文（截断长文本）
        display_text = text[:100] + "..." if len(text) > 100 else text
        print(f"原文: {display_text}")
        
        # Token化
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        # 带特殊token的编码
        encoded_with_special = tokenizer.encode(text, add_special_tokens=True)
        
        # 解码验证
        decoded = tokenizer.decode(token_ids)
        decoded_with_special = tokenizer.decode(encoded_with_special)
        
        print(f"Token数量: {len(tokens)}")
        print(f"Token IDs: {token_ids[:15]}{'...' if len(token_ids) > 15 else ''}")
        
        # 显示部分tokens（特别关注@相关的tokens）
        if len(tokens) <= 20:
            print(f"Tokens: {tokens}")
        else:
            # 显示前几个和后几个，以及@相关的tokens
            at_tokens = [i for i, token in enumerate(tokens) if "@" in token]
            print(f"前10个tokens: {tokens[:10]}")
            if at_tokens:
                print(f"包含@的tokens位置: {at_tokens[:5]}{'...' if len(at_tokens) > 5 else ''}")
                print(f"@相关tokens示例: {[tokens[i] for i in at_tokens[:5]]}")
            print(f"后5个tokens: {tokens[-5:]}")
        
        # 检查解码是否正确
        is_correct = (decoded.strip() == text.strip())
        print(f"解码正确性: {'✅' if is_correct else '❌'}")
        
        if not is_correct:
            print(f"原文长度: {len(text)}")
            print(f"解码长度: {len(decoded)}")
            print(f"解码结果: {decoded[:100]}{'...' if len(decoded) > 100 else ''}")
        
        # 统计@符号处理情况
        if "@" in text:
            original_at_count = text.count("@")
            decoded_at_count = decoded.count("@")
            print(f"@符号: 原文{original_at_count}个 → 解码{decoded_at_count}个")
    
    print(f"\n{'='*60}")
    print("图像token ID分析")
    print(f"{'='*60}")
    
    # 分析@符号的token化方式
    at_variations = ["@", "@@", "@" * 5, "@" * 50, "@" * 196]
    
    for at_seq in at_variations:
        tokens = tokenizer.tokenize(at_seq)
        token_ids = tokenizer.encode(at_seq, add_special_tokens=False)
        
        print(f"\n'{at_seq[:20]}{'...' if len(at_seq) > 20 else ''}' ({len(at_seq)}个@):")
        print(f"  Token化为: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Token数量: {len(tokens)}")
    
    # 检查特殊token的ID
    print(f"\n{'='*60}")
    print("特殊Token ID映射")
    print(f"{'='*60}")
    
    special_tokens = ["<unk>", "<bos>", "<eos>", "<pad>", "@", "<image>"]
    for token in special_tokens:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"{token:>10}: {token_id}")
        except:
            print(f"{token:>10}: 未找到")
    
    print(f"\n✅ 演示完成！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自定义Tokenizer使用演示")
    parser.add_argument("--tokenizer", required=True,
                       help="训练好的tokenizer路径")
    
    args = parser.parse_args()
    
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer路径不存在: {tokenizer_path}")
        print("请先运行 train_tokenizer.py 训练tokenizer")
        return
    
    demo_tokenizer_usage(str(tokenizer_path))


if __name__ == "__main__":
    main()