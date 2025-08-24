"""
VLM训练数据集 - 支持JSONL格式的多模态对话数据
适配MiniMind-V训练器架构
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer, GPT2Tokenizer


class VLMDataset(Dataset):
    """多模态对话数据集类"""
    
    def __init__(
        self, 
        data_path: str, 
        images_path: str, 
        tokenizer,
        preprocess=None,
        image_special_token="<image>",
        max_length: int = 640
    ):
        self.data_path = Path(data_path)
        self.images_path = Path(images_path)
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.image_special_token = image_special_token
        self.max_length = max_length
        
        # 默认图像预处理
        if self.preprocess is None:
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # 加载数据
        self.samples = self._load_jsonl_data()
        print(f"✅ 加载了 {len(self.samples)} 个训练样本")
    
    def _load_jsonl_data(self) -> List[Dict]:
        """加载JSONL格式的数据"""
        samples = []
        
        if not self.data_path.exists():
            print(f"❌ 数据文件不存在: {self.data_path}")
            return samples
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    sample = json.loads(line)
                    
                    # 验证数据格式
                    if not self._validate_sample(sample):
                        print(f"跳过第 {line_num} 行: 数据格式不符合要求")
                        continue
                    
                    samples.append(sample)
                    
                except json.JSONDecodeError as e:
                    print(f"第 {line_num} 行JSON解析错误: {e}")
                    continue
                except Exception as e:
                    print(f"第 {line_num} 行处理错误: {e}")
                    continue
        
        return samples
    
    def _validate_sample(self, sample: Dict) -> bool:
        """验证样本格式"""
        required_fields = ["conversations", "image"]
        if not all(field in sample for field in required_fields):
            return False
        
        conversations = sample["conversations"]
        if not isinstance(conversations, list) or len(conversations) == 0:
            return False
            
        for conv in conversations:
            if not isinstance(conv, dict):
                return False
            if not all(key in conv for key in ["role", "content"]):
                return False
            if conv["role"] not in ["user", "assistant"]:
                return False
                
        return True
    
    def _process_conversations(self, conversations: List[Dict]) -> str:
        """将对话转换为文本"""
        text = ""
        
        for conv in conversations:
            role = conv["role"]
            content = conv["content"]
            
            # 处理图像标记
            if "<image>" in content:
                content = content.replace("<image>", self.image_special_token)
            
            # 添加角色标记
            if role == "user":
                text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}\n"
        
        text += "<|endoftext|>"
        return text
    
    def _load_image(self, image_name: str) -> torch.Tensor:
        """加载和预处理图像"""
        image_path = self.images_path / image_name
        
        if not image_path.exists():
            print(f"警告: 图像文件不存在 {image_path}")
            return torch.zeros(3, 224, 224)
        
        try:
            image = Image.open(image_path).convert('RGB')
            return self.preprocess(image)
        except Exception as e:
            print(f"图像加载错误 {image_path}: {e}")
            return torch.zeros(3, 224, 224)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回格式: (input_ids, labels, loss_mask, pixel_values)
        """
        sample = self.samples[idx]
        
        # 处理文本
        text = self._process_conversations(sample["conversations"])
        
        # Tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # 创建labels (用于语言建模)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # 向左移动一位
        labels[-1] = -100  # 最后一个token不参与损失计算
        
        # 创建loss_mask
        loss_mask = attention_mask.clone().float()
        
        # 加载图像
        pixel_values = self._load_image(sample["image"])
        
        return input_ids, labels, loss_mask, pixel_values


def create_vlm_dataset(
    data_path: str, 
    images_path: str, 
    tokenizer_path: str = None,
    tokenizer = None,
    max_length: int = 640,
    image_special_token: str = "<image>"
) -> VLMDataset:
    """
    创建VLM数据集
    
    Args:
        data_path: JSONL数据文件路径
        images_path: 图像文件夹路径
        tokenizer_path: tokenizer路径 (可选)
        tokenizer: tokenizer对象 (可选，优先使用)
        max_length: 最大序列长度
        image_special_token: 图像特殊token
        
    Returns:
        VLMDataset实例
    """
    # 加载或使用提供的tokenizer
    if tokenizer is None:
        if tokenizer_path:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
                print(f"✅ 加载自定义tokenizer: {tokenizer_path}")
            except Exception as e:
                print(f"❌ 加载自定义tokenizer失败: {e}")
                tokenizer = None
        
        if tokenizer is None:
            print("🔄 使用GPT2 tokenizer作为备选")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            special_tokens = {
                "pad_token": "<pad>",
                "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|endoftext|>", image_special_token]
            }
            tokenizer.add_special_tokens(special_tokens)
    
    # 创建数据集
    dataset = VLMDataset(
        data_path=data_path,
        images_path=images_path,
        tokenizer=tokenizer,
        max_length=max_length,
        image_special_token=image_special_token
    )
    
    return dataset