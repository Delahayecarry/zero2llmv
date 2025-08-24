#!/usr/bin/env python3
"""
Zero2LLMV 训练脚本
支持SwanLab监控的多模态大语言模型训练
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# YAML configuration imports
from pydantic import ValidationError

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

import swanlab
from tqdm import tqdm

# 项目模块导入
from models import llmconfig, CausalLM, VLM, VLLMconfig
from configs import (
    load_yaml_config,
    merge_config_with_args, validate_config, 
    save_merged_config
)


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 模型配置
    model_type: str = "vlm"  # 模型类型: "llm" or "vlm" 
    model_config_path: str = ""  # 模型配置文件路径
    
    # 数据配置
    data_path: str = "data/processed"  # 训练数据路径
    max_seq_length: int = 512  # 最大序列长度
    batch_size: int = 8  # 批次大小
    num_workers: int = 4  # 数据加载器线程数
    
    # 训练配置
    num_epochs: int = 3  # 训练轮数
    learning_rate: float = 2e-5  # 学习率
    weight_decay: float = 0.01  # 权重衰减
    warmup_steps: int = 500  # 预热步数
    gradient_accumulation_steps: int = 4  # 梯度累积步数
    max_grad_norm: float = 1.0  # 梯度裁剪阈值
    
    # 混合精度训练
    use_amp: bool = True  # 是否使用自动混合精度
    
    # 保存与日志
    output_dir: str = "outputs"  # 输出目录
    save_steps: int = 1000  # 保存检查点的步数间隔
    logging_steps: int = 100  # 日志记录的步数间隔
    eval_steps: int = 500  # 评估的步数间隔
    
    # SwanLab配置
    swanlab_project: str = "VLLM"  # swanlab项目名
    swanlab_workspace: str = "delahayecarry"  # swanlab工作空间
    swanlab_experiment_name: str = ""  # 实验名称
    swanlab_description: str = ""  # 实验描述
    swanlab_logdir: str = ""  # 日志目录
    
    # SwanLab API配置
    swanlab_api_key: str = ""  # API密钥


class SimpleDataset(Dataset):
    """简单的训练数据集"""
    
    def __init__(self, data_path: str, config: TrainingConfig):
        self.data_path = Path(data_path)
        self.config = config
        self.samples = self.load_samples()
        
    def load_samples(self) -> List[Dict]:
        """加载训练样本"""
        samples = []
        
        # 这里应该根据你的实际数据格式来实现
        # 示例：假设有一个samples.json文件
        samples_file = self.data_path / "samples.json"
        if samples_file.exists():
            with open(samples_file, 'r', encoding='utf-8') as f:
                samples = json.load(f)
        else:
            # 创建一些示例数据
            print(f"未找到 {samples_file}，创建示例数据...")
            samples = [
                {
                    "text": "这是一个示例文本",
                    "image_path": None,
                    "label": "示例标签"
                } for _ in range(100)
            ]
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 这里应该实现实际的数据处理逻辑
        # 包括文本tokenization、图像处理等
        return {
            "text": sample.get("text", ""),
            "input_ids": torch.randint(0, 30000, (self.config.max_seq_length,)),  # 示例
            "attention_mask": torch.ones(self.config.max_seq_length),
            "labels": torch.randint(0, 30000, (self.config.max_seq_length,)),
            "pixel_values": torch.randn(1, 3, 224, 224) if sample.get("image_path") else None
        }


def collate_fn(batch):
    """数据批处理函数"""
    batch_input_ids = torch.stack([item["input_ids"] for item in batch])
    batch_attention_mask = torch.stack([item["attention_mask"] for item in batch])
    batch_labels = torch.stack([item["labels"] for item in batch])
    
    # 处理图像数据
    pixel_values = None
    if any(item["pixel_values"] is not None for item in batch):
        pixel_values = []
        for item in batch:
            if item["pixel_values"] is not None:
                pixel_values.append(item["pixel_values"])
            else:
                pixel_values.append(torch.zeros(1, 3, 224, 224))
        pixel_values = torch.stack(pixel_values)
    
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
        "pixel_values": pixel_values
    }


class Trainer:
    """训练器类"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        self.model = self.create_model()
        self.model.to(self.device)
        
        # 初始化数据
        self.train_dataset = SimpleDataset(config.data_path, config)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn
        )
        
        # 初始化优化器
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # 混合精度训练
        self.scaler = GradScaler() if config.use_amp else None
        
        # 初始化SwanLab监控
        self.use_swanlab = True  # 默认尝试使用SwanLab
        self.setup_swanlab()
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def create_model(self):
        """创建模型"""
        if self.config.model_type == "vlm":
            # 创建VLM配置
            model_config = VLLMconfig(
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=12,
                vocab_size=30000,
                max_positions_embeddings=self.config.max_seq_length,
                dropout=0.1
            )
            model = VLM(model_config)
        else:
            # 创建纯LLM配置
            model_config = llmconfig(
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=12,
                vocab_size=30000,
                max_positions_embeddings=self.config.max_seq_length,
                dropout=0.1
            )
            model = CausalLM(model_config)
            
        return model
    
    def create_optimizer(self):
        """创建优化器"""
        # 参数分组：通常对不同类型的参数使用不同的权重衰减
        param_groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
                "weight_decay": self.config.weight_decay
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
                "weight_decay": 0.0
            }
        ]
        
        return optim.AdamW(param_groups, lr=self.config.learning_rate)
    
    def create_scheduler(self):
        """创建学习率调度器"""
        total_steps = len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        
        return optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
    
    def setup_swanlab(self):
        """设置swanlab监控"""
        swanlab_config = {
            "model_type": self.config.model_type,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "num_epochs": self.config.num_epochs,
            "max_seq_length": self.config.max_seq_length,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "weight_decay": self.config.weight_decay,
            "warmup_steps": self.config.warmup_steps,
            "use_amp": self.config.use_amp,
        }
        
        # 配置swanlab环境
        if self.config.swanlab_api_key:
            os.environ["SWANLAB_API_KEY"] = self.config.swanlab_api_key
            
        # 检查是否有API key
        api_key = os.environ.get("SWANLAB_API_KEY") or self.config.swanlab_api_key
        
        if not api_key:
            print("⚠️  未设置SWANLAB_API_KEY，跳过SwanLab监控")
            print("   如需启用监控，请设置: export SWANLAB_API_KEY='your-key'")
            self.use_swanlab = False
            return
            
        try:
            # 初始化swanlab
            swanlab.init(
                project=self.config.swanlab_project,
                workspace=self.config.swanlab_workspace,
                experiment_name=self.config.swanlab_experiment_name or f"{self.config.model_type}-{self.config.learning_rate}",
                description=self.config.swanlab_description,
                logdir=self.config.swanlab_logdir or None,
                config=swanlab_config
            )
            print("✓ SwanLab监控已启用")
            self.use_swanlab = True
        except Exception as e:
            print(f"⚠️  SwanLab初始化失败: {e}")
            print("   继续训练但不进行监控...")
            self.use_swanlab = False
    
    def compute_loss(self, batch, model_output):
        """计算损失"""
        logits = model_output.logits
        labels = batch["labels"].to(self.device)
        
        # 只对非padding位置计算损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算交叉熵损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 添加MoE辅助损失
        if hasattr(model_output, 'aux_loss') and model_output.aux_loss is not None:
            loss = loss + model_output.aux_loss
            
        return loss
    
    def train_step(self, batch):
        """单步训练"""
        self.model.train()
        
        # 数据移到设备
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device) if batch["pixel_values"] is not None else None
        
        # 前向传播
        if self.config.use_amp:
            with autocast():
                if self.config.model_type == "vlm":
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values
                    )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                loss = self.compute_loss(batch, outputs)
        else:
            if self.config.model_type == "vlm":
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            loss = self.compute_loss(batch, outputs)
        
        # 梯度累积
        loss = loss / self.config.gradient_accumulation_steps
        
        # 反向传播
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        return loss.item()
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        epoch_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            loss = self.train_step(batch)
            epoch_loss += loss
            
            # 梯度累积和优化
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_amp:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # 优化器步进
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 梯度裁剪
                    clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    # 优化器步进
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1

                current_lr = self.scheduler.get_lr()[0]
                # 日志记录
                if self.global_step % self.config.logging_steps == 0:
                    pass
                if self.use_swanlab:
                    swanlab.log({
                        "train/loss": loss * self.config.gradient_accumulation_steps,
                        "train/learning_rate": current_lr,
                        "train/global_step": self.global_step,
                        "train/epoch": epoch + (step + 1) / num_batches
                    })
                
                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
            
            # 更新进度条
            avg_loss = epoch_loss / (step + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'step': self.global_step
            })
        
        return epoch_loss / num_batches
    
    def save_checkpoint(self, checkpoint_name: str):
        """保存检查点"""
        checkpoint_dir = Path(self.config.output_dir) / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }, checkpoint_dir / "pytorch_model.bin")
        
        # 保存配置
        with open(checkpoint_dir / "training_config.json", 'w', encoding='utf-8') as f:
            json.dump(vars(self.config), f, indent=2, ensure_ascii=False)
            
        print(f"检查点已保存到: {checkpoint_dir}")
        
        # 记录到swanlab
        if self.use_swanlab:
            swanlab.log({"checkpoint/saved": self.global_step})
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"从检查点恢复训练: epoch {self.epoch}, step {self.global_step}")
    
    def train(self):
        """主训练循环"""
        print("开始训练...")
        print(f"训练数据集大小: {len(self.train_dataset)}")
        print(f"批次大小: {self.config.batch_size}")
        print(f"总epoch数: {self.config.num_epochs}")
        print(f"梯度累积步数: {self.config.gradient_accumulation_steps}")
        print(f"预计总步数: {len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps}")
        
        try:
            for epoch in range(self.config.num_epochs):
                self.epoch = epoch
                
                print(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")
                
                epoch_loss = self.train_epoch(epoch)
                
                print(f"Epoch {epoch + 1} 平均损失: {epoch_loss:.4f}")
                
                # 记录epoch级别的指标
                if self.use_swanlab:
                    swanlab.log({
                        "epoch/loss": epoch_loss,
                        "epoch/epoch": epoch + 1
                    })
                
                # 保存epoch检查点
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint("best_model")
                    print(f"最佳模型已更新 (损失: {self.best_loss:.4f})")
                
                # 每个epoch结束保存检查点
                self.save_checkpoint(f"epoch-{epoch+1}")
                
        except KeyboardInterrupt:
            print("\n训练被用户中断")
        except Exception as e:
            print(f"\n训练过程中发生错误: {e}")
            raise
        finally:
            # 保存最终检查点
            self.save_checkpoint("final_model")
            print("训练完成")
            
            # 关闭swanlab
            if self.use_swanlab:
                swanlab.finish()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Zero2LLMV 训练脚本 - 支持YAML配置和CLI参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用YAML配置文件:
  python train.py --config configs/config.yaml
  
  # 使用YAML配置并覆盖特定参数:
  python train.py --config configs/config.yaml --learning_rate 5e-5 --batch_size 16
  
  # 仅使用CLI参数（向后兼容）:
  python train.py --model_type vlm --batch_size 8 --learning_rate 2e-5
  
  # 使用实验配置:
  python train.py --config configs/experiments/high_lr_experiment.yaml
        """
    )
    
    # YAML配置相关参数（新增）
    config_group = parser.add_argument_group('YAML Configuration', '配置文件相关参数')
    config_group.add_argument("--config", type=str, default="",
                       help="YAML配置文件路径 (如: configs/config.yaml)")
    config_group.add_argument("--allow-default-fallback", action="store_true",
                       help="验证失败时使用默认配置（用于调试）")
    config_group.add_argument("--save-merged-config", action="store_true",
                       help="保存最终合并的配置到输出目录")
    
    # 模型配置
    model_group = parser.add_argument_group('Model Configuration', '模型相关配置')
    model_group.add_argument("--model_type", type=str, default="vlm", choices=["llm", "vlm"],
                       help="模型类型: llm (仅语言模型) 或 vlm (视觉语言模型)")
    model_group.add_argument("--model_config_path", type=str, default="",
                       help="模型配置文件路径（可选）")
    
    # 数据配置
    data_group = parser.add_argument_group('Data Configuration', '数据相关配置')
    data_group.add_argument("--data_path", type=str, default="data/processed",
                       help="训练数据路径")
    data_group.add_argument("--max_seq_length", type=int, default=512,
                       help="最大序列长度 (1-8192)")
    data_group.add_argument("--batch_size", type=int, default=8,
                       help="每GPU批次大小 (1-1024)")
    data_group.add_argument("--num_workers", type=int, default=4,
                       help="数据加载器线程数 (0-64)")
    
    # 训练配置
    training_group = parser.add_argument_group('Training Configuration', '训练相关配置')
    training_group.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数 (1-1000)")
    training_group.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率 (1e-6 到 1e-1)")
    training_group.add_argument("--weight_decay", type=float, default=0.01,
                       help="权重衰减系数 (0-1)")
    training_group.add_argument("--warmup_steps", type=int, default=500,
                       help="学习率预热步数")
    training_group.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="梯度累积步数 (1-128)")
    training_group.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="梯度裁剪阈值 (0.1-10)")
    training_group.add_argument("--use_amp", action="store_true",
                       help="使用自动混合精度训练")
    
    # 保存与日志
    checkpoint_group = parser.add_argument_group('Checkpoint Configuration', '检查点和日志配置')
    checkpoint_group.add_argument("--output_dir", type=str, default="outputs",
                       help="输出目录路径")
    checkpoint_group.add_argument("--save_steps", type=int, default=1000,
                       help="保存检查点的步数间隔")
    checkpoint_group.add_argument("--logging_steps", type=int, default=100,
                       help="日志记录的步数间隔")
    checkpoint_group.add_argument("--eval_steps", type=int, default=500,
                       help="评估的步数间隔")
    
    # SwanLab配置
    swanlab_group = parser.add_argument_group('SwanLab Configuration', 'SwanLab实验跟踪配置')
    swanlab_group.add_argument("--swanlab_project", type=str, default="VLLM",
                       help="swanlab项目名")
    swanlab_group.add_argument("--swanlab_workspace", type=str, default="delahayecarry",
                       help="swanlab工作空间")
    swanlab_group.add_argument("--swanlab_experiment_name", type=str, default="",
                       help="实验名称（可选，留空自动生成）")
    swanlab_group.add_argument("--swanlab_description", type=str, default="",
                       help="实验描述")
    swanlab_group.add_argument("--swanlab_logdir", type=str, default="",
                       help="日志目录")
    
    # SwanLab API配置（推荐使用环境变量）
    api_group = parser.add_argument_group('API Configuration', '推荐使用环境变量配置')
    api_group.add_argument("--swanlab_api_key", type=str, default="",
                       help="SwanLab API密钥（推荐使用环境变量 SWANLAB_API_KEY）")
    
    # 恢复训练
    parser.add_argument("--resume_from_checkpoint", type=str, default="",
                       help="从检查点恢复训练")
    
    return parser.parse_args()


def main():
    """主函数 - 支持YAML配置和CLI参数合并"""
    args = parse_args()
    
    # 检查已弃用参数
    if hasattr(args, 'swanlab_api_key') and args.swanlab_api_key:
        print("\n警告: --swanlab_api_key 参数不推荐使用")
        print("请使用环境变量:")
        print("  export SWANLAB_API_KEY='your_api_key_here'")
        print()
    
    print("=" * 70)
    print("Zero2LLMV 训练框架 - YAML配置支持")
    print("=" * 70)
    
    # 加载YAML配置（如果提供）
    yaml_config = {}
    if args.config:
        try:
            print(f"正在加载YAML配置: {args.config}")
            yaml_config = load_yaml_config(args.config)
            print("✓ YAML配置加载成功")
        except Exception as e:
            print(f"✗ YAML配置加载失败: {str(e)}")
            if not getattr(args, 'allow_default_fallback', False):
                sys.exit(1)
            print("使用默认配置继续...")
    else:
        print("未指定YAML配置文件，使用CLI参数")
    
    # 合并配置
    try:
        print("正在合并配置参数...")
        merged_config = merge_config_with_args(yaml_config, args)
        print("✓ 配置参数合并成功")
    except Exception as e:
        print(f"✗ 配置合并失败: {str(e)}")
        sys.exit(1)
    
    # 验证配置
    try:
        print("正在验证配置...")
        validated_config = validate_config(
            merged_config, 
            allow_defaults=getattr(args, 'allow_default_fallback', False)
        )
        print("✓ 配置验证成功")
    except ValidationError as e:
        print(f"✗ 配置验证失败:\n{str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 配置验证遇到错误: {str(e)}")
        sys.exit(1)
    
    # 转换为旧版TrainingConfig格式
    config_dict = validated_config.to_training_config_dict()
    
    # 从环境变量获取敏感信息
    config_dict['swanlab_api_key'] = os.environ.get('SWANLAB_API_KEY', '')
    
    # 创建TrainingConfig实例
    config = TrainingConfig(**config_dict)
    
    # 保存合并后的配置（如果请求）
    if getattr(args, 'save_merged_config', False):
        try:
            print("正在保存合并配置...")
            # 保存到输出目录
            save_merged_config(merged_config, config.output_dir)
            print("✓ 合并配置保存成功")
        except Exception as e:
            print(f"警告: 无法保存合并配置: {str(e)}")
    
    # 显示最终配置
    print()
    print("=" * 70)
    print("最终训练配置:")
    print("=" * 70)
    
    # 按组织结构显示配置
    config_sections = {
        "模型配置": {
            "model_type": config.model_type,
            "model_config_path": config.model_config_path or "[使用默认配置]"
        },
        "数据配置": {
            "data_path": config.data_path,
            "max_seq_length": config.max_seq_length,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers
        },
        "训练配置": {
            "num_epochs": config.num_epochs,
            "learning_rate": f"{config.learning_rate:.2e}",
            "weight_decay": config.weight_decay,
            "warmup_steps": config.warmup_steps,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_grad_norm": config.max_grad_norm,
            "use_amp": config.use_amp
        },
        "检查点配置": {
            "output_dir": config.output_dir,
            "save_steps": config.save_steps,
            "logging_steps": config.logging_steps,
            "eval_steps": config.eval_steps
        }
    }
    
    for section_name, section_config in config_sections.items():
        print(f"\n{section_name}:")
        for key, value in section_config.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 恢复检查点
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()