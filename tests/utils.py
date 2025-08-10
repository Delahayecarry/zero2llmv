"""
测试工具函数
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, Any
import numpy as np


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str = "tensor"):
    """断言张量形状"""
    actual_shape = tuple(tensor.shape)
    assert actual_shape == expected_shape, (
        f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"
    )


def assert_tensor_close(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                       rtol: float = 1e-4, atol: float = 1e-6, name: str = "tensors"):
    """断言两个张量数值接近"""

    assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), (
        f"{name} values not close enough. "
        f"Max diff: {torch.max(torch.abs(tensor1 - tensor2)).item():.8f}"
    )


def count_parameters(module: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def check_gradient_flow(model: nn.Module, loss: torch.Tensor) -> dict:
    """检查梯度流"""
    loss.backward()
    
    gradient_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_stats[name] = {
                'grad_norm': grad_norm,
                'param_norm': param.norm().item(),
                'has_nan': torch.isnan(param.grad).any().item(),
                'has_inf': torch.isinf(param.grad).any().item()
            }
        else:
            gradient_stats[name] = {'grad_norm': 0.0, 'no_grad': True}
    
    return gradient_stats


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """创建因果掩码用于测试"""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask


def create_padding_mask(batch_size: int, seq_len: int, 
                       pad_ratio: float = 0.2, device: torch.device = None) -> torch.Tensor:
    """创建填充掩码用于测试"""
    mask = torch.ones(batch_size, seq_len, device=device)
    for i in range(batch_size):
        # 随机选择填充长度
        pad_len = int(seq_len * pad_ratio * torch.rand(1).item())
        if pad_len > 0:
            # 在序列末尾添加padding
            mask[i, -pad_len:] = 0
    return mask


def measure_memory_usage(func, *args, **kwargs):
    """测量函数内存使用"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        result = func(*args, **kwargs)
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return result, {'allocated': memory_allocated, 'peak': memory_peak}
    else:
        # CPU情况下简单返回结果
        return func(*args, **kwargs), {'allocated': 0, 'peak': 0}


def set_seed(seed: int = 42):
    """设置随机种子以确保测试可重复"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class ModelTester:
    """模型测试工具类"""
    
    @staticmethod
    def test_forward_backward(model: nn.Module, input_ids: torch.Tensor, 
                            target_shape: Optional[Tuple[int, ...]] = None) -> dict:
        """测试前向和反向传播"""
        model.train()
        set_seed(42)
        
        # 前向传播
        output = model(input_ids)
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
            
        if target_shape:
            assert_tensor_shape(logits, target_shape, "model output")
            
        # 反向传播测试
        loss = logits.mean()
        gradient_stats = check_gradient_flow(model, loss)
        
        return {
            'output_shape': tuple(logits.shape),
            'output_mean': logits.mean().item(),
            'output_std': logits.std().item(),
            'gradient_stats': gradient_stats,
            'has_nan_output': torch.isnan(logits).any().item(),
            'has_inf_output': torch.isinf(logits).any().item()
        }
    
    @staticmethod
    def test_deterministic(model: nn.Module, input_ids: torch.Tensor, 
                          num_runs: int = 3) -> bool:
        """测试模型确定性（相同输入应产生相同输出）"""
        model.eval()
        outputs = []
        
        for _ in range(num_runs):
            set_seed(42)
            with torch.no_grad():
                output = model(input_ids)
                if hasattr(output, 'logits'):
                    outputs.append(output.logits.clone())
                else:
                    outputs.append(output.clone())
        
        # 检查所有输出是否相同
        for i in range(1, num_runs):
            if not torch.allclose(outputs[0], outputs[i], rtol=1e-5, atol=1e-7):
                return False
        return True
    
    @staticmethod
    def test_different_batch_sizes(model: nn.Module, base_input: torch.Tensor, 
                                 batch_sizes: list = [1, 2, 4]) -> dict:
        """测试不同批次大小"""
        model.eval()
        results = {}
        
        seq_len = base_input.shape[1]
        vocab_size = base_input.max().item() + 1
        
        for batch_size in batch_sizes:
            set_seed(42)
            test_input = torch.randint(0, vocab_size, (batch_size, seq_len), 
                                     device=base_input.device)
            
            with torch.no_grad():
                output = model(test_input)
                if hasattr(output, 'logits'):
                    logits = output.logits
                else:
                    logits = output
                    
                results[batch_size] = {
                    'shape': tuple(logits.shape),
                    'mean': logits.mean().item(),
                    'std': logits.std().item()
                }
        
        return results