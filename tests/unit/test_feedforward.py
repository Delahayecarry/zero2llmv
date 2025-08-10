"""
前馈网络测试
"""
import pytest
import torch
from models.llm import FeedForward
from configs.llmconfig import llmconfig
from tests.utils import assert_tensor_shape, assert_tensor_close, set_seed, count_parameters


class TestFeedForward:
    """测试前馈网络"""
    
    def test_feedforward_forward(self):
        """测试前馈网络前向传播"""
        config = llmconfig(
            hidden_size=256,
            intermediate_size=512,  # 指定中间层维度
            hidden_act="gelu",
            dropout=0.0
        )
        
        ff = FeedForward(config)
        
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = ff(x)
        
        # 检查输出形状
        assert_tensor_shape(output, (batch_size, seq_len, config.hidden_size), "FeedForward output")
        
    def test_feedforward_auto_intermediate_size(self):
        """测试中间层维度的自动计算"""
        config = llmconfig(
            hidden_size=768,
            intermediate_size=None,  # 自动计算
            hidden_act="gelu",
            dropout=0.0
        )
        
        ff = FeedForward(config)
        
        # 应该自动设置为 ~8/3 * hidden_size，并对齐到64的倍数
        expected_intermediate = int(768 * 8 / 3)  # 2048
        expected_aligned = 64 * ((expected_intermediate + 64 - 1) // 64)  # 对齐到64
        
        assert config.intermediate_size == expected_aligned
        
    def test_feedforward_different_activations(self):
        """测试不同的激活函数"""
        activations = ["gelu", "relu", "silu", "swish"]
        
        for act in activations:
            config = llmconfig(
                hidden_size=128,
                intermediate_size=256,
                hidden_act=act,
                dropout=0.0
            )
            
            ff = FeedForward(config)
            
            batch_size = 1
            seq_len = 8
            x = torch.randn(batch_size, seq_len, config.hidden_size)
            
            # 前向传播应该成功
            output = ff(x)
            assert_tensor_shape(output, (batch_size, seq_len, config.hidden_size), 
                              f"FeedForward with {act}")
            
    def test_feedforward_swiGLU_computation(self):
        """测试SwiGLU计算的正确性"""
        config = llmconfig(
            hidden_size=64,
            intermediate_size=128,
            hidden_act="silu",  # SwiGLU使用SiLU激活
            dropout=0.0
        )
        
        ff = FeedForward(config)
        
        # 使用简单的输入进行测试
        x = torch.ones(1, 1, config.hidden_size)
        
        # 手动计算SwiGLU
        gate = ff.gate_proj(x)  # 门控投影
        up = ff.up_proj(x)      # 上采样投影
        activated_gate = ff.act_fn(gate)  # 激活门控
        
        # SwiGLU = down_proj(activated_gate * up)
        manual_output = ff.down_proj(activated_gate * up)
        
        # 自动计算
        auto_output = ff(x)
        
        # 结果应该相同（除了dropout，这里已经设为0）
        assert_tensor_close(manual_output, auto_output, name="SwiGLU computation")
        
    def test_feedforward_gradient_flow(self):
        """测试前馈网络的梯度流"""
        config = llmconfig(
            hidden_size=128,
            intermediate_size=256,
            dropout=0.0
        )
        
        ff = FeedForward(config)
        ff.train()
        
        x = torch.randn(2, 8, config.hidden_size, requires_grad=True)
        
        output = ff(x)
        loss = output.sum()
        loss.backward()
        
        # 检查输入梯度
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        # 检查参数梯度
        for name, param in ff.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN in {name} gradient"
                
    def test_feedforward_parameter_count(self):
        """测试前馈网络参数数量"""
        config = llmconfig(
            hidden_size=256,
            intermediate_size=512,
            dropout=0.0
        )
        
        ff = FeedForward(config)
        
        # 计算预期参数数量
        # gate_proj: hidden_size * intermediate_size
        # up_proj: hidden_size * intermediate_size  
        # down_proj: intermediate_size * hidden_size
        expected_params = (
            config.hidden_size * config.intermediate_size +  # gate_proj
            config.hidden_size * config.intermediate_size +  # up_proj
            config.intermediate_size * config.hidden_size    # down_proj
        )
        
        actual_params = count_parameters(ff)
        assert actual_params == expected_params
        
    def test_feedforward_dropout(self):
        """测试dropout的影响"""
        config = llmconfig(
            hidden_size=128,
            intermediate_size=256,
            dropout=0.5  # 高dropout率
        )
        
        ff = FeedForward(config)
        
        x = torch.randn(2, 8, config.hidden_size)
        
        # 训练模式：dropout应该生效
        ff.train()
        output_train1 = ff(x)
        output_train2 = ff(x)
        
        # 由于dropout的随机性，两次输出应该不同
        assert not torch.allclose(output_train1, output_train2)
        
        # 评估模式：dropout应该关闭
        ff.eval()
        with torch.no_grad():
            output_eval1 = ff(x)
            output_eval2 = ff(x)
            
        # 评估模式下，两次输出应该相同
        assert_tensor_close(output_eval1, output_eval2, name="eval mode consistency")