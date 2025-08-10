"""
基础组件测试：RMSNorm、位置编码等
"""
import pytest
import torch
import math
from models.llm import RMSNorm, precompute_freqs_cis, apply_rotary_pos_emb, repeat_kv
from tests.utils import assert_tensor_shape, assert_tensor_close, set_seed


class TestRMSNorm:
    """测试RMSNorm层归一化"""
    
    def test_rms_norm_forward(self):
        """测试RMSNorm前向传播"""
        dim = 512
        batch_size = 2
        seq_len = 10
        
        rms_norm = RMSNorm(dim)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = rms_norm(x)
        
        # 检查输出形状
        assert_tensor_shape(output, (batch_size, seq_len, dim), "RMSNorm output")
        
        # 检查归一化效果（均方根应该接近1）
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1, keepdim=True))
        expected_rms = torch.ones_like(rms)
        assert_tensor_close(rms, expected_rms, rtol=1e-3, name="RMS values")
        
    def test_rms_norm_zero_input(self):
        """测试零输入情况"""
        dim = 256
        rms_norm = RMSNorm(dim, eps=1e-5)
        x = torch.zeros(2, 5, dim)
        
        output = rms_norm(x)
        # 零输入应该输出零（或接近零）
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)
        
    def test_rms_norm_backward(self):
        """测试RMSNorm反向传播"""
        dim = 128
        rms_norm = RMSNorm(dim)
        x = torch.randn(2, 8, dim, requires_grad=True)
        
        output = rms_norm(x)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度存在
        assert x.grad is not None
        assert rms_norm.weight.grad is not None
        
        # 检查梯度不包含NaN或Inf
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert not torch.isnan(rms_norm.weight.grad).any()
        
    def test_rms_norm_different_dtypes(self):
        """测试不同数据类型"""
        dim = 256
        rms_norm = RMSNorm(dim)
        
        # 测试float16
        x_fp16 = torch.randn(1, 4, dim, dtype=torch.float16)
        output_fp16 = rms_norm(x_fp16)
        assert output_fp16.dtype == torch.float16
        
        # 测试float32
        x_fp32 = torch.randn(1, 4, dim, dtype=torch.float32)
        output_fp32 = rms_norm(x_fp32)
        assert output_fp32.dtype == torch.float32


class TestPositionalEncoding:
    """测试位置编码相关函数"""
    
    def test_precompute_freqs_cis(self):
        """测试预计算频率"""
        dim = 64
        end = 128
        theta = 10000.0
        
        freqs_cos, freqs_sin = precompute_freqs_cis(dim, end, theta)
        
        # 检查输出形状
        assert_tensor_shape(freqs_cos, (end, dim), "cos frequencies")
        assert_tensor_shape(freqs_sin, (end, dim), "sin frequencies")
        
        # 检查数值范围（cos和sin应该在[-1, 1]范围内）
        assert torch.all(freqs_cos >= -1.0) and torch.all(freqs_cos <= 1.0)
        assert torch.all(freqs_sin >= -1.0) and torch.all(freqs_sin <= 1.0)
        
        # 检查cos²+sin²=1（三角恒等式）
        cos_sin_squared = freqs_cos ** 2 + freqs_sin ** 2
        expected = torch.ones_like(cos_sin_squared)
        assert_tensor_close(cos_sin_squared, expected, rtol=1e-5, name="cos²+sin²")
        
    def test_apply_rotary_pos_emb(self):
        """测试应用旋转位置编码"""
        batch_size = 2
        seq_len = 16
        num_heads = 8
        head_dim = 64
        
        # 准备输入
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        # 预计算频率
        cos, sin = precompute_freqs_cis(head_dim, seq_len, 10000.0)
        
        # 应用位置编码
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 检查输出形状
        assert_tensor_shape(q_embed, (batch_size, seq_len, num_heads, head_dim), "q_embed")
        assert_tensor_shape(k_embed, (batch_size, seq_len, num_heads, head_dim), "k_embed")
        
        # 检查RoPE的旋转特性：||q_embed|| = ||q||
        q_norm = torch.norm(q, dim=-1)
        q_embed_norm = torch.norm(q_embed, dim=-1)
        assert_tensor_close(q_norm, q_embed_norm, rtol=1e-5, name="RoPE norm preservation")
        
    def test_rotary_pos_emb_different_seq_lengths(self):
        """测试不同序列长度的位置编码"""
        batch_size = 1
        num_heads = 4
        head_dim = 32
        
        for seq_len in [8, 16, 32, 64]:
            q = torch.randn(batch_size, seq_len, num_heads, head_dim)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim)
            
            cos, sin = precompute_freqs_cis(head_dim, seq_len * 2, 10000.0)  # 预计算更长的序列
            cos_slice = cos[:seq_len]
            sin_slice = sin[:seq_len]
            
            q_embed, k_embed = apply_rotary_pos_emb(q, k, cos_slice, sin_slice)
            
            assert_tensor_shape(q_embed, (batch_size, seq_len, num_heads, head_dim))
            assert_tensor_shape(k_embed, (batch_size, seq_len, num_heads, head_dim))


class TestRepeatKV:
    """测试KV重复函数（用于分组查询注意力）"""
    
    def test_repeat_kv_basic(self):
        """测试基础KV重复功能"""
        batch_size = 2
        seq_len = 16
        num_kv_heads = 4
        head_dim = 64
        n_rep = 2
        
        kv = torch.randn(batch_size, seq_len, num_kv_heads, head_dim)
        repeated_kv = repeat_kv(kv, n_rep)
        
        expected_shape = (batch_size, seq_len, num_kv_heads * n_rep, head_dim)
        assert_tensor_shape(repeated_kv, expected_shape, "repeated KV")
        
        # 检查重复的正确性
        for i in range(num_kv_heads):
            for j in range(n_rep):
                original_head = kv[:, :, i, :]
                repeated_head = repeated_kv[:, :, i * n_rep + j, :]
                assert_tensor_close(original_head, repeated_head, name=f"head {i} rep {j}")
                
    def test_repeat_kv_no_repeat(self):
        """测试n_rep=1的情况（不应该改变张量）"""
        batch_size = 2
        seq_len = 8
        num_kv_heads = 6
        head_dim = 32
        
        kv = torch.randn(batch_size, seq_len, num_kv_heads, head_dim)
        repeated_kv = repeat_kv(kv, 1)
        
        # 应该返回完全相同的张量
        assert torch.equal(kv, repeated_kv)
        
    def test_repeat_kv_different_ratios(self):
        """测试不同的重复比例"""
        batch_size = 1
        seq_len = 4
        num_kv_heads = 2
        head_dim = 16
        
        kv = torch.randn(batch_size, seq_len, num_kv_heads, head_dim)
        
        for n_rep in [1, 2, 4, 8]:
            repeated_kv = repeat_kv(kv, n_rep)
            expected_shape = (batch_size, seq_len, num_kv_heads * n_rep, head_dim)
            assert_tensor_shape(repeated_kv, expected_shape, f"n_rep={n_rep}")


class TestHelperFunctions:
    """测试其他辅助函数"""
    
    def test_math_constants(self):
        """测试数学常数的使用"""
        # 测试sqrt在RoPE中的使用
        head_dim = 64
        scale = 1.0 / math.sqrt(head_dim)
        assert scale == 1.0 / 8.0
        
    def test_device_consistency(self):
        """测试设备一致性"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        dim = 128
        seq_len = 32
        
        # 测试RMSNorm
        rms_norm = RMSNorm(dim).to(device)
        x = torch.randn(1, seq_len, dim, device=device)
        output = rms_norm(x)
        assert output.device == device
        
        # 测试位置编码
        cos, sin = precompute_freqs_cis(dim, seq_len)
        cos = cos.to(device)
        sin = sin.to(device)
        
        q = torch.randn(1, seq_len, 4, dim, device=device)
        k = torch.randn(1, seq_len, 4, dim, device=device)
        
        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_embed.device == device
        assert k_embed.device == device
        
    def test_deterministic_behavior(self):
        """测试确定性行为"""
        dim = 256
        seq_len = 16
        
        # 使用相同的种子应该产生相同的结果
        set_seed(42)
        cos1, sin1 = precompute_freqs_cis(dim, seq_len)
        
        set_seed(42)
        cos2, sin2 = precompute_freqs_cis(dim, seq_len)
        
        assert_tensor_close(cos1, cos2, name="deterministic cos")
        assert_tensor_close(sin1, sin2, name="deterministic sin")