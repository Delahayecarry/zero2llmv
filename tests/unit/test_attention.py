"""
注意力机制测试
"""
import pytest
import torch
import math
from models.llm import Attention, precompute_freqs_cis
from configs.llmconfig import llmconfig
from tests.utils import assert_tensor_shape, assert_tensor_close, set_seed, ModelTester


class TestAttention:
    """测试注意力机制"""
    
    def test_attention_forward_basic(self):
        """测试基础注意力前向传播"""
        config = llmconfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
            num_hidden_layers=2,
            dropout=0.0  # 关闭dropout以便测试
        )
        
        attention = Attention(config)
        
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # 准备位置编码
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, 
            seq_len, 
            config.rope_theta
        )
        position_embeddings = (cos, sin)
        
        # 前向传播
        output, past_kv = attention(x, position_embeddings)
        
        # 检查输出形状
        assert_tensor_shape(output, (batch_size, seq_len, config.hidden_size), "attention output")
        assert past_kv is None  # use_cache=False时应该为None
        
    def test_attention_with_cache(self):
        """测试带缓存的注意力"""
        config = llmconfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            dropout=0.0
        )
        
        attention = Attention(config)
        
        batch_size = 1
        seq_len = 8
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # 位置编码
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, 
            seq_len * 2,  # 准备更长的位置编码
            config.rope_theta
        )
        position_embeddings = (cos[:seq_len], sin[:seq_len])
        
        # 第一次前向传播（启用缓存）
        output1, past_kv = attention(x, position_embeddings, use_cache=True)
        
        # 检查KV缓存
        assert past_kv is not None
        assert len(past_kv) == 2  # key和value
        
        key_cache, value_cache = past_kv
        expected_cache_shape = (batch_size, seq_len, config.num_key_value_heads, 
                               config.hidden_size // config.num_attention_heads)
        assert_tensor_shape(key_cache, expected_cache_shape, "key cache")
        assert_tensor_shape(value_cache, expected_cache_shape, "value cache")
        
        # 第二次前向传播（使用缓存）
        x2 = torch.randn(batch_size, 4, config.hidden_size)
        position_embeddings2 = (cos[seq_len:seq_len+4], sin[seq_len:seq_len+4])
        
        output2, past_kv2 = attention(x2, position_embeddings2, 
                                    past_key_value=past_kv, use_cache=True)
        
        # 检查输出形状
        assert_tensor_shape(output2, (batch_size, 4, config.hidden_size), "cached attention output")
        
        # 检查新的KV缓存形状
        key_cache2, value_cache2 = past_kv2
        expected_cache_shape2 = (batch_size, seq_len + 4, config.num_key_value_heads,
                                config.hidden_size // config.num_attention_heads)
        assert_tensor_shape(key_cache2, expected_cache_shape2, "extended key cache")
        assert_tensor_shape(value_cache2, expected_cache_shape2, "extended value cache")
        
    def test_grouped_query_attention(self):
        """测试分组查询注意力(GQA)"""
        config = llmconfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,  # GQA: KV头数少于Query头数
            dropout=0.0
        )
        
        attention = Attention(config)
        
        # 验证头数配置
        assert attention.n_local_heads == 8
        assert attention.n_local_kv_heads == 4
        assert attention.rep == 2  # 每个KV头对应2个Query头
        
        batch_size = 2
        seq_len = 12
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            seq_len,
            config.rope_theta
        )
        position_embeddings = (cos, sin)
        
        output, _ = attention(x, position_embeddings)
        
        # 输出形状应该正确
        assert_tensor_shape(output, (batch_size, seq_len, config.hidden_size), "GQA output")
        
    def test_attention_mask(self):
        """测试注意力掩码"""
        config = llmconfig(
            hidden_size=128,
            num_attention_heads=4,
            dropout=0.0
        )
        
        attention = Attention(config)
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # 创建注意力掩码（第一个序列的后半部分被mask掉）
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, seq_len//2:] = 0  # mask掉后半部分
        
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            seq_len,
            config.rope_theta
        )
        position_embeddings = (cos, sin)
        
        # 带掩码的前向传播
        output_with_mask, _ = attention(x, position_embeddings, attention_mask=attention_mask)
        
        # 不带掩码的前向传播
        output_without_mask, _ = attention(x, position_embeddings)
        
        # 输出形状应该相同
        assert output_with_mask.shape == output_without_mask.shape
        
        # 第二个序列（没有被mask）的输出应该相同
        assert_tensor_close(
            output_with_mask[1], 
            output_without_mask[1], 
            rtol=1e-5, 
            name="unmasked sequence output"
        )
        
    def test_causal_mask(self):
        """测试因果掩码（确保只能看到之前的位置）"""
        config = llmconfig(
            hidden_size=64,
            num_attention_heads=2,
            dropout=0.0
        )
        
        attention = Attention(config)
        
        batch_size = 1
        seq_len = 4
        
        # 创建特殊的输入，每个位置都不同
        x = torch.zeros(batch_size, seq_len, config.hidden_size)
        for i in range(seq_len):
            x[0, i, :] = i + 1  # 位置i的值为i+1
            
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            seq_len,
            config.rope_theta
        )
        position_embeddings = (cos, sin)
        
        attention.eval()  # 设置为eval模式以避免dropout
        with torch.no_grad():
            output, _ = attention(x, position_embeddings)
            
        # 由于因果掩码，每个位置只能看到自己和之前的位置
        # 这个测试主要确保没有报错，具体的因果性需要更复杂的测试
        assert_tensor_shape(output, (batch_size, seq_len, config.hidden_size))
        
    def test_attention_different_seq_lengths(self):
        """测试不同序列长度"""
        config = llmconfig(
            hidden_size=128,
            num_attention_heads=4,
            dropout=0.0
        )
        
        attention = Attention(config)
        
        for seq_len in [1, 4, 8, 16, 32]:
            batch_size = 2
            x = torch.randn(batch_size, seq_len, config.hidden_size)
            
            cos, sin = precompute_freqs_cis(
                config.hidden_size // config.num_attention_heads,
                seq_len,
                config.rope_theta
            )
            position_embeddings = (cos, sin)
            
            output, _ = attention(x, position_embeddings)
            
            expected_shape = (batch_size, seq_len, config.hidden_size)
            assert_tensor_shape(output, expected_shape, f"seq_len={seq_len}")
            
    def test_attention_gradient_flow(self):
        """测试注意力机制的梯度流"""
        config = llmconfig(
            hidden_size=128,
            num_attention_heads=4,
            dropout=0.0
        )
        
        attention = Attention(config)
        attention.train()
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)
        
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            seq_len,
            config.rope_theta
        )
        position_embeddings = (cos, sin)
        
        output, _ = attention(x, position_embeddings)
        loss = output.mean()
        
        loss.backward()
        
        # 检查输入梯度
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        # 检查参数梯度
        for name, param in attention.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
                
    def test_attention_deterministic(self):
        """测试注意力机制的确定性"""
        config = llmconfig(
            hidden_size=128,
            num_attention_heads=4,
            dropout=0.0
        )
        
        attention = Attention(config)
        attention.eval()
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            seq_len,
            config.rope_theta
        )
        position_embeddings = (cos, sin)
        
        # 多次运行应该产生相同的结果
        outputs = []
        for _ in range(3):
            set_seed(42)
            with torch.no_grad():
                output, _ = attention(x, position_embeddings)
                outputs.append(output.clone())
                
        for i in range(1, len(outputs)):
            assert_tensor_close(outputs[0], outputs[i], name=f"run {i}")
            
    def test_flash_attention_disabled(self):
        """测试禁用Flash Attention的情况"""
        config = llmconfig(
            hidden_size=128,
            num_attention_heads=4,
            flash_attn=False,
            dropout=0.0
        )
        
        attention = Attention(config)
        assert attention.flash == False
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            seq_len,
            config.rope_theta
        )
        position_embeddings = (cos, sin)
        
        # 应该正常工作
        output, _ = attention(x, position_embeddings)
        assert_tensor_shape(output, (batch_size, seq_len, config.hidden_size))
        
    def test_attention_head_dimension(self):
        """测试注意力头维度计算"""
        configs = [
            (256, 8),   # head_dim = 32
            (512, 16),  # head_dim = 32  
            (768, 12),  # head_dim = 64
        ]
        
        for hidden_size, num_heads in configs:
            config = llmconfig(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                dropout=0.0
            )
            
            attention = Attention(config)
            expected_head_dim = hidden_size // num_heads
            assert attention.head_dim == expected_head_dim
            
    def test_attention_parameter_count(self):
        """测试注意力层参数数量"""
        config = llmconfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,  # GQA
            dropout=0.0
        )
        
        attention = Attention(config)
        
        # 计算预期参数数量
        # Q投影：hidden_size * (num_attention_heads * head_dim)
        # K投影：hidden_size * (num_key_value_heads * head_dim)  
        # V投影：hidden_size * (num_key_value_heads * head_dim)
        # O投影：(num_attention_heads * head_dim) * hidden_size
        
        head_dim = config.hidden_size // config.num_attention_heads
        
        q_params = config.hidden_size * (config.num_attention_heads * head_dim)
        kv_params = config.hidden_size * (config.num_key_value_heads * head_dim) * 2  # K + V
        o_params = (config.num_attention_heads * head_dim) * config.hidden_size
        
        expected_params = q_params + kv_params + o_params
        actual_params = sum(p.numel() for p in attention.parameters())
        
        assert actual_params == expected_params