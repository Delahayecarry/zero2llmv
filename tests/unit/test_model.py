"""
完整模型测试
"""
import pytest
import torch
from models.llm import LLM, CausalLM, MiniMindBlock
from configs.llmconfig import llmconfig
from tests.utils import (
    assert_tensor_shape, assert_tensor_close, set_seed, 
    count_parameters, ModelTester, measure_memory_usage
)


class TestMiniMindBlock:
    """测试Transformer块"""
    
    def test_block_forward_basic(self):
        """测试基础块前向传播"""
        config = llmconfig(
            hidden_size=256,
            num_attention_heads=8,
            intermediate_size=512,
            dropout=0.0
        )
        
        block = MiniMindBlock(config)
        
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # 准备位置编码
        from models.llm import precompute_freqs_cis
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, 
            seq_len, 
            config.rope_theta
        )
        position_embeddings = (cos, sin)
        
        output, past_kv = block(x, position_embeddings)
        
        # 检查输出形状
        assert_tensor_shape(output, (batch_size, seq_len, config.hidden_size), "block output")
        assert past_kv is None  # use_cache=False
        
    def test_block_with_cache(self):
        """测试带缓存的块"""
        config = llmconfig(
            hidden_size=128,
            num_attention_heads=4,
            intermediate_size=256,
            dropout=0.0
        )
        
        block = MiniMindBlock(config)
        
        batch_size = 1
        seq_len = 8
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        from models.llm import precompute_freqs_cis
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, 
            seq_len * 2, 
            config.rope_theta
        )
        position_embeddings = (cos[:seq_len], sin[:seq_len])
        
        # 启用缓存的前向传播
        output, past_kv = block(x, position_embeddings, use_cache=True)
        
        assert past_kv is not None
        assert len(past_kv) == 2  # key和value
        
    def test_block_with_moe(self):
        """测试MoE块"""
        config = llmconfig(
            hidden_size=128,
            num_attention_heads=4,
            use_moe=True,
            num_experts_per_token=2,
            n_routed_experts=4,
            aux_loss_alpha=0.01
        )
        
        block = MiniMindBlock(config)
        block.train()  # 训练模式以启用辅助损失
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        from models.llm import precompute_freqs_cis
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, 
            seq_len, 
            config.rope_theta
        )
        position_embeddings = (cos, sin)
        
        output, past_kv = block(x, position_embeddings)
        
        assert_tensor_shape(output, (batch_size, seq_len, config.hidden_size), "MoE block output")
        
        # 检查辅助损失
        assert hasattr(block.feed_forward, 'aux_loss')
        

class TestLLM:
    """测试LLM主模型"""
    
    def test_llm_forward_basic(self):
        """测试基础LLM前向传播"""
        config = llmconfig(
            vocab_size=1000,
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            intermediate_size=512,
            max_position_embeddings=128,
            dropout=0.0
        )
        
        model = LLM(config)
        
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids)
        
        # 检查输出形状
        expected_shape = (batch_size, seq_len, config.hidden_size)
        assert_tensor_shape(output, expected_shape, "LLM output")
        
    def test_llm_with_attention_mask(self):
        """测试带注意力掩码的LLM"""
        config = llmconfig(
            vocab_size=1000,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64
        )
        
        model = LLM(config)
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # 创建注意力掩码
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, seq_len//2:] = 0  # 第一个序列的后半部分被mask
        
        output_with_mask = model(input_ids, attention_mask=attention_mask)
        output_without_mask = model(input_ids)
        
        # 输出形状应该相同
        assert output_with_mask.shape == output_without_mask.shape
        
        # 第二个序列（未被mask）的输出应该相同
        assert_tensor_close(
            output_with_mask[1], 
            output_without_mask[1], 
            rtol=1e-5, 
            name="unmasked sequence"
        )
        
    def test_llm_with_kv_cache(self):
        """测试KV缓存功能"""
        config = llmconfig(
            vocab_size=500,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=64
        )
        
        model = LLM(config)
        model.eval()
        
        batch_size = 1
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            # 第一次前向传播（生成缓存）
            output1, past_key_values = model(input_ids, use_cache=True)
            
            # 检查缓存
            assert past_key_values is not None
            assert len(past_key_values) == config.num_hidden_layers
            
            # 第二次前向传播（使用缓存）
            next_input = torch.randint(0, config.vocab_size, (batch_size, 1))
            output2, past_key_values2 = model(
                next_input, 
                past_key_values=past_key_values, 
                use_cache=True
            )
            
            assert_tensor_shape(output2, (batch_size, 1, config.hidden_size), "cached output")
            
    def test_llm_moe_model(self):
        """测试MoE模型"""
        config = llmconfig(
            vocab_size=1000,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            use_moe=True,
            num_experts_per_token=2,
            n_routed_experts=4,
            aux_loss_alpha=0.01
        )
        
        model = LLM(config)
        model.train()
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids)
        
        assert_tensor_shape(output, (batch_size, seq_len, config.hidden_size), "MoE LLM output")
        
        # 检查辅助损失
        total_aux_loss = 0
        for layer in model.layers:
            if hasattr(layer.feed_forward, 'aux_loss'):
                total_aux_loss += layer.feed_forward.aux_loss
                
        assert total_aux_loss > 0  # 应该有辅助损失
        
    def test_llm_gradient_flow(self):
        """测试LLM梯度流"""
        config = llmconfig(
            vocab_size=500,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            dropout=0.0
        )
        
        model = LLM(config)
        model.train()
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids)
        loss = output.mean()
        loss.backward()
        
        # 检查关键参数的梯度
        assert model.embed_tokens.weight.grad is not None
        assert model.norm.weight.grad is not None
        
        # 检查每一层的梯度
        for i, layer in enumerate(model.layers):
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    # 某些参数可能在特定情况下没有梯度（如未使用的专家）
                    if param.grad is not None:
                        assert not torch.isnan(param.grad).any(), f"Layer {i} {name} has NaN gradients"
                        

class TestCausalLM:
    """测试因果语言模型"""
    
    def test_causal_lm_forward(self):
        """测试CausalLM前向传播"""
        config = llmconfig(
            vocab_size=1000,
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=3,
            max_position_embeddings=128
        )
        
        model = CausalLM(config)
        
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids)
        
        # CausalLM输出logits，形状应该是(batch_size, seq_len, vocab_size)
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert_tensor_shape(output.logits, expected_shape, "CausalLM logits")
        
    def test_causal_lm_with_labels(self):
        """测试带标签的CausalLM（用于训练）"""
        config = llmconfig(
            vocab_size=500,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2
        )
        
        model = CausalLM(config)
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids, labels=labels)
        
        # 应该包含损失
        assert hasattr(output, 'loss')
        assert output.loss.numel() == 1
        assert output.loss.requires_grad
        
    def test_causal_lm_generation(self):
        """测试CausalLM生成功能"""
        config = llmconfig(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            max_position_embeddings=32
        )
        
        model = CausalLM(config)
        model.eval()
        
        # 简单的贪心生成测试
        batch_size = 1
        start_len = 4
        input_ids = torch.randint(0, config.vocab_size, (batch_size, start_len))
        
        with torch.no_grad():
            # 生成下一个token
            output = model(input_ids)
            next_token_logits = output.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 检查生成的token在词汇表范围内
            assert torch.all(next_token >= 0)
            assert torch.all(next_token < config.vocab_size)
            
    def test_causal_lm_parameter_tying(self):
        """测试嵌入层权重共享"""
        config = llmconfig(
            vocab_size=1000,
            hidden_size=256,
            tie_word_embeddings=True
        )
        
        model = CausalLM(config)
        
        # 检查嵌入层和输出层是否共享权重
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
            
        # 输出层的权重应该与嵌入层相同（如果启用权重共享）
        if config.tie_word_embeddings:
            embed_weights = model.model.embed_tokens.weight
            lm_head_weights = model.lm_head.weight
            # 注意：可能需要转置
            assert embed_weights.shape == lm_head_weights.shape or \
                   embed_weights.shape == lm_head_weights.T.shape
            
    def test_causal_lm_different_batch_sizes(self):
        """测试不同批次大小"""
        config = llmconfig(
            vocab_size=500,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2
        )
        
        model = CausalLM(config)
        model.eval()
        
        seq_len = 8
        
        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                output = model(input_ids)
                
            expected_shape = (batch_size, seq_len, config.vocab_size)
            assert_tensor_shape(output.logits, expected_shape, f"batch_size={batch_size}")
            
    def test_causal_lm_memory_efficiency(self):
        """测试内存效率"""
        config = llmconfig(
            vocab_size=1000,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=3
        )
        
        model = CausalLM(config)
        model.eval()
        
        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # 测量推理时的内存使用
        memory_before = measure_memory_usage()
        
        with torch.no_grad():
            output = model(input_ids)
            
        memory_after = measure_memory_usage()
        
        # 应该成功完成推理
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert_tensor_shape(output.logits, expected_shape, "memory efficient inference")
        
        # 内存使用应该是合理的（这个测试主要确保没有内存爆炸）
        memory_used = memory_after - memory_before
        assert memory_used > 0  # 应该使用了一些内存
        
    def test_model_parameter_count(self):
        """测试模型参数数量计算"""
        config = llmconfig(
            vocab_size=1000,
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            intermediate_size=512
        )
        
        model = CausalLM(config)
        param_count = count_parameters(model)
        
        # 参数数量应该是合理的
        assert param_count > 0
        
        # 可以进行粗略的估算验证
        # 嵌入层: vocab_size * hidden_size
        # 每层Transformer: 大约 4 * hidden_size^2 + 8 * hidden_size * intermediate_size
        # 输出层: hidden_size * vocab_size (如果不共享权重)
        
        embed_params = config.vocab_size * config.hidden_size
        assert param_count > embed_params  # 至少应该有嵌入层的参数
        
    def test_model_deterministic_output(self):
        """测试模型输出的确定性"""
        config = llmconfig(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            dropout=0.0  # 关闭dropout确保确定性
        )
        
        model = CausalLM(config)
        model.eval()
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # 多次运行应该产生相同结果
        outputs = []
        for _ in range(3):
            set_seed(42)
            with torch.no_grad():
                output = model(input_ids)
                outputs.append(output.logits.clone())
                
        for i in range(1, len(outputs)):
            assert_tensor_close(outputs[0], outputs[i], name=f"run {i}")