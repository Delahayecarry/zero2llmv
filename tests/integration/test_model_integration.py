"""
集成测试 - 测试各组件的协同工作
"""
import pytest
import torch
import math
from models.llm import LLM, CausalLM
from configs.llmconfig import llmconfig
from tests.utils import assert_tensor_shape, assert_tensor_close, set_seed, ModelTester


class TestModelIntegration:
    """测试模型各组件集成"""
    
    def test_end_to_end_standard_model(self):
        """测试标准Transformer端到端推理"""
        config = llmconfig(
            vocab_size=1000,
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=6,
            intermediate_size=512,
            max_position_embeddings=256,
            use_moe=False,
            dropout=0.0
        )
        
        model = CausalLM(config)
        model.eval()
        
        # 创建测试输入
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
            
        # 检查输出
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert_tensor_shape(output.logits, expected_shape, "end-to-end standard model")
        
        # 检查输出是否合理（logits不应该是NaN或Inf）
        assert not torch.isnan(output.logits).any()
        assert not torch.isinf(output.logits).any()
        
    def test_end_to_end_moe_model(self):
        """测试MoE模型端到端推理"""
        config = llmconfig(
            vocab_size=500,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=4,
            use_moe=True,
            num_experts_per_token=2,
            n_routed_experts=8,
            n_shared_experts=1,
            aux_loss_alpha=0.01
        )
        
        model = CausalLM(config)
        model.eval()
        
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
            
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert_tensor_shape(output.logits, expected_shape, "end-to-end MoE model")
        
        # MoE模型应该正常工作
        assert not torch.isnan(output.logits).any()
        assert not torch.isinf(output.logits).any()
        
    def test_grouped_query_attention_integration(self):
        """测试分组查询注意力集成"""
        config = llmconfig(
            vocab_size=1000,
            hidden_size=256,
            num_attention_heads=16,
            num_key_value_heads=8,  # GQA: KV头数少于Query头数
            num_hidden_layers=4,
            intermediate_size=512
        )
        
        model = CausalLM(config)
        model.eval()
        
        batch_size = 2
        seq_len = 24
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
            
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert_tensor_shape(output.logits, expected_shape, "GQA integration")
        
    def test_kv_cache_consistency(self):
        """测试KV缓存的一致性"""
        config = llmconfig(
            vocab_size=500,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=3,
            max_position_embeddings=128
        )
        
        model = CausalLM(config)
        model.eval()
        
        batch_size = 1
        context_len = 8
        gen_len = 4
        
        # 完整序列的输入
        full_input = torch.randint(0, config.vocab_size, (batch_size, context_len + gen_len))
        
        with torch.no_grad():
            # 不使用缓存的完整推理
            full_output = model(full_input)
            
            # 使用缓存的分段推理
            context_input = full_input[:, :context_len]
            context_output, past_kv = model(context_input, use_cache=True)
            
            # 逐步生成
            cached_outputs = [context_output]
            for i in range(gen_len):
                next_token = full_input[:, context_len + i:context_len + i + 1]
                next_output, past_kv = model(
                    next_token, 
                    past_key_values=past_kv, 
                    use_cache=True
                )
                cached_outputs.append(next_output)
                
            # 拼接缓存输出
            cached_full_output = torch.cat([out.logits for out in cached_outputs], dim=1)
            
        # 检查一致性（应该非常接近，允许一些数值误差）
        assert_tensor_close(
            full_output.logits, 
            cached_full_output, 
            rtol=1e-4, 
            atol=1e-5,
            name="KV cache consistency"
        )
        
    def test_training_inference_mode_switch(self):
        """测试训练和推理模式切换"""
        config = llmconfig(
            vocab_size=500,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            dropout=0.1  # 启用dropout来测试模式切换
        )
        
        model = CausalLM(config)
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # 训练模式
        model.train()
        train_output = model(input_ids, labels=labels)
        
        # 应该有损失
        assert hasattr(train_output, 'loss')
        assert train_output.loss.requires_grad
        
        # 推理模式
        model.eval()
        with torch.no_grad():
            eval_output = model(input_ids)
            
        # 输出形状应该相同
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert_tensor_shape(train_output.logits, expected_shape, "training mode")
        assert_tensor_shape(eval_output.logits, expected_shape, "inference mode")
        
    def test_gradient_accumulation(self):
        """测试梯度累积"""
        config = llmconfig(
            vocab_size=500,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2
        )
        
        model = CausalLM(config)
        model.train()
        
        batch_size = 2
        seq_len = 8
        
        # 清零梯度
        model.zero_grad()
        
        total_loss = 0
        num_accumulation_steps = 3
        
        for step in range(num_accumulation_steps):
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            output = model(input_ids, labels=labels)
            loss = output.loss / num_accumulation_steps  # 平均损失
            loss.backward()
            
            total_loss += loss.item()
            
        # 检查梯度存在
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                
    def test_different_sequence_lengths(self):
        """测试不同序列长度的处理"""
        config = llmconfig(
            vocab_size=1000,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=256
        )
        
        model = CausalLM(config)
        model.eval()
        
        batch_size = 2
        
        for seq_len in [8, 16, 32, 64, 128]:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                output = model(input_ids)
                
            expected_shape = (batch_size, seq_len, config.vocab_size)
            assert_tensor_shape(output.logits, expected_shape, f"seq_len={seq_len}")
            
    def test_attention_mask_integration(self):
        """测试注意力掩码的端到端集成"""
        config = llmconfig(
            vocab_size=500,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=3
        )
        
        model = CausalLM(config)
        model.eval()
        
        batch_size = 3
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # 创建不同的掩码模式
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 8:] = 0    # 第一个序列：只看前8个token
        attention_mask[1, 12:] = 0   # 第二个序列：只看前12个token
        # 第三个序列：看全部token
        
        with torch.no_grad():
            masked_output = model(input_ids, attention_mask=attention_mask)
            unmasked_output = model(input_ids)
            
        # 第三个序列（完全未masked）应该与unmasked输出相同
        assert_tensor_close(
            masked_output.logits[2], 
            unmasked_output.logits[2], 
            rtol=1e-5,
            name="unmasked sequence consistency"
        )
        
    def test_model_configuration_variations(self):
        """测试不同配置的模型都能正常工作"""
        
        configurations = [
            # 小模型
            {
                "vocab_size": 1000,
                "hidden_size": 64,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "intermediate_size": 128
            },
            # 中等模型
            {
                "vocab_size": 5000,
                "hidden_size": 256,
                "num_attention_heads": 8,
                "num_hidden_layers": 6,
                "intermediate_size": 512
            },
            # GQA模型
            {
                "vocab_size": 2000,
                "hidden_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_hidden_layers": 4
            },
            # MoE模型
            {
                "vocab_size": 1000,
                "hidden_size": 128,
                "num_attention_heads": 4,
                "num_hidden_layers": 3,
                "use_moe": True,
                "num_experts_per_token": 2,
                "n_routed_experts": 8
            }
        ]
        
        for i, config_dict in enumerate(configurations):
            config = llmconfig(**config_dict)
            model = CausalLM(config)
            model.eval()
            
            batch_size = 1
            seq_len = 8
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                output = model(input_ids)
                
            expected_shape = (batch_size, seq_len, config.vocab_size)
            assert_tensor_shape(output.logits, expected_shape, f"config_{i}")
            assert not torch.isnan(output.logits).any(), f"NaN in config_{i}"
            assert not torch.isinf(output.logits).any(), f"Inf in config_{i}"


class TestModelCompatibility:
    """测试模型兼容性"""
    
    def test_huggingface_compatibility(self):
        """测试与HuggingFace格式的兼容性"""
        config = llmconfig(
            vocab_size=1000,
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4
        )
        
        model = CausalLM(config)
        
        # 检查模型是否有必要的属性
        assert hasattr(model, 'model')
        assert hasattr(model, 'lm_head')
        assert hasattr(model.config, 'vocab_size')
        assert hasattr(model.config, 'hidden_size')
        
    def test_state_dict_save_load(self):
        """测试模型状态字典的保存和加载"""
        config = llmconfig(
            vocab_size=500,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=2
        )
        
        # 创建原始模型
        original_model = CausalLM(config)
        original_model.eval()
        
        # 保存状态字典
        state_dict = original_model.state_dict()
        
        # 创建新模型并加载状态字典
        new_model = CausalLM(config)
        new_model.load_state_dict(state_dict)
        new_model.eval()
        
        # 测试两个模型输出是否相同
        batch_size = 1
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            original_output = original_model(input_ids)
            new_output = new_model(input_ids)
            
        assert_tensor_close(
            original_output.logits, 
            new_output.logits, 
            name="state dict consistency"
        )
        
    def test_model_device_transfer(self):
        """测试模型设备转换"""
        config = llmconfig(
            vocab_size=500,
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2
        )
        
        model = CausalLM(config)
        
        # 测试CPU
        model = model.to('cpu')
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        
        with torch.no_grad():
            cpu_output = model(input_ids)
            
        assert cpu_output.logits.device.type == 'cpu'
        
        # 如果有CUDA，测试GPU
        if torch.cuda.is_available():
            model = model.to('cuda')
            input_ids = input_ids.to('cuda')
            
            with torch.no_grad():
                gpu_output = model(input_ids)
                
            assert gpu_output.logits.device.type == 'cuda'
            
    def test_model_dtype_conversion(self):
        """测试模型数据类型转换"""
        config = llmconfig(
            vocab_size=500,
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2
        )
        
        model = CausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        
        # 测试float32
        model = model.float()
        with torch.no_grad():
            fp32_output = model(input_ids)
        assert fp32_output.logits.dtype == torch.float32
        
        # 测试float16（如果支持）
        if torch.cuda.is_available():
            model = model.half().cuda()
            input_ids = input_ids.cuda()
            with torch.no_grad():
                fp16_output = model(input_ids)
            assert fp16_output.logits.dtype == torch.float16