"""
配置类测试
"""
import pytest
import torch
from configs.llmconfig import llmconfig


class TestLLMConfig:
    """测试LLM配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = llmconfig()
        
        # 检查基本参数
        assert config.hidden_size == 768
        assert config.num_attention_heads == 12
        assert config.num_key_value_heads == 12  # 默认应该等于attention_heads
        assert config.vocab_size == 30000
        assert config.num_hidden_layers == 12
        assert config.max_position_embeddings == 512
        
        # 检查MoE参数
        assert config.use_moe == False
        assert config.num_experts_per_token == 1
        assert config.n_routed_experts == 2
        
        # 检查别名
        assert config.hidden_dim == config.hidden_size
        
    def test_custom_config(self):
        """测试自定义配置"""
        config = llmconfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=5000,
            num_hidden_layers=6,
            use_moe=True,
            num_experts_per_token=2,
            n_routed_experts=8
        )
        
        assert config.hidden_size == 256
        assert config.num_attention_heads == 8
        assert config.num_key_value_heads == 4
        assert config.vocab_size == 5000
        assert config.num_hidden_layers == 6
        assert config.use_moe == True
        assert config.num_experts_per_token == 2
        assert config.n_routed_experts == 8
        
    def test_kv_heads_default(self):
        """测试KV头数的默认行为"""
        # 当num_key_value_heads为None时，应该等于num_attention_heads
        config = llmconfig(
            num_attention_heads=16,
            num_key_value_heads=None
        )
        assert config.num_key_value_heads == 16
        
    def test_gqa_config(self):
        """测试分组查询注意力配置"""
        config = llmconfig(
            num_attention_heads=16,
            num_key_value_heads=8  # GQA: KV头数少于Query头数
        )
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 8
        
    def test_moe_config(self):
        """测试MoE配置"""
        config = llmconfig(
            use_moe=True,
            num_experts_per_token=2,
            n_routed_experts=8,
            n_shared_experts=2,
            aux_loss_alpha=0.01,
            seq_aux=True,
            norm_topk_prob=True
        )
        
        assert config.use_moe == True
        assert config.num_experts_per_token == 2
        assert config.n_routed_experts == 8
        assert config.n_shared_experts == 2
        assert config.aux_loss_alpha == 0.01
        assert config.seq_aux == True
        assert config.norm_topk_prob == True
        
    def test_intermediate_size_calculation(self):
        """测试中间层维度的自动计算"""
        config = llmconfig(
            hidden_size=768,
            intermediate_size=None  # 应该自动计算
        )
        # 应该在FeedForward初始化时自动设置
        assert config.intermediate_size is None
        
    def test_parameter_validation(self):
        """测试参数验证"""
        # 测试正常的配置
        config = llmconfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4
        )
        
        # 验证头维度计算
        expected_head_dim = 256 // 8
        assert expected_head_dim == 32
        
    def test_rope_config(self):
        """测试RoPE配置"""
        config = llmconfig(
            rope_theta=10000.0,
            max_position_embeddings=2048
        )
        
        assert config.rope_theta == 10000.0
        assert config.max_position_embeddings == 2048
        assert config.max_positions_embeddings == 2048  # 检查别名
        
    def test_activation_function(self):
        """测试激活函数配置"""
        config = llmconfig(hidden_act="swish")
        assert config.hidden_act == "swish"
        
        config = llmconfig(hidden_act="gelu")
        assert config.hidden_act == "gelu"
        
    def test_dropout_config(self):
        """测试dropout配置"""
        config = llmconfig(dropout=0.15)
        assert config.dropout == 0.15
        
    def test_flash_attention_config(self):
        """测试Flash Attention配置"""
        config = llmconfig(flash_attn=False)
        assert config.flash_attn == False
        
        config = llmconfig(flash_attn=True) 
        assert config.flash_attn == True
        
    def test_scoring_function_mapping(self):
        """测试评分函数名称映射"""
        config = llmconfig(scoring_function="softmax")
        # 应该映射到scoring_func
        assert config.scoring_func == "softmax"
        
    def test_config_serialization(self):
        """测试配置的序列化和反序列化"""
        original_config = llmconfig(
            hidden_size=256,
            num_attention_heads=8,
            use_moe=True,
            num_experts_per_token=2
        )
        
        # 转换为字典
        config_dict = original_config.to_dict()
        
        # 从字典重建配置
        restored_config = llmconfig.from_dict(config_dict)
        
        # 检查主要参数是否相同
        assert restored_config.hidden_size == original_config.hidden_size
        assert restored_config.num_attention_heads == original_config.num_attention_heads
        assert restored_config.use_moe == original_config.use_moe
        assert restored_config.num_experts_per_token == original_config.num_experts_per_token
        
    def test_model_type(self):
        """测试模型类型"""
        config = llmconfig()
        assert config.model_type == "zero2llm-v"
        
    def test_token_ids(self):
        """测试特殊token ID"""
        config = llmconfig()
        assert config.bos_token_id == 0
        assert config.eos_token_id == 1
        
        # 测试自定义token ID
        config = llmconfig(bos_token_id=100, eos_token_id=101)
        assert config.bos_token_id == 100
        assert config.eos_token_id == 101