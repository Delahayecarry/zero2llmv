"""
测试夹具 - 提供各种测试配置和样本数据
"""
import torch
import pytest
from configs.llmconfig import llmconfig


class ModelFixtures:
    """模型测试夹具"""
    
    @staticmethod
    def tiny_model_config():
        """超小模型配置，用于快速测试"""
        return llmconfig(
            vocab_size=100,
            hidden_size=32,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=64,
            max_position_embeddings=128,
            dropout=0.0
        )
    
    @staticmethod
    def small_model_config():
        """小模型配置"""
        return llmconfig(
            vocab_size=1000,
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=3,
            intermediate_size=256,
            max_position_embeddings=256,
            dropout=0.1
        )
        
    @staticmethod
    def medium_model_config():
        """中等模型配置"""
        return llmconfig(
            vocab_size=5000,
            hidden_size=512,
            num_attention_heads=8,
            num_hidden_layers=6,
            intermediate_size=1024,
            max_position_embeddings=512,
            dropout=0.1
        )
        
    @staticmethod
    def gqa_model_config():
        """分组查询注意力模型配置"""
        return llmconfig(
            vocab_size=2000,
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,  # GQA
            num_hidden_layers=4,
            intermediate_size=512,
            max_position_embeddings=512
        )
        
    @staticmethod
    def moe_model_config():
        """MoE模型配置"""
        return llmconfig(
            vocab_size=2000,
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            use_moe=True,
            num_experts_per_token=2,
            n_routed_experts=8,
            n_shared_experts=1,
            aux_loss_alpha=0.01,
            intermediate_size=512
        )
        
    @staticmethod
    def large_moe_config():
        """大型MoE模型配置"""
        return llmconfig(
            vocab_size=10000,
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=6,
            num_hidden_layers=8,
            use_moe=True,
            num_experts_per_token=2,
            n_routed_experts=16,
            n_shared_experts=2,
            aux_loss_alpha=0.01,
            intermediate_size=2048,
            max_position_embeddings=1024
        )


class DataFixtures:
    """数据测试夹具"""
    
    @staticmethod
    def create_random_input(vocab_size, batch_size, seq_len, device='cpu'):
        """创建随机输入数据"""
        return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
    @staticmethod
    def create_sequential_input(vocab_size, batch_size, seq_len, device='cpu'):
        """创建序列化输入数据（用于可重现测试）"""
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        for b in range(batch_size):
            for s in range(seq_len):
                input_ids[b, s] = (b * seq_len + s) % vocab_size
        return input_ids
        
    @staticmethod
    def create_attention_mask(batch_size, seq_len, mask_ratio=0.2, device='cpu'):
        """创建注意力掩码"""
        mask = torch.ones(batch_size, seq_len, device=device)
        # 随机mask掉一些token
        num_masked = int(seq_len * mask_ratio)
        for b in range(batch_size):
            mask_indices = torch.randperm(seq_len)[:num_masked]
            mask[b, mask_indices] = 0
        return mask
        
    @staticmethod
    def create_causal_mask(seq_len, device='cpu'):
        """创建因果掩码（下三角矩阵）"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
        
    @staticmethod
    def create_padding_mask(batch_size, seq_len, pad_ratio=0.3, device='cpu'):
        """创建填充掩码"""
        mask = torch.ones(batch_size, seq_len, device=device)
        for b in range(batch_size):
            # 随机决定填充长度
            pad_len = int(torch.randint(0, int(seq_len * pad_ratio) + 1, (1,)).item())
            if pad_len > 0:
                # 从末尾开始填充
                mask[b, -pad_len:] = 0
        return mask


class BenchmarkFixtures:
    """性能测试夹具"""
    
    @staticmethod
    def get_benchmark_configs():
        """获取性能测试配置"""
        return {
            'tiny': ModelFixtures.tiny_model_config(),
            'small': ModelFixtures.small_model_config(),
            'medium': ModelFixtures.medium_model_config(),
            'gqa': ModelFixtures.gqa_model_config(),
            'moe': ModelFixtures.moe_model_config(),
        }
        
    @staticmethod
    def get_benchmark_inputs():
        """获取性能测试输入"""
        return {
            'tiny': {
                'batch_sizes': [1, 2, 4],
                'seq_lens': [8, 16, 32],
                'vocab_size': 100
            },
            'small': {
                'batch_sizes': [1, 2, 4, 8],
                'seq_lens': [16, 32, 64, 128],
                'vocab_size': 1000
            },
            'medium': {
                'batch_sizes': [1, 2, 4],
                'seq_lens': [32, 64, 128, 256],
                'vocab_size': 5000
            }
        }


# pytest fixtures
@pytest.fixture
def tiny_config():
    """Pytest fixture for tiny model config"""
    return ModelFixtures.tiny_model_config()


@pytest.fixture
def small_config():
    """Pytest fixture for small model config"""
    return ModelFixtures.small_model_config()


@pytest.fixture
def medium_config():
    """Pytest fixture for medium model config"""
    return ModelFixtures.medium_model_config()


@pytest.fixture
def gqa_config():
    """Pytest fixture for GQA model config"""
    return ModelFixtures.gqa_model_config()


@pytest.fixture
def moe_config():
    """Pytest fixture for MoE model config"""
    return ModelFixtures.moe_model_config()


@pytest.fixture
def random_input():
    """Pytest fixture for random input data"""
    def _create_input(vocab_size=1000, batch_size=2, seq_len=16, device='cpu'):
        return DataFixtures.create_random_input(vocab_size, batch_size, seq_len, device)
    return _create_input


@pytest.fixture
def sequential_input():
    """Pytest fixture for sequential input data"""
    def _create_input(vocab_size=1000, batch_size=2, seq_len=16, device='cpu'):
        return DataFixtures.create_sequential_input(vocab_size, batch_size, seq_len, device)
    return _create_input


@pytest.fixture
def attention_mask():
    """Pytest fixture for attention mask"""
    def _create_mask(batch_size=2, seq_len=16, mask_ratio=0.2, device='cpu'):
        return DataFixtures.create_attention_mask(batch_size, seq_len, mask_ratio, device)
    return _create_mask


# 设备相关的fixture
@pytest.fixture
def device():
    """自动选择设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def cpu_device():
    """CPU设备"""
    return torch.device('cpu')


@pytest.fixture
def cuda_device():
    """CUDA设备（如果可用）"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        pytest.skip("CUDA not available")


# 数据类型相关的fixture
@pytest.fixture(params=[torch.float32, torch.float16])
def dtype(request):
    """参数化的数据类型fixture"""
    return request.param


@pytest.fixture
def fp32_dtype():
    """Float32数据类型"""
    return torch.float32


@pytest.fixture
def fp16_dtype():
    """Float16数据类型"""
    return torch.float16