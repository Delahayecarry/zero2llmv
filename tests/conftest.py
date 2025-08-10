"""
pytest 配置文件和共享fixtures
"""
import pytest
import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from configs.llmconfig import llmconfig


@pytest.fixture(scope="session")
def device():
    """测试设备fixture"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session") 
def small_config():
    """小型模型配置，用于快速测试"""
    return llmconfig(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA测试
        num_hidden_layers=2,
        vocab_size=1000,
        max_position_embeddings=128,
        intermediate_size=256
    )


@pytest.fixture(scope="session")
def moe_config():
    """MoE模型配置"""
    return llmconfig(
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        vocab_size=1000,
        max_position_embeddings=128,
        use_moe=True,
        num_experts_per_token=2,
        n_routed_experts=4,
        n_shared_experts=1,
        aux_loss_alpha=0.01
    )


@pytest.fixture(scope="session")
def standard_config():
    """标准配置，用于更全面的测试"""
    return llmconfig(
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=4,
        vocab_size=5000,
        max_position_embeddings=512
    )


@pytest.fixture
def sample_input_ids(small_config):
    """样本输入数据"""
    batch_size = 2
    seq_len = 16
    return torch.randint(0, small_config.vocab_size, (batch_size, seq_len))


@pytest.fixture
def sample_attention_mask():
    """样本注意力掩码"""
    return torch.ones(2, 16, dtype=torch.float)


# pytest 配置
def pytest_configure(config):
    """pytest 配置钩子"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试项集合"""
    if config.getoption("--runxfail"):
        # --runxfail given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)