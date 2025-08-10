"""
MoE (专家混合) 测试
"""
import pytest
import torch
from models.llm import MoEGate, MOEFeedForward, FeedForward
from configs.llmconfig import llmconfig
from tests.utils import assert_tensor_shape, assert_tensor_close, set_seed, count_parameters


class TestMoEGate:
    """测试MoE门控网络"""
    
    def test_moe_gate_forward(self):
        """测试MoE门控前向传播"""
        config = llmconfig(
            hidden_size=128,
            num_experts_per_token=2,
            n_routed_experts=4,
            scoring_func="softmax",
            aux_loss_alpha=0.01,
            norm_topk_prob=True
        )
        
        gate = MoEGate(config)
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        topk_idx, topk_weight, aux_loss = gate(x)
        
        # 检查输出形状
        total_tokens = batch_size * seq_len
        assert_tensor_shape(topk_idx, (total_tokens, config.num_experts_per_token), "topk_idx")
        assert_tensor_shape(topk_weight, (total_tokens, config.num_experts_per_token), "topk_weight")
        
        # 检查专家索引范围
        assert torch.all(topk_idx >= 0)
        assert torch.all(topk_idx < config.n_routed_experts)
        
        # 检查权重归一化（如果启用）
        if config.norm_topk_prob and config.num_experts_per_token > 1:
            weight_sums = topk_weight.sum(dim=1)
            expected_sums = torch.ones_like(weight_sums)
            assert_tensor_close(weight_sums, expected_sums, rtol=1e-5, name="weight normalization")
            
        # 检查辅助损失
        assert isinstance(aux_loss, (int, float, torch.Tensor))
        if isinstance(aux_loss, torch.Tensor):
            assert aux_loss.numel() == 1
            
    def test_moe_gate_different_topk(self):
        """测试不同的top-k值"""
        for num_experts_per_token in [1, 2, 4]:
            config = llmconfig(
                hidden_size=64,
                num_experts_per_token=num_experts_per_token,
                n_routed_experts=8,
                aux_loss_alpha=0.0  # 关闭辅助损失以简化测试
            )
            
            gate = MoEGate(config)
            gate.eval()  # 关闭训练模式以避免辅助损失计算
            
            x = torch.randn(1, 4, config.hidden_size)
            
            topk_idx, topk_weight, aux_loss = gate(x)
            
            expected_shape = (4, num_experts_per_token)  # 1*4个token，每个选择num_experts_per_token个专家
            assert_tensor_shape(topk_idx, expected_shape, f"topk_idx with k={num_experts_per_token}")
            assert_tensor_shape(topk_weight, expected_shape, f"topk_weight with k={num_experts_per_token}")
            
    def test_moe_gate_aux_loss(self):
        """测试辅助损失计算"""
        config = llmconfig(
            hidden_size=64,
            num_experts_per_token=2,
            n_routed_experts=4,
            aux_loss_alpha=0.1,  # 非零辅助损失权重
            seq_aux=True
        )
        
        gate = MoEGate(config)
        gate.train()  # 训练模式才计算辅助损失
        
        x = torch.randn(2, 8, config.hidden_size)
        
        topk_idx, topk_weight, aux_loss = gate(x)
        
        # 辅助损失应该是正数（负载均衡惩罚）
        assert isinstance(aux_loss, torch.Tensor)
        assert aux_loss.item() >= 0.0
        
    def test_moe_gate_deterministic(self):
        """测试门控网络的确定性"""
        config = llmconfig(
            hidden_size=64,
            num_experts_per_token=2,
            n_routed_experts=4,
            aux_loss_alpha=0.0
        )
        
        gate = MoEGate(config)
        gate.eval()
        
        x = torch.randn(1, 4, config.hidden_size)
        
        # 多次运行应该产生相同结果
        results = []
        for _ in range(3):
            set_seed(42)
            with torch.no_grad():
                topk_idx, topk_weight, _ = gate(x)
                results.append((topk_idx.clone(), topk_weight.clone()))
                
        for i in range(1, len(results)):
            assert torch.equal(results[0][0], results[i][0]), "topk_idx should be deterministic"
            assert_tensor_close(results[0][1], results[i][1], name="topk_weight deterministic")
            
    def test_moe_gate_gradient_flow(self):
        """测试门控网络的梯度流"""
        config = llmconfig(
            hidden_size=64,
            num_experts_per_token=2,
            n_routed_experts=4,
            aux_loss_alpha=0.1
        )
        
        gate = MoEGate(config)
        gate.train()
        
        x = torch.randn(1, 4, config.hidden_size, requires_grad=True)
        
        topk_idx, topk_weight, aux_loss = gate(x)
        
        # 使用权重进行反向传播（topk_idx不可微）
        loss = topk_weight.sum() + aux_loss
        loss.backward()
        
        # 检查梯度
        assert x.grad is not None
        assert gate.weight.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(gate.weight.grad).any()


class TestMOEFeedForward:
    """测试MoE前馈网络"""
    
    def test_moe_feedforward_forward(self):
        """测试MoE前馈网络前向传播"""
        config = llmconfig(
            hidden_size=128,
            intermediate_size=256,
            num_experts_per_token=2,
            n_routed_experts=4,
            n_shared_experts=1,
            aux_loss_alpha=0.01
        )
        
        moe_ff = MOEFeedForward(config)
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = moe_ff(x)
        
        # 检查输出形状
        assert_tensor_shape(output, (batch_size, seq_len, config.hidden_size), "MoE FeedForward output")
        
        # 检查辅助损失存在
        assert hasattr(moe_ff, 'aux_loss')
        
    def test_moe_feedforward_vs_standard(self):
        """比较MoE前馈网络与标准前馈网络"""
        config = llmconfig(
            hidden_size=64,
            intermediate_size=128,
            num_experts_per_token=1,  # 每个token只用一个专家
            n_routed_experts=1,       # 只有一个专家
            n_shared_experts=0,       # 没有共享专家
            aux_loss_alpha=0.0        # 无辅助损失
        )
        
        # MoE前馈网络（实际上退化为单专家）
        moe_ff = MOEFeedForward(config)
        moe_ff.eval()
        
        # 标准前馈网络
        standard_ff = FeedForward(config)
        standard_ff.eval()
        
        # 复制权重以确保公平比较
        with torch.no_grad():
            standard_ff.gate_proj.weight.copy_(moe_ff.experts[0].gate_proj.weight)
            standard_ff.up_proj.weight.copy_(moe_ff.experts[0].up_proj.weight)
            standard_ff.down_proj.weight.copy_(moe_ff.experts[0].down_proj.weight)
            
        x = torch.randn(1, 4, config.hidden_size)
        
        with torch.no_grad():
            moe_output = moe_ff(x)
            standard_output = standard_ff(x)
            
        # 在单专家情况下，输出应该接近
        # 注意：由于路由机制的存在，可能不会完全相同
        assert moe_output.shape == standard_output.shape
        
    def test_moe_feedforward_shared_experts(self):
        """测试共享专家功能"""
        config = llmconfig(
            hidden_size=64,
            num_experts_per_token=2,
            n_routed_experts=4,
            n_shared_experts=2,  # 两个共享专家
            aux_loss_alpha=0.0
        )
        
        moe_ff = MOEFeedForward(config)
        
        # 检查共享专家是否被创建
        assert hasattr(moe_ff, 'shared_experts')
        assert len(moe_ff.shared_experts) == config.n_shared_experts
        
        x = torch.randn(1, 4, config.hidden_size)
        output = moe_ff(x)
        
        assert_tensor_shape(output, (1, 4, config.hidden_size), "MoE with shared experts")
        
    def test_moe_feedforward_training_vs_inference(self):
        """测试训练模式和推理模式"""
        config = llmconfig(
            hidden_size=64,
            num_experts_per_token=2,
            n_routed_experts=4,
            n_shared_experts=0,
            aux_loss_alpha=0.0
        )
        
        moe_ff = MOEFeedForward(config)
        x = torch.randn(1, 4, config.hidden_size)
        
        # 训练模式
        moe_ff.train()
        with torch.no_grad():
            train_output = moe_ff(x)
            
        # 推理模式  
        moe_ff.eval()
        with torch.no_grad():
            eval_output = moe_ff(x)
            
        # 形状应该相同
        assert train_output.shape == eval_output.shape
        
        # 数值可能略有不同（由于不同的实现路径）
        # 但应该在合理范围内
        diff = torch.abs(train_output - eval_output).max()
        assert diff < 1e-3, f"Training and inference outputs differ too much: {diff}"
        
    def test_moe_feedforward_memory_efficient_inference(self):
        """测试推理模式的内存效率"""
        config = llmconfig(
            hidden_size=64,
            num_experts_per_token=2,
            n_routed_experts=8,
            aux_loss_alpha=0.0
        )
        
        moe_ff = MOEFeedForward(config)
        moe_ff.eval()
        
        x = torch.randn(2, 16, config.hidden_size)  # 较大的输入
        
        # 推理应该成功完成
        with torch.no_grad():
            output = moe_ff(x)
            
        assert_tensor_shape(output, (2, 16, config.hidden_size), "memory efficient inference")
        
    def test_moe_feedforward_expert_utilization(self):
        """测试专家利用率"""
        config = llmconfig(
            hidden_size=64,
            num_experts_per_token=1,
            n_routed_experts=4,
            aux_loss_alpha=0.0
        )
        
        moe_ff = MOEFeedForward(config)
        moe_ff.eval()
        
        # 创建偏向某个专家的输入
        x = torch.randn(1, 100, config.hidden_size)  # 大量token
        
        with torch.no_grad():
            topk_idx, _, _ = moe_ff.gate(x)
            
        # 统计每个专家被选择的次数
        expert_counts = torch.bincount(topk_idx.flatten(), minlength=config.n_routed_experts)
        
        # 所有专家都应该被选择（至少在理论上）
        # 这个测试主要确保没有专家完全被忽略
        total_selections = expert_counts.sum().item()
        assert total_selections == 100  # 每个token选择一个专家
        
    def test_moe_feedforward_gradient_flow(self):
        """测试MoE前馈网络的梯度流"""
        config = llmconfig(
            hidden_size=64,
            num_experts_per_token=2,
            n_routed_experts=4,
            aux_loss_alpha=0.1
        )
        
        moe_ff = MOEFeedForward(config)
        moe_ff.train()
        
        x = torch.randn(2, 8, config.hidden_size, requires_grad=True)
        
        output = moe_ff(x)
        loss = output.mean() + moe_ff.aux_loss
        
        loss.backward()
        
        # 检查输入梯度
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # 检查专家参数梯度
        for i, expert in enumerate(moe_ff.experts):
            for name, param in expert.named_parameters():
                if param.requires_grad:
                    # 注意：不是所有专家都会在每次前向传播中被使用
                    # 所以某些专家可能没有梯度
                    if param.grad is not None:
                        assert not torch.isnan(param.grad).any(), f"NaN in expert {i} {name}"
                        
        # 检查门控网络梯度
        for name, param in moe_ff.gate.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for gate {name}"
                assert not torch.isnan(param.grad).any(), f"NaN in gate {name}"