"""
性能测试 - 测试模型的性能特性
"""
import pytest
import torch
import time
import gc
from contextlib import contextmanager
from models.llm import LLM, CausalLM, Attention, FeedForward, MOEFeedForward
from configs.llmconfig import llmconfig
from tests.utils import (
    assert_tensor_shape, set_seed, ModelTester,
    measure_memory_usage, count_parameters
)


@contextmanager
def timer():
    """计时上下文管理器"""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    

def clear_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class TestInferencePerformance:
    """测试推理性能"""
    
    def test_forward_pass_speed(self):
        """测试前向传播速度"""
        config = llmconfig(
            vocab_size=10000,
            hidden_size=512,
            num_attention_heads=8,
            num_hidden_layers=6,
            intermediate_size=1024,
            dropout=0.0
        )
        
        model = CausalLM(config)
        model.eval()
        
        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
                
        # 计时多次运行
        num_runs = 10
        total_time = 0
        
        with torch.no_grad():
            for _ in range(num_runs):
                with timer() as get_time:
                    output = model(input_ids)
                total_time += get_time()
                
        avg_time = total_time / num_runs
        tokens_per_second = (batch_size * seq_len) / avg_time
        
        print(f"平均推理时间: {avg_time:.4f}s")
        print(f"处理速度: {tokens_per_second:.2f} tokens/s")
        
        # 性能基准：应该能够在合理时间内完成
        assert avg_time < 5.0  # 5秒内完成一次前向传播
        assert tokens_per_second > 50  # 至少50 tokens/s
        
    def test_kv_cache_speedup(self):
        """测试KV缓存的加速效果"""
        config = llmconfig(
            vocab_size=5000,
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            max_position_embeddings=512
        )
        
        model = CausalLM(config)
        model.eval()
        
        batch_size = 1
        context_len = 64
        gen_len = 32
        
        # 完整输入
        full_input = torch.randint(0, config.vocab_size, (batch_size, context_len + gen_len))
        
        # 测试不使用缓存的性能
        with torch.no_grad():
            with timer() as get_time:
                for i in range(gen_len):
                    current_input = full_input[:, :context_len + i + 1]
                    _ = model(current_input)
            time_without_cache = get_time()
            
        # 测试使用缓存的性能
        with torch.no_grad():
            with timer() as get_time:
                # 编码阶段
                context_input = full_input[:, :context_len]
                _, past_kv = model(context_input, use_cache=True)
                
                # 生成阶段
                for i in range(gen_len):
                    next_token = full_input[:, context_len + i:context_len + i + 1]
                    _, past_kv = model(next_token, past_key_values=past_kv, use_cache=True)
            time_with_cache = get_time()
            
        speedup = time_without_cache / time_with_cache
        
        print(f"无缓存时间: {time_without_cache:.4f}s")
        print(f"有缓存时间: {time_with_cache:.4f}s")  
        print(f"加速比: {speedup:.2f}x")
        
        # KV缓存应该显著提升速度
        assert speedup > 1.5  # 至少1.5x加速
        
    def test_batch_processing_efficiency(self):
        """测试批处理效率"""
        config = llmconfig(
            vocab_size=5000,
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4
        )
        
        model = CausalLM(config)
        model.eval()
        
        seq_len = 64
        
        # 测试不同批次大小的处理时间
        batch_sizes = [1, 2, 4, 8]
        times = []
        
        for batch_size in batch_sizes:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            # 预热
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_ids)
                    
            # 计时
            with torch.no_grad():
                with timer() as get_time:
                    for _ in range(5):
                        _ = model(input_ids)
                avg_time = get_time() / 5
                times.append(avg_time)
                
        # 计算每个样本的平均处理时间
        time_per_sample = [t / bs for t, bs in zip(times, batch_sizes)]
        
        print("批次大小 -> 每样本时间:")
        for bs, tps in zip(batch_sizes, time_per_sample):
            print(f"  {bs} -> {tps:.4f}s")
            
        # 较大批次应该有更好的每样本效率
        assert time_per_sample[-1] < time_per_sample[0]  # 批次8应该比批次1更高效
        
    def test_sequence_length_scaling(self):
        """测试序列长度缩放性能"""
        config = llmconfig(
            vocab_size=2000,
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=3,
            max_position_embeddings=1024
        )
        
        model = CausalLM(config)
        model.eval()
        
        batch_size = 2
        seq_lengths = [32, 64, 128, 256, 512]
        times = []
        
        for seq_len in seq_lengths:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            # 预热
            with torch.no_grad():
                for _ in range(2):
                    _ = model(input_ids)
                    
            # 计时
            with torch.no_grad():
                with timer() as get_time:
                    for _ in range(3):
                        _ = model(input_ids)
                avg_time = get_time() / 3
                times.append(avg_time)
                
        print("序列长度 -> 时间:")
        for sl, t in zip(seq_lengths, times):
            print(f"  {sl} -> {t:.4f}s")
            
        # 验证时间复杂度不会太差
        # 注意力机制的复杂度是O(n²)，所以时间增长应该是可控的
        time_ratio = times[-1] / times[0]  # 512 vs 32
        length_ratio = seq_lengths[-1] / seq_lengths[0]  # 16x
        
        # 时间增长不应该超过长度平方比的2倍
        assert time_ratio < (length_ratio ** 2) * 2
        
    def test_moe_vs_standard_performance(self):
        """测试MoE与标准模型的性能对比"""
        base_config = {
            "vocab_size": 2000,
            "hidden_size": 256,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "intermediate_size": 512
        }
        
        # 标准模型
        standard_config = llmconfig(**base_config, use_moe=False)
        standard_model = CausalLM(standard_config)
        standard_model.eval()
        
        # MoE模型
        moe_config = llmconfig(
            **base_config,
            use_moe=True,
            num_experts_per_token=2,
            n_routed_experts=8
        )
        moe_model = CausalLM(moe_config)
        moe_model.eval()
        
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, base_config["vocab_size"], (batch_size, seq_len))
        
        # 测试标准模型
        with torch.no_grad():
            # 预热
            for _ in range(3):
                _ = standard_model(input_ids)
                
            with timer() as get_time:
                for _ in range(5):
                    _ = standard_model(input_ids)
            standard_time = get_time() / 5
            
        # 测试MoE模型
        with torch.no_grad():
            # 预热  
            for _ in range(3):
                _ = moe_model(input_ids)
                
            with timer() as get_time:
                for _ in range(5):
                    _ = moe_model(input_ids)
            moe_time = get_time() / 5
            
        print(f"标准模型时间: {standard_time:.4f}s")
        print(f"MoE模型时间: {moe_time:.4f}s")
        print(f"MoE/标准比例: {moe_time/standard_time:.2f}")
        
        # MoE模型可能稍慢，但不应该慢太多
        assert moe_time / standard_time < 3.0  # 不超过3倍


class TestMemoryPerformance:
    """测试内存性能"""
    
    def test_memory_usage_scaling(self):
        """测试内存使用的缩放性"""
        configs = [
            # 小模型
            {"hidden_size": 128, "num_hidden_layers": 2, "vocab_size": 1000},
            # 中模型  
            {"hidden_size": 256, "num_hidden_layers": 4, "vocab_size": 2000},
            # 大模型
            {"hidden_size": 512, "num_hidden_layers": 6, "vocab_size": 5000},
        ]
        
        memory_usage = []
        
        for i, config_dict in enumerate(configs):
            clear_memory()
            
            config = llmconfig(
                num_attention_heads=config_dict["hidden_size"] // 64,
                **config_dict
            )
            
            memory_before = measure_memory_usage()
            model = CausalLM(config)
            memory_after = measure_memory_usage()
            
            model_memory = memory_after - memory_before
            memory_usage.append(model_memory)
            
            print(f"配置 {i}: {model_memory:.2f}MB")
            
            # 清理
            del model
            clear_memory()
            
        # 内存使用应该随模型大小合理增长
        assert all(m > 0 for m in memory_usage)
        assert memory_usage[1] > memory_usage[0]  # 中模型 > 小模型
        assert memory_usage[2] > memory_usage[1]  # 大模型 > 中模型
        
    def test_gradient_memory_overhead(self):
        """测试梯度的内存开销"""
        config = llmconfig(
            vocab_size=2000,
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4
        )
        
        clear_memory()
        memory_baseline = measure_memory_usage()
        
        # 创建模型
        model = CausalLM(config)
        memory_model = measure_memory_usage()
        
        # 创建输入并进行前向传播
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        model.train()
        output = model(input_ids, labels=labels)
        memory_forward = measure_memory_usage()
        
        # 反向传播
        output.loss.backward()
        memory_backward = measure_memory_usage()
        
        model_memory = memory_model - memory_baseline
        forward_memory = memory_forward - memory_model
        gradient_memory = memory_backward - memory_forward
        
        print(f"模型内存: {model_memory:.2f}MB")
        print(f"前向传播额外内存: {forward_memory:.2f}MB")
        print(f"梯度内存: {gradient_memory:.2f}MB")
        
        # 梯度内存不应该超过模型参数内存的太多倍
        if gradient_memory > 0:  # 如果有梯度内存使用
            gradient_ratio = gradient_memory / model_memory
            assert gradient_ratio < 3.0  # 梯度内存不超过模型内存的3倍
            
    def test_kv_cache_memory_usage(self):
        """测试KV缓存的内存使用"""
        config = llmconfig(
            vocab_size=1000,
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            max_position_embeddings=512
        )
        
        model = CausalLM(config)
        model.eval()
        
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        clear_memory()
        memory_before = measure_memory_usage()
        
        with torch.no_grad():
            # 不使用缓存
            _ = model(input_ids)
            memory_no_cache = measure_memory_usage()
            
            # 使用缓存
            _, past_kv = model(input_ids, use_cache=True)
            memory_with_cache = measure_memory_usage()
            
        no_cache_memory = memory_no_cache - memory_before
        cache_memory = memory_with_cache - memory_before
        cache_overhead = cache_memory - no_cache_memory
        
        print(f"无缓存内存: {no_cache_memory:.2f}MB")
        print(f"有缓存内存: {cache_memory:.2f}MB")
        print(f"缓存开销: {cache_overhead:.2f}MB")
        
        # 缓存会增加内存使用，但应该在合理范围内
        assert cache_overhead >= 0  # 缓存应该增加内存使用
        assert cache_overhead < no_cache_memory  # 缓存开销不应超过基础内存


class TestParameterEfficiency:
    """测试参数效率"""
    
    def test_parameter_count_scaling(self):
        """测试参数数量的缩放"""
        configs = [
            {"hidden_size": 128, "num_hidden_layers": 2},
            {"hidden_size": 256, "num_hidden_layers": 4}, 
            {"hidden_size": 512, "num_hidden_layers": 6},
        ]
        
        param_counts = []
        
        for config_dict in configs:
            config = llmconfig(
                vocab_size=5000,
                num_attention_heads=config_dict["hidden_size"] // 64,
                **config_dict
            )
            
            model = CausalLM(config)
            param_count = count_parameters(model)
            param_counts.append(param_count)
            
            print(f"hidden_size={config_dict['hidden_size']}, layers={config_dict['num_hidden_layers']}: {param_count:,} 参数")
            
        # 参数数量应该随配置合理增长
        assert param_counts[1] > param_counts[0]
        assert param_counts[2] > param_counts[1]
        
    def test_moe_parameter_efficiency(self):
        """测试MoE的参数效率"""
        base_config = {
            "vocab_size": 2000,
            "hidden_size": 256,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "intermediate_size": 512
        }
        
        # 标准模型
        standard_config = llmconfig(**base_config, use_moe=False)
        standard_model = CausalLM(standard_config)
        standard_params = count_parameters(standard_model)
        
        # MoE模型
        moe_config = llmconfig(
            **base_config,
            use_moe=True,
            num_experts_per_token=2,
            n_routed_experts=8
        )
        moe_model = CausalLM(moe_config)
        moe_params = count_parameters(moe_model)
        
        print(f"标准模型参数: {standard_params:,}")
        print(f"MoE模型参数: {moe_params:,}")
        print(f"MoE/标准参数比例: {moe_params/standard_params:.2f}")
        
        # MoE模型参数更多，但每次前向传播只使用部分参数
        assert moe_params > standard_params
        
        # 计算每次前向传播的有效参数使用
        # MoE中每次只使用 num_experts_per_token/n_routed_experts 的专家参数
        expert_utilization_ratio = moe_config.num_experts_per_token / moe_config.n_routed_experts
        print(f"专家利用率: {expert_utilization_ratio:.2f}")
        
        assert 0 < expert_utilization_ratio <= 1
        
    def test_attention_head_efficiency(self):
        """测试注意力头的效率"""
        base_config = {
            "vocab_size": 2000,
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "intermediate_size": 512
        }
        
        # 多头注意力
        mha_config = llmconfig(**base_config, num_attention_heads=8, num_key_value_heads=8)
        mha_model = CausalLM(mha_config)
        mha_params = count_parameters(mha_model)
        
        # 分组查询注意力
        gqa_config = llmconfig(**base_config, num_attention_heads=8, num_key_value_heads=4)
        gqa_model = CausalLM(gqa_config)
        gqa_params = count_parameters(gqa_model)
        
        print(f"MHA参数: {mha_params:,}")
        print(f"GQA参数: {gqa_params:,}")
        print(f"参数减少: {(mha_params-gqa_params)/mha_params*100:.1f}%")
        
        # GQA应该使用更少的参数
        assert gqa_params < mha_params
        
        # 但性能应该相近（这里只测试能正常运行）
        batch_size = 1
        seq_len = 16
        input_ids = torch.randint(0, base_config["vocab_size"], (batch_size, seq_len))
        
        with torch.no_grad():
            mha_output = mha_model(input_ids)
            gqa_output = gqa_model(input_ids)
            
        # 两种模型都应该产生合理的输出
        assert not torch.isnan(mha_output.logits).any()
        assert not torch.isnan(gqa_output.logits).any()