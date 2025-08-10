# Zero2LLMV 测试框架

这是一个全面的测试框架，用于验证 Zero2LLMV 模型架构的各个组件。

## 测试结构

```
tests/
├── __init__.py                 # 测试包初始化
├── conftest.py                 # pytest配置和共享fixture
├── utils.py                    # 测试工具函数
├── run_tests.py               # 测试运行脚本
├── unit/                      # 单元测试
│   ├── test_config.py         # 配置类测试
│   ├── test_components.py     # 基础组件测试 (RMSNorm, RoPE等)
│   ├── test_attention.py      # 注意力机制测试
│   ├── test_feedforward.py    # 前馈网络测试
│   ├── test_moe.py           # MoE相关测试
│   └── test_model.py         # 完整模型测试
├── integration/               # 集成测试
│   └── test_model_integration.py  # 模型集成测试
├── performance/               # 性能测试
│   └── test_performance.py   # 性能基准测试
└── fixtures/                  # 测试夹具和样本数据
    ├── __init__.py
    └── fixtures.py           # 测试夹具定义
```

## 快速开始

### 运行所有测试
```bash
# 使用测试运行脚本
python tests/run_tests.py --mode all

# 或直接使用 pytest
uv run pytest tests/
```

### 快速测试（推荐用于开发）
```bash
python tests/run_tests.py --mode quick
```

### 运行特定类型的测试
```bash
# 单元测试
python tests/run_tests.py --mode unit

# 集成测试  
python tests/run_tests.py --mode integration

# 性能测试
python tests/run_tests.py --mode performance

# 带覆盖率的测试
python tests/run_tests.py --mode coverage
```

## 测试类别

### 1. 单元测试 (`tests/unit/`)

测试各个组件的独立功能：

- **配置测试** (`test_config.py`): 验证模型配置的正确性
- **基础组件测试** (`test_components.py`): RMSNorm、位置编码等核心组件
- **注意力机制测试** (`test_attention.py`): 多头注意力、GQA、KV缓存等
- **前馈网络测试** (`test_feedforward.py`): SwiGLU前馈网络
- **MoE测试** (`test_moe.py`): 专家混合模型相关功能
- **模型测试** (`test_model.py`): 完整的LLM和CausalLM模型

### 2. 集成测试 (`tests/integration/`)

测试组件间的协同工作：

- 端到端推理测试
- KV缓存一致性测试  
- 训练/推理模式切换
- 不同模型配置的集成测试
- HuggingFace兼容性测试

### 3. 性能测试 (`tests/performance/`)

评估模型性能特性：

- 推理速度基准
- KV缓存加速效果
- 内存使用测试
- 参数效率分析
- 不同配置的性能对比

## 测试工具

### 核心工具函数 (`tests/utils.py`)

- `assert_tensor_shape()`: 验证张量形状
- `assert_tensor_close()`: 验证张量数值接近度  
- `count_parameters()`: 计算模型参数数量
- `measure_memory_usage()`: 测量内存使用
- `ModelTester`: 模型测试辅助类

### 测试夹具 (`tests/fixtures/`)

提供预定义的测试配置和数据：

- `ModelFixtures`: 各种模型配置（tiny, small, medium, GQA, MoE等）
- `DataFixtures`: 测试数据生成（输入、掩码等）
- `BenchmarkFixtures`: 性能测试配置

## 测试配置 (`conftest.py`)

提供共享的pytest fixture：

- 不同大小的模型配置
- 标准、GQA、MoE模型配置  
- 设备选择（CPU/CUDA）
- 数据类型配置

## 运行选项

### 测试运行脚本选项

```bash
python tests/run_tests.py --help
```

常用选项：
- `--mode {quick,unit,integration,performance,all,coverage}`: 测试模式
- `--verbose, -v`: 详细输出
- `--parallel, -p`: 并行运行
- `--pattern PATTERN`: 匹配特定测试
- `--markers MARKERS`: 运行特定标记的测试

### 直接使用pytest

```bash
# 运行特定测试文件
uv run pytest tests/unit/test_attention.py

# 运行特定测试方法
uv run pytest tests/unit/test_attention.py::TestAttention::test_attention_forward_basic

# 显示详细输出
uv run pytest -v tests/

# 并行运行
uv run pytest -n auto tests/

# 只运行失败的测试
uv run pytest --lf tests/

# 生成HTML覆盖率报告
uv run pytest --cov=models --cov-report=html tests/
```

## 测试覆盖的功能

### 核心架构组件
- ✅ RMSNorm层归一化
- ✅ 旋转位置编码 (RoPE)
- ✅ 多头注意力机制
- ✅ 分组查询注意力 (GQA)
- ✅ SwiGLU前馈网络
- ✅ 专家混合模型 (MoE)

### 模型功能
- ✅ 完整的LLM和CausalLM模型
- ✅ KV缓存机制
- ✅ 训练和推理模式
- ✅ 梯度计算和反向传播
- ✅ 不同数据类型支持

### 性能特性
- ✅ 推理速度测试
- ✅ 内存使用分析
- ✅ 参数效率评估
- ✅ 批处理效率
- ✅ 序列长度缩放

## 测试数据

测试使用各种配置和数据：

- **模型大小**: 从tiny (32 hidden_size) 到 large (768+ hidden_size)
- **序列长度**: 8 到 1024 tokens
- **批次大小**: 1 到 16
- **词汇表大小**: 100 到 50,000
- **专家配置**: 2到16个专家，不同的专家选择策略

## 持续集成

建议在CI/CD流水线中使用：

```yaml
# GitHub Actions示例
- name: 运行快速测试
  run: python tests/run_tests.py --mode quick

- name: 运行完整测试套件  
  run: python tests/run_tests.py --mode all --parallel

- name: 生成覆盖率报告
  run: python tests/run_tests.py --mode coverage
```

## 贡献指南

添加新测试时请遵循：

1. 将单元测试放在 `tests/unit/` 目录
2. 将集成测试放在 `tests/integration/` 目录  
3. 性能测试放在 `tests/performance/` 目录
4. 使用适当的断言和错误消息
5. 添加必要的中文注释说明测试目的
6. 确保测试具有确定性和可重现性

## 故障排除

### 常见问题

1. **CUDA内存不足**: 使用更小的批次大小或模型配置
2. **测试超时**: 增加pytest超时设置或使用更小的测试配置  
3. **随机性问题**: 确保使用 `set_seed()` 函数设置固定种子
4. **导入错误**: 确保在项目根目录运行测试

### 调试技巧

```bash
# 运行单个测试并显示详细信息
uv run pytest -v -s tests/unit/test_attention.py::TestAttention::test_attention_forward_basic

# 在第一个失败处停止
uv run pytest -x tests/

# 显示最慢的10个测试
uv run pytest --durations=10 tests/
```