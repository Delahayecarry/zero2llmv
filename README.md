# Zero2LLMV 🚀

**支持YAML配置的多模态大语言模型训练框架**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-orange.svg)](https://pytorch.org/)
[![SwanLab](https://img.shields.io/badge/SwanLab-监控支持-green.svg)](https://swanlab.cn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## ✨ 核心特性

🔧 **灵活配置管理**：基于YAML配置文件的训练系统  
🤖 **多模态支持**：同时支持大语言模型(LLM)和视觉语言模型(VLM)  
⚡ **高效训练**：支持混合精度训练、梯度累积和KV缓存优化  
📊 **实验监控**：集成SwanLab监控，完全替代WandB  
🧠 **专家混合**：MoE架构支持，提升模型效率  
🔄 **自动验证**：完整的配置验证和参数检查机制

## 📦 安装部署

### 环境要求

- Python 3.10+
- PyTorch 2.8+
- CUDA 11.8+ (GPU训练)

### 快速安装

```bash
# 克隆项目
git clone https://github.com/delahayecarry/zero2llmv.git
cd zero2llmv

# 使用uv安装依赖（推荐）
uv sync

# 或者使用pip安装
pip install -e .

# 开发环境依赖
uv sync --dev
# 或
pip install -e ".[dev]"
```

### 验证安装

```bash
# 激活虚拟环境（如果使用uv）
source .venv/bin/activate

# 运行测试验证
uv run pytest
```

## 🚀 快速开始

### 基础训练

```bash
# 使用默认配置训练VLM模型
uv run python train.py --config configs/config.yaml

# CLI参数覆盖YAML配置
uv run python train.py --config configs/config.yaml --batch_size 16 --learning_rate 3e-5

# 使用实验配置
uv run python train.py --config configs/experiments/high_lr_experiment.yaml

# 纯命令行参数训练
uv run python train.py --model_type vlm --batch_size 8 --learning_rate 2e-5
```

### 配置SwanLab监控

```bash
# 设置环境变量
export SWANLAB_API_KEY="your-api-key"
uv run python train.py --config configs/config.yaml

# 通过命令行参数
uv run python train.py \
    --config configs/config.yaml \
    --swanlab-project "VLLM" \
    --swanlab-workspace "delahayecarry" \
    --swanlab-experiment-name "my-experiment"
```

## 📁 项目结构

```
zero2llmv/
├── src/zero2llmv/           # 主包代码
│   ├── models/              # 模型定义
│   │   ├── llm.py          # 大语言模型
│   │   ├── vision_encoder.py # 视觉编码器
│   │   └── llmconfig.py    # 模型配置
│   └── configs/             # 配置管理
│       ├── training_models.py # Pydantic配置模型
│       └── config_loader.py   # 配置加载器
├── configs/                 # 配置文件
│   ├── config.yaml         # 默认配置
│   ├── experiments/        # 实验配置
│   └── README.md           # 配置使用说明
├── tests/                  # 测试用例
│   ├── functional/         # 功能测试
│   ├── unit/              # 单元测试
│   └── integration/       # 集成测试
├── docs/                  # 文档
├── data/                  # 数据目录
├── train.py              # 训练脚本
└── pyproject.toml        # 项目配置
```

## ⚙️ 配置系统

### YAML配置文件结构

```yaml
# 模型配置
model:
  model_type: "vlm"           # llm 或 vlm
  model_config_path: ""       # 自定义模型配置

# 数据配置
data:
  data_path: "data/processed"
  max_seq_length: 512
  batch_size: 8
  num_workers: 4

# 训练参数
training:
  num_epochs: 3
  learning_rate: 2e-5
  weight_decay: 0.01
  use_amp: true

# SwanLab监控
swanlab:
  project: "VLLM"                 # SwanLab项目名
  workspace: "delahayecarry"      # SwanLab工作空间
  experiment_name: ""             # 实验名称（留空自动生成）
  description: ""                 # 实验描述
  logdir: ""                      # 日志目录
```

### 参数优先级

1. **YAML配置文件** (最低优先级)
2. **环境变量** (中等优先级)  
3. **命令行参数** (最高优先级)

## 🤖 支持的模型架构

### 大语言模型 (LLM)
- ✅ Transformer架构基础
- ✅ RMSNorm标准化  
- ✅ 旋转位置编码 (RoPE)
- ✅ 分组查询注意力 (GQA)
- ✅ SwiGLU激活函数
- ✅ 专家混合模型 (MoE)

### 视觉语言模型 (VLM)  
- ✅ CLIP视觉编码器集成
- ✅ 图像-文本对齐训练
- ✅ 多模态融合机制
- ✅ KV缓存优化

## 🧪 实验管理

### 预定义实验

```bash
# 高学习率实验
uv run python train.py --config configs/experiments/high_lr_experiment.yaml

# 大批量训练
uv run python train.py --config configs/experiments/large_batch.yaml  

# 长上下文训练
uv run python train.py --config configs/experiments/long_context.yaml
```

### 创建自定义实验

```bash
# 复制基础配置
cp configs/config.yaml my_experiment.yaml

# 编辑配置参数
# vim my_experiment.yaml

# 运行实验
uv run python train.py --config my_experiment.yaml
```

## 🧪 测试运行

```bash
# 运行全部测试
uv run pytest

# 运行功能测试
uv run pytest tests/functional/ -v

# 运行特定测试
uv run pytest tests/functional/test_yaml_loading.py -v

# 生成覆盖率报告
uv run pytest --cov=zero2llmv tests/
```

## 🛠️ 开发工具

### 代码格式化

```bash
# 使用black格式化代码
uv run black src/ tests/

# 使用isort整理导入
uv run isort src/ tests/

# 使用ruff进行代码检查
uv run ruff check src/ tests/
```

### 提交前检查

```bash
# 运行完整检查
uv run pytest && uv run black --check src/ tests/ && uv run isort --check src/ tests/ && uv run ruff check src/ tests/

# 或使用Makefile
make check
```

### 使用Makefile

```bash
# 查看可用命令
make help

# 安装依赖
make install        # 基础依赖
make install-dev    # 开发依赖

# 代码质量
make lint          # 代码检查
make format        # 代码格式化
make test          # 运行测试
make test-cov      # 测试+覆盖率

# 项目管理
make clean         # 清理临时文件
make build         # 构建包
```

## 🔧 高级配置

### 分布式训练

```yaml
training:
  distributed: true
  num_gpus: 8
  gradient_accumulation_steps: 4
```

### 内存优化

```yaml
training:
  use_amp: true              # 混合精度
  gradient_checkpointing: true # 梯度检查点
  dataloader_pin_memory: true  # 内存固定
```

### 自托管SwanLab

```yaml
swanlab:
  project: "VLLM"
  workspace: "yourworkspace"
  api_key: "your-api-key"        # 推荐使用环境变量 SWANLAB_API_KEY
  logdir: "./logs"               # 本地日志保存目录
```

## 📊 SwanLab监控集成

### SwanLab特性

Zero2LLMV 已完全集成 SwanLab 实验监控平台，提供以下功能：

- ✅ **自动日志记录**：训练损失、学习率、验证指标
- ✅ **实验管理**：项目和工作空间组织
- ✅ **参数追踪**：自动记录所有超参数配置
- ✅ **可视化图表**：实时训练曲线和指标展示
- ✅ **实验比较**：不同实验之间的对比分析

### SwanLab配置示例

```yaml
# 基础配置
swanlab:
  project: "VLLM"                    # 项目名称
  workspace: "delahayecarry"         # 工作空间
  experiment_name: "vlm-baseline"    # 实验名称
  description: "VLM基准实验"          # 实验描述

# 高级配置  
swanlab:
  project: "VLLM"
  workspace: "delahayecarry"
  experiment_name: "high-lr-exp"
  description: "高学习率实验"
  logdir: "./swanlab_logs"           # 本地日志目录
```

### 环境变量配置

```bash
# 推荐使用环境变量存储敏感信息
export SWANLAB_API_KEY="your-swanlab-api-key"

# 运行训练
uv run python train.py --config configs/config.yaml
```

### 监控指标

SwanLab 会自动记录以下指标：

**训练指标**：
- `train/loss` - 训练损失
- `train/learning_rate` - 学习率
- `train/global_step` - 全局步数
- `train/epoch` - 训练轮次

**系统指标**：
- `checkpoint/saved` - 检查点保存事件
- `epoch/loss` - 每轮平均损失

**配置参数**：
- 模型类型和架构参数
- 训练超参数
- 数据加载配置

## 📋 路线图

- [ ] **分布式训练**：多GPU分布式训练支持
- [ ] **更多视觉编码器**：支持ViT、DINO等多种视觉模型  
- [ ] **模型量化**：4-bit和8-bit量化支持
- [ ] **推理优化**：vLLM推理引擎集成
- [ ] **模型导出**：ONNX和TensorRT导出支持
- [ ] **Web界面**：基于Gradio的训练监控界面


## 📄 开源协议

本项目使用 [MIT 开源协议](LICENSE)。

## 🙏 致谢

- [Transformers](https://github.com/huggingface/transformers) - 预训练模型
- [PyTorch](https://pytorch.org/) - 深度学习框架  
- [SwanLab](https://swanlab.cn/) - 实验跟踪监控平台
- [Pydantic](https://pydantic.dev/) - 数据验证
- [uv](https://github.com/astral-sh/uv) - Python包管理

