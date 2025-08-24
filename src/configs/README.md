# Zero2LLMV YAML 配置系统使用指南

## 概述

Zero2LLMV 现在支持基于 YAML 的配置系统，提供更好的**可复现性、可追踪性、可验证性**和**灵活的覆盖机制**。

## 核心特性

🎯 **优先级配置系统**：默认值 < YAML 文件 < 环境变量 < 命令行参数  
🔧 **全面参数验证**：Pydantic 模型确保配置正确性  
📊 **实验追踪**：自动保存配置到实验目录  
🔄 **向后兼容**：完全兼容现有 CLI 使用方式  
🛡️ **安全设计**：敏感信息通过环境变量管理

## 快速开始

### 1. 基础使用

```bash
# 使用 YAML 配置运行训练
python train.py --config configs/config.yaml

# 使用实验配置
python train.py --config configs/experiments/high_lr_experiment.yaml

# CLI 参数覆盖 YAML
python train.py --config configs/config.yaml --batch_size 16 --learning_rate 3e-5
```

### 2. 向后兼容模式

```bash
# 现有命令完全不变
python train.py --model_type vlm --batch_size 8 --learning_rate 2e-5

# 自动生成 YAML 配置以便复现
python train.py --model_type vlm --save-merged-config
```

## 配置文件结构

### 主配置文件 (`configs/config.yaml`)

```yaml
# 模型配置
model:
  model_type: "vlm"           # 模型类型：llm 或 vlm
  model_config_path: ""       # 模型配置文件路径（可选）

# 数据配置  
data:
  data_path: "data/processed" # 训练数据目录
  max_seq_length: 512         # 最大序列长度 (1-8192)
  batch_size: 8               # 训练批次大小 (1-1024)
  num_workers: 4              # 数据加载器工作进程数 (0-64)

# 训练超参数
training:
  num_epochs: 3               # 训练轮数 (1-1000)
  learning_rate: 2e-5         # 初始学习率 (0-1)
  weight_decay: 0.01          # 权重衰减系数 (0-1)
  warmup_steps: 500           # 预热步数
  gradient_accumulation_steps: 4  # 梯度累积步数 (1-128)
  max_grad_norm: 1.0          # 梯度裁剪阈值 (0-10)
  use_amp: true               # 是否使用混合精度训练

# 检查点和日志
checkpoints:
  output_dir: "outputs"       # 输出目录
  save_steps: 1000            # 检查点保存间隔
  logging_steps: 100          # 日志记录间隔
  eval_steps: 500             # 评估间隔

# WandB 实验追踪
wandb:
  project: "zero2llmv"        # WandB 项目名
  name: ""                    # 实验名（空时自动生成）
  notes: ""                   # 实验描述
  tags: []                    # 标签列表
```

### 实验配置模板

项目提供了多个预设实验配置：

```
configs/experiments/
├── high_lr_experiment.yaml    # 高学习率实验
├── large_batch.yaml          # 大批次实验 
├── long_context.yaml         # 长上下文实验
└── llm_only.yaml             # 纯 LLM 训练
```

## 详细使用方法

### 1. 标准训练流程

```bash
# 步骤 1: 复制并编辑配置文件
cp configs/config.yaml my_experiment.yaml

# 步骤 2: 根据需要修改配置
# 编辑 my_experiment.yaml

# 步骤 3: 设置环境变量（可选）
export WANDB_BASE_URL="https://your-wandb-server.com"
export WANDB_API_KEY="your-api-key"

# 步骤 4: 开始训练
python train.py --config my_experiment.yaml
```

### 2. 快速参数调试

```bash
# 在 YAML 基础上快速调整参数
python train.py \
    --config configs/config.yaml \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_epochs 5
```

### 3. 实验系列训练

```bash
# 高学习率实验
python train.py --config configs/experiments/high_lr_experiment.yaml

# 大批次实验
python train.py --config configs/experiments/large_batch.yaml

# 长上下文实验  
python train.py --config configs/experiments/long_context.yaml
```

### 4. 自托管 WandB 配置

```bash
# 方式一：环境变量
export WANDB_BASE_URL="https://your-wandb-server.com"
export WANDB_API_KEY="your-api-key"
python train.py --config configs/config.yaml

# 方式二：命令行参数
python train.py \
    --config configs/config.yaml \
    --wandb-base-url "https://your-wandb-server.com" \
    --wandb-api-key "your-api-key"
```

## 配置验证和错误处理

### 自动验证功能

系统会自动验证：
- ✅ 参数类型和范围检查
- ✅ 参数间依赖关系验证
- ✅ 配置文件语法检查
- ✅ 敏感参数保护

### 错误处理示例

```bash
# 配置错误时会显示详细信息
python train.py --config invalid_config.yaml

# 输出示例:
# ❌ 配置验证失败:
# 字段 'learning_rate': 值 2.0 超出范围 (0-1)
# 建议: 使用典型的学习率值如 2e-5 或 1e-4
```

### 调试模式

```bash
# 允许使用默认值回退
python train.py --config config.yaml --allow-default-fallback

# 保存最终合并的配置
python train.py --config config.yaml --save-merged-config
```

## 命令行参数完整列表

### 配置相关
- `--config`: YAML 配置文件路径
- `--allow-default-fallback`: 配置错误时使用默认值
- `--save-merged-config`: 保存最终合并的配置

### 模型参数
- `--model_type`: 模型类型 (llm/vlm)
- `--model_config_path`: 模型配置文件路径

### 数据参数
- `--data_path`: 训练数据路径
- `--max_seq_length`: 最大序列长度
- `--batch_size`: 批次大小
- `--num_workers`: 数据加载器线程数

### 训练参数
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--weight_decay`: 权重衰减
- `--warmup_steps`: 预热步数
- `--gradient_accumulation_steps`: 梯度累积步数
- `--max_grad_norm`: 梯度裁剪阈值
- `--use_amp`: 启用混合精度训练

### 检查点参数
- `--output_dir`: 输出目录
- `--save_steps`: 保存间隔
- `--logging_steps`: 日志间隔
- `--eval_steps`: 评估间隔

### WandB 参数
- `--wandb_project`: 项目名
- `--wandb_name`: 实验名
- `--wandb_notes`: 实验描述
- `--wandb_tags`: 标签列表

## 最佳实践

### 1. 配置管理

```bash
# ✅ 推荐：版本化配置文件
git add configs/my_experiment.yaml
git commit -m "Add experiment configuration"

# ✅ 推荐：使用描述性实验名
wandb:
  name: "vlm-lr2e5-bs8-seq512-v1"
  notes: "基线 VLM 训练，标准超参数"
  tags: ["baseline", "vlm", "production"]
```

### 2. 参数调优

```bash
# ✅ 快速迭代：YAML + CLI 覆盖
python train.py --config base.yaml --learning_rate 1e-4
python train.py --config base.yaml --learning_rate 5e-5
python train.py --config base.yaml --learning_rate 2e-5

# ✅ 系统化：创建专门的实验配置
cp configs/config.yaml configs/experiments/lr_sweep_1e4.yaml
# 编辑配置文件
python train.py --config configs/experiments/lr_sweep_1e4.yaml
```

### 3. 实验追踪

```bash
# ✅ 实验自动保存配置到输出目录
python train.py --config my_config.yaml
# 生成: outputs/{timestamp}/config.yaml 和 config.json

# ✅ WandB 自动记录配置
# 所有参数会自动上传到 WandB 进行追踪
```

## 故障排除

### 常见问题

**Q: YAML 语法错误**
```
A: 检查缩进、引号和冒号。使用 YAML 验证工具或 VS Code YAML 插件。
```

**Q: 参数验证失败** 
```
A: 查看错误消息中的建议值范围，调整配置文件中的对应参数。
```

**Q: WandB 连接失败**
```bash
A: 检查环境变量和网络连接：
   wandb login --host=https://your-wandb-server.com
   wandb whoami
```

**Q: 找不到配置文件**
```
A: 使用绝对路径或确保从项目根目录运行：
   python train.py --config ./configs/config.yaml
```

### 获取帮助

```bash
# 查看所有可用参数
python train.py --help

# 验证配置文件（不开始训练）
python validate_config_system.py configs/config.yaml

# 运行测试确保系统正常
python -m pytest tests/functional/ -v
```

## 从旧版本迁移

### CLI 到 YAML 映射

| CLI 参数 | YAML 路径 |
|---------|-----------|
| `--model_type vlm` | `model.model_type: "vlm"` |
| `--batch_size 8` | `data.batch_size: 8` |
| `--learning_rate 2e-5` | `training.learning_rate: 2e-5` |
| `--num_epochs 3` | `training.num_epochs: 3` |
| `--output_dir outputs` | `checkpoints.output_dir: "outputs"` |
| `--wandb_project zero2llmv` | `wandb.project: "zero2llmv"` |

### 渐进迁移策略

```bash
# 阶段 1：继续使用现有 CLI（无需改变）
python train.py --model_type vlm --batch_size 8

# 阶段 2：生成 YAML 配置文件
python train.py --model_type vlm --batch_size 8 --save-merged-config

# 阶段 3：切换到 YAML 配置
python train.py --config generated_config.yaml

# 阶段 4：优化和定制 YAML 配置
# 编辑和版本化 generated_config.yaml
```

现在您的 Zero2LLMV 项目已经具备了现代化的 YAML 配置系统！🎉