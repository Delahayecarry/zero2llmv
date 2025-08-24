# zerollm-v 🚀

**高效的多模态视觉语言模型训练框架**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-orange.svg)](https://pytorch.org/)
[![SwanLab](https://img.shields.io/badge/SwanLab-监控支持-green.svg)](https://swanlab.cn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## ✨ 项目特色

🎯 **YAML配置驱动**: 完全基于配置文件的训练，无命令行参数依赖  
👁️ **视觉语言模型**: 支持图像理解的多模态大语言模型训练  
📊 **SwanLab监控**: 实时训练监控和实验管理  
⚡ **高效训练**: 混合精度训练、梯度累积、分布式支持  
🔧 **简洁架构**: 清晰的项目结构，易于扩展和部署

## 🚀 快速开始

### 环境准备

```bash
# Python 3.10+ 和 PyTorch 2.8+ 是必需的
python --version  # 确保 >= 3.10
```

### 安装依赖

```bash
# 克隆项目
git clone <your-repo-url>
cd zerollm-v

# 使用 uv 安装（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 准备数据和模型

1. **准备训练数据**:
   ```bash
   # 创建数据目录
   mkdir -p dataset/
   # 将您的 JSONL 格式数据放入 dataset/pretrain_data.jsonl
   # 将图像文件放入 dataset/pretrain_images/
   ```

2. **准备预训练模型**:
   ```bash
   # 创建模型目录
   mkdir -p model/
   # 放入您的语言模型权重和tokenizer文件
   ```

### 配置训练

编辑 `configs/vlm_training.yaml` 文件：

```yaml
model:
  hidden_size: 512          # 模型隐藏层大小
  num_hidden_layers: 8      # 层数
  max_seq_len: 640          # 最大序列长度
  
data:
  data_path: "../dataset/pretrain_data.jsonl"
  images_path: "../dataset/pretrain_images"
  batch_size: 16            # 批处理大小
  
training:
  num_epochs: 4             # 训练轮数
  learning_rate: 4e-4       # 学习率
  
swanlab:
  project: "zerollm-v"      # SwanLab 项目名
  experiment_name: "vlm-pretrain"
```

### 开始训练

```bash
# 使用默认配置训练
python src/train_vlm.py

# 使用自定义配置
python src/train_vlm.py path/to/your/config.yaml
```

## 📁 项目结构

```
zerollm-v/
├── configs/
│   └── vlm_training.yaml         # VLM训练配置
├── docs/                         # 项目文档
├── src/                          # 源代码
│   ├── train_vlm.py             # 主训练脚本
│   ├── configs/                 # 配置系统
│   │   ├── config_loader.py     # 配置加载器
│   │   ├── training_models.py   # 配置模型定义
│   │   └── experiments/         # 实验配置示例
│   ├── dataset/                 # 数据处理模块
│   │   └── vlm_dataset.py
│   └── models/                  # 模型定义
│       ├── llm.py
│       ├── vision_encoder.py
│       └── vision_model/        # 视觉模型权重
├── tokenizer/                   # Tokenizer文件
└── README.md
```

## ⚙️ 训练配置

### 基础配置

- **模型配置**: 支持不同规模的VLM模型
- **数据配置**: JSONL格式多模态数据支持
- **训练配置**: 学习率、批大小、训练轮数等
- **监控配置**: SwanLab实验跟踪

### 高级功能

- **混合精度训练**: 节省显存，加速训练
- **梯度累积**: 支持更大的有效批处理大小
- **分布式训练**: 多GPU并行训练支持
- **检查点管理**: 自动保存和恢复训练状态

## 📊 监控和实验管理

项目集成了 SwanLab 用于训练监控：

- 实时损失和学习率曲线
- 训练进度和性能指标
- 模型检查点管理
- 实验对比和分析

## 🌐 云端部署

### 配置文件准备

确保配置文件中的路径适合云端环境：

```yaml
model:
  vision_model_path: "./model/vision_model/clip-vit-base-patch16"
  llm_weights_dir: "./out"

data:
  data_path: "./dataset/pretrain_data.jsonl"
  images_path: "./dataset/pretrain_images"

checkpoints:
  output_dir: "./outputs"
```

### 运行训练

```bash
# 在云端服务器上
cd zerollm-v
python src/train_vlm.py configs/vlm_training.yaml
```

## 🔧 自定义开发

### 添加新的数据集支持

扩展 `src/dataset/vlm_dataset.py` 来支持更多数据格式。

### 模型架构修改

在 `src/models/` 目录下添加或修改模型定义。

### 配置扩展

通过修改 `src/configs/training_models.py` 添加新的配置选项。

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🆘 问题和支持

如有问题或需要支持，请：

1. 查看 [Issues](../../issues) 页面
2. 创建新的 Issue 描述问题
3. 查看项目文档获取更多信息

---

**zerollm-v - 让多模态AI训练变得简单高效！** 🎉