# JSONL 多模态训练适配指南

## 🎯 概述
已成功将你的 JSONL 格式数据适配到 MiniMind-V trainer 架构中。现在可以使用对话格式的多模态数据进行训练。

## 📁 文件结构
```
zero2LLMV/
├── vlm_dataset.py          # JSONL格式的VLM数据集类
├── trainer_jsonl.py        # 适配的训练器
├── tokenizer/             # 你的自定义tokenizer
├── your_data.jsonl        # 你的JSONL训练数据
└── your_images/           # 对应的图片文件夹
```

## 📝 数据格式要求

### JSONL文件格式
每行一个JSON对象：
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "提供给定图像的简要描述。\n<image>"
    },
    {
      "role": "assistant",
      "content": "橄榄油是自由使用的健康成分。"
    }
  ],
  "image": "GCC_train_002582585.jpg"
}
```

### 图片文件夹
- 包含JSONL中引用的所有图片文件
- 支持常见图像格式 (jpg, png, etc.)
- 图片会被自动调整为 224x224 分辨率

## 🚀 使用方法

### 1. 准备数据
```bash
# 将你的数据文件放置到合适位置
cp your_training_data.jsonl ./data.jsonl
cp -r your_images/ ./images/
```

### 2. 基本训练命令
```bash
python trainer_jsonl.py \
    --data_path data.jsonl \
    --images_path images/ \
    --tokenizer_path tokenizer/ \
    --epochs 4 \
    --batch_size 16 \
    --learning_rate 4e-4
```

### 3. 完整参数说明
```bash
python trainer_jsonl.py \
    --data_path "path/to/your/data.jsonl" \
    --images_path "path/to/your/images/" \
    --tokenizer_path "path/to/tokenizer/" \
    --epochs 4 \
    --batch_size 16 \
    --learning_rate 4e-4 \
    --max_seq_len 640 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --accumulation_steps 1 \
    --grad_clip 1.0 \
    --log_interval 100 \
    --save_interval 1000 \
    --out_dir "./checkpoints" \
    --use_wandb \
    --wandb_project "YourProject"
```

## ⚙️ 关键特性

### 1. 智能Tokenizer加载
- ✅ 优先加载你放在 `tokenizer/` 文件夹的自定义tokenizer
- ✅ 自动回退到GPT2 tokenizer (如果自定义tokenizer不可用)
- ✅ 自动添加特殊token: `<|user|>`, `<|assistant|>`, `<|endoftext|>`, `<image>`

### 2. 图像标记处理
- ✅ 自动将对话中的 `<image>` 替换为你的 `image_special_token`
- ✅ 支持多轮对话格式
- ✅ 图像预处理和标准化

### 3. 训练优化
- ✅ 支持混合精度训练 (AMP)
- ✅ 支持分布式训练 (DDP) 
- ✅ 支持梯度累积
- ✅ 支持WandB监控

### 4. 数据验证
- ✅ 自动验证JSONL格式
- ✅ 跳过格式错误的数据行
- ✅ 处理缺失的图像文件

## 🔧 集成到你的项目

### 替换你现有的VLMDataset
```python
# 在你的trainer中替换这行：
# from dataset.lm_dataset import VLMDataset

# 改为：
from vlm_dataset import create_vlm_dataset

# 然后替换数据集创建代码：
train_ds = create_vlm_dataset(
    data_path=args.data_path,
    images_path=args.images_path, 
    tokenizer_path=args.tokenizer_path,
    max_length=max_seq_len
)
```

### 更新模型初始化
确保你的模型支持以下接口：
```python
# 前向传播
result = model(input_ids, pixel_values=pixel_values)

# 返回对象应包含:
result.logits    # [batch, seq_len, vocab_size]
result.aux_loss  # auxiliary loss tensor
```

## 📊 测试验证

运行测试确保一切正常：
```bash
# 测试数据集
python vlm_dataset.py

# 测试训练流程
python trainer_jsonl.py --epochs 1 --batch_size 2 --log_interval 1
```

## 🎯 下一步
1. 将你的真实模型 `MiniMindVLM` 集成到 `init_model()` 函数
2. 将你的真实数据替换测试数据
3. 调整超参数开始训练
4. 可选：启用WandB监控训练过程

## 💡 提示
- 如果遇到内存问题，减小 `batch_size` 或 `max_seq_len`
- 使用 `--accumulation_steps` 来模拟更大的batch size
- 定期保存检查点避免训练中断
- 使用 `--ddp` 进行多GPU训练