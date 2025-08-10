# MiniMindVLM 使用指南与API文档

## 🚀 快速开始

### 环境配置

```bash
# 安装依赖
pip install torch torchvision transformers pillow

# 或使用 uv (推荐)
uv add torch torchvision transformers pillow
```

### 基础使用示例

```python
import torch
from PIL import Image
from models.vision_encoder import MiniMindVLM, VLLMconfig

# 1. 初始化模型配置
config = VLLMconfig(
    # 基础语言模型配置
    vocab_size=50000,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    intermediate_size=3072,
    max_position_embeddings=2048,
    
    # 多模态特定配置
    image_special_token='@' * 196,  # 图像占位符 (196个patch)
    image_ids=[34] * 196,           # 图像token的ID序列
    
    # 高级特性 (可选)
    use_moe=False,                  # 是否启用MoE
    flash_attn=True,                # Flash Attention加速
    dropout=0.1,
    rope_theta=10000.0
)

# 2. 加载模型
model = MiniMindVLM(
    params=config,
    vision_model_path="./models/clip-vit-base-patch16"  # CLIP模型路径
)

# 3. 准备多模态输入
text = "请描述这张图片中的内容："
image = Image.open("example.jpg")

# 4. 预处理
# 假设已有tokenizer
input_text = text + config.image_special_token  # 添加图像占位符
input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']

# 图像预处理
pixel_values = MiniMindVLM.image2tensor(image, model.processor)
pixel_values = pixel_values.unsqueeze(0)  # 添加batch维度

# 5. 模型推理
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        use_cache=True,  # 启用KV缓存加速
        logits_to_keep=1  # 只保留最后一个token的logits
    )

# 6. 获取预测结果
logits = outputs.logits
probabilities = torch.softmax(logits[0, -1, :], dim=-1)
predicted_token_id = torch.argmax(probabilities).item()

print(f"预测的下一个token ID: {predicted_token_id}")
```

## 🔧 高级配置示例

### MoE多专家模型配置

```python
# 大规模MoE配置
moe_config = VLLMconfig(
    # 基础配置
    vocab_size=100000,
    hidden_size=1024,
    num_attention_heads=16,
    num_key_value_heads=8,  # GQA: 减少KV头数
    num_hidden_layers=24,
    
    # MoE专家混合配置
    use_moe=True,
    num_experts_per_token=2,    # 每个token激活2个专家
    n_routed_experts=16,        # 总共16个专家
    n_shared_experts=2,         # 2个共享专家
    aux_loss_alpha=0.01,        # 负载均衡损失权重
    norm_topk_prob=True,        # 归一化专家权重
    
    # 性能优化
    flash_attn=True,
    use_cache=True,
    max_position_embeddings=4096
)

# 初始化MoE模型
moe_model = MiniMindVLM(
    params=moe_config,
    vision_model_path="openai/clip-vit-large-patch14"
)

print(f"模型参数量: {sum(p.numel() for p in moe_model.parameters()):,}")
print(f"激活参数量: {sum(p.numel() for p in moe_model.parameters() if p.requires_grad):,}")
```

### 批量推理示例

```python
def batch_multimodal_inference(model, texts, images, batch_size=4):
    """
    批量多模态推理函数
    
    Args:
        model: MiniMindVLM模型实例
        texts: 文本列表
        images: PIL图像列表  
        batch_size: 批处理大小
        
    Returns:
        预测结果列表
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_images = images[i:i+batch_size]
        
        # 预处理批次
        input_ids_list = []
        pixel_values_list = []
        
        for text, image in zip(batch_texts, batch_images):
            # 文本处理
            input_text = text + model.params.image_special_token
            input_ids = tokenizer(input_text, 
                                return_tensors='pt', 
                                padding=True, 
                                truncation=True)['input_ids']
            input_ids_list.append(input_ids)
            
            # 图像处理
            pixel_values = MiniMindVLM.image2tensor(image, model.processor)
            pixel_values_list.append(pixel_values)
        
        # 批量张量
        batch_input_ids = torch.cat(input_ids_list, dim=0)
        batch_pixel_values = torch.stack(pixel_values_list, dim=0)
        
        # 批量推理
        with torch.no_grad():
            batch_outputs = model(
                input_ids=batch_input_ids,
                pixel_values=batch_pixel_values,
                use_cache=False  # 批量推理时关闭缓存
            )
        
        # 处理输出
        batch_logits = batch_outputs.logits
        batch_predictions = torch.argmax(batch_logits[:, -1, :], dim=-1)
        
        results.extend(batch_predictions.cpu().tolist())
    
    return results

# 使用示例
texts = ["描述图片:", "这是什么:", "图片内容:"]
images = [Image.open(f"image_{i}.jpg") for i in range(3)]

predictions = batch_multimodal_inference(model, texts, images)
```

## ⚡ 生成式推理示例

### 自回归文本生成

```python
def multimodal_generate(model, text_prompt, image, max_length=100, 
                       temperature=1.0, top_k=50, top_p=0.9):
    """
    多模态自回归生成函数
    
    Args:
        model: MiniMindVLM模型
        text_prompt: 文本提示
        image: PIL图像
        max_length: 生成的最大长度
        temperature: 采样温度
        top_k: Top-K采样参数
        top_p: Top-P (nucleus) 采样参数
        
    Returns:
        generated_text: 生成的文本
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 准备初始输入
    input_text = text_prompt + model.params.image_special_token
    input_ids = tokenizer(input_text, return_tensors='pt')['input_ids'].to(device)
    
    # 图像预处理
    pixel_values = MiniMindVLM.image2tensor(image, model.processor)
    pixel_values = pixel_values.unsqueeze(0).to(device)
    
    generated_tokens = []
    past_key_values = None
    
    with torch.no_grad():
        for step in range(max_length):
            # 当前输入 (第一步包含图像，后续步骤仅文本)
            current_pixel_values = pixel_values if step == 0 else None
            
            # 模型前向传播
            outputs = model(
                input_ids=input_ids,
                pixel_values=current_pixel_values,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # 获取下一个token的logits
            next_token_logits = outputs.logits[0, -1, :] / temperature
            
            # Top-K + Top-P 采样
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(0, top_k_indices, top_k_logits)
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样下一个token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # 检查结束条件
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token.item())
            
            # 准备下一步输入
            input_ids = next_token.unsqueeze(0)
            past_key_values = outputs.past_key_values
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

# 使用示例
image = Image.open("cat.jpg")
prompt = "这张图片显示了"

generated_description = multimodal_generate(
    model=model,
    text_prompt=prompt,
    image=image,
    max_length=50,
    temperature=0.7,
    top_k=40,
    top_p=0.9
)

print(f"生成的描述: {prompt}{generated_description}")
```

## 🧠 训练相关API

### 训练循环示例

```python
def train_multimodal_model(model, train_dataloader, optimizer, num_epochs):
    """
    多模态模型训练循环
    """
    model.train()
    device = next(model.parameters()).device
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_aux_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # 获取批次数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device) if 'pixel_values' in batch else None
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels  # 传入labels会自动计算交叉熵损失
            )
            
            # 计算总损失
            main_loss = outputs.loss
            aux_loss = outputs.aux_loss if hasattr(outputs, 'aux_loss') else 0
            total_loss_value = main_loss + aux_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss_value.backward()
            
            # 梯度裁剪 (可选)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            total_loss += main_loss.item()
            total_aux_loss += aux_loss if isinstance(aux_loss, float) else aux_loss.item()
            
            # 日志输出
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}: '
                      f'Loss={main_loss.item():.4f}, '
                      f'Aux_Loss={aux_loss:.4f if isinstance(aux_loss, float) else aux_loss.item():.4f}')
        
        avg_loss = total_loss / len(train_dataloader)
        avg_aux_loss = total_aux_loss / len(train_dataloader)
        print(f'Epoch {epoch} completed: Avg_Loss={avg_loss:.4f}, Avg_Aux_Loss={avg_aux_loss:.4f}')

# 优化器配置示例
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 分层学习率：视觉投影器使用更高学习率
vision_params = []
language_params = []

for name, param in model.named_parameters():
    if 'vision_proj' in name:
        vision_params.append(param)
    else:
        language_params.append(param)

optimizer = AdamW([
    {'params': language_params, 'lr': 1e-4, 'weight_decay': 0.01},
    {'params': vision_params, 'lr': 5e-4, 'weight_decay': 0.01}  # 视觉投影器使用更高学习率
])

scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
```

## 📊 性能分析和监控

```python
import time
import psutil
import torch.profiler

def benchmark_model_performance(model, sample_inputs, num_runs=100):
    """
    性能基准测试函数
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(**sample_inputs)
    
    # 同步GPU (如果使用CUDA)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 性能测试
    times = []
    memory_usage = []
    
    for i in range(num_runs):
        # 记录开始时间和内存
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else psutil.virtual_memory().used
        
        # 推理
        with torch.no_grad():
            outputs = model(**sample_inputs)
        
        # 同步并记录结束时间和内存
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        end_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else psutil.virtual_memory().used
        
        times.append(end_time - start_time)
        memory_usage.append(end_memory - start_memory)
    
    # 计算统计信息
    import numpy as np
    times = np.array(times)
    memory_usage = np.array(memory_usage)
    
    print(f"性能统计 (基于 {num_runs} 次运行):")
    print(f"  平均延迟: {times.mean()*1000:.2f} ± {times.std()*1000:.2f} ms")
    print(f"  最小延迟: {times.min()*1000:.2f} ms")
    print(f"  最大延迟: {times.max()*1000:.2f} ms")
    print(f"  内存使用: {memory_usage.mean()/1024/1024:.2f} ± {memory_usage.std()/1024/1024:.2f} MB")
    
    return {
        'latency_mean': times.mean(),
        'latency_std': times.std(),
        'memory_mean': memory_usage.mean(),
        'memory_std': memory_usage.std()
    }

# 使用PyTorch Profiler进行详细分析
def profile_model_execution(model, sample_inputs):
    """
    使用PyTorch Profiler分析模型执行
    """
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as profiler:
        with torch.no_grad():
            outputs = model(**sample_inputs)
    
    # 输出profiling结果
    print("Top 10 GPU operations by time:")
    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    print("\nTop 10 CPU operations by time:")
    print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    # 导出Chrome trace文件
    profiler.export_chrome_trace("model_trace.json")
    print("Trace exported to model_trace.json")
    
    return profiler
```

## 🔧 模型配置详解

### 完整配置选项

```python
class VLLMconfig(llmconfig):
    """
    MiniMindVLM配置类 - 所有可配置参数详解
    """
    
    def __init__(
        self,
        # === 基础语言模型配置 ===
        vocab_size: int = 30000,              # 词汇表大小
        hidden_size: int = 768,               # 隐藏层维度
        num_attention_heads: int = 12,        # 注意力头数
        num_key_value_heads: Optional[int] = None,  # KV头数 (GQA)
        num_hidden_layers: int = 12,          # Transformer层数
        intermediate_size: Optional[int] = None,    # FFN中间层维度
        max_position_embeddings: int = 2048,  # 最大位置编码长度
        
        # === 多模态特定配置 ===
        image_special_token: str = '@' * 196,       # 图像占位符token
        image_ids: List[int] = [34] * 196,          # 图像token ID序列
        ve_hidden_size: int = 768,                  # 视觉编码器输出维度
        
        # === 激活函数和正则化 ===
        hidden_act: str = "silu",             # 激活函数类型
        dropout: float = 0.1,                 # Dropout概率
        tie_word_embeddings: bool = True,     # 是否共享输入输出嵌入
        
        # === 位置编码配置 ===
        rope_theta: float = 10000.0,          # RoPE基数
        
        # === MoE专家混合配置 ===
        use_moe: bool = False,                # 是否启用MoE
        num_experts_per_token: int = 1,       # 每token激活的专家数
        n_routed_experts: int = 2,            # 路由专家总数
        n_shared_experts: int = 0,            # 共享专家数量
        aux_loss_alpha: float = 0.0,          # 辅助损失权重
        norm_topk_prob: bool = False,         # 是否归一化TopK概率
        seq_aux: bool = True,                 # 序列级辅助损失
        
        # === 性能优化配置 ===
        flash_attn: bool = True,              # Flash Attention
        use_cache: bool = True,               # KV缓存
        
        # === 特殊token配置 ===
        bos_token_id: int = 0,                # 开始token ID
        eos_token_id: int = 1,                # 结束token ID
        pad_token_id: int = 2,                # 填充token ID
        
        **kwargs
    ):
        # 设置图像相关配置
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.ve_hidden_size = ve_hidden_size
        
        # 调用父类初始化
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_act=hidden_act,
            dropout=dropout,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            use_moe=use_moe,
            num_experts_per_token=num_experts_per_token,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            aux_loss_alpha=aux_loss_alpha,
            norm_topk_prob=norm_topk_prob,
            seq_aux=seq_aux,
            flash_attn=flash_attn,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs
        )

# 不同规模的预设配置
def get_model_configs():
    """获取不同规模的预设配置"""
    
    configs = {
        "tiny": VLLMconfig(
            vocab_size=10000,
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=6,
            intermediate_size=512,
            max_position_embeddings=1024
        ),
        
        "small": VLLMconfig(
            vocab_size=30000,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            intermediate_size=3072,
            max_position_embeddings=2048
        ),
        
        "medium": VLLMconfig(
            vocab_size=50000,
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=8,  # GQA
            num_hidden_layers=24,
            intermediate_size=4096,
            max_position_embeddings=4096
        ),
        
        "large_moe": VLLMconfig(
            vocab_size=100000,
            hidden_size=1536,
            num_attention_heads=24,
            num_key_value_heads=12,
            num_hidden_layers=32,
            use_moe=True,
            num_experts_per_token=2,
            n_routed_experts=16,
            n_shared_experts=2,
            aux_loss_alpha=0.01,
            max_position_embeddings=8192
        )
    }
    
    return configs

# 使用预设配置
configs = get_model_configs()
small_model = MiniMindVLM(params=configs["small"])
```

## 🔍 故障排除指南

### 常见问题解决

```python
def diagnose_model_issues(model, sample_input):
    """
    模型问题诊断工具
    """
    print("=== MiniMindVLM 模型诊断 ===")
    
    # 1. 检查模型配置
    config = model.params
    print(f"✓ 模型配置: {config.hidden_size}d, {config.num_hidden_layers}层")
    print(f"✓ 词汇表大小: {config.vocab_size}")
    print(f"✓ MoE状态: {'启用' if config.use_moe else '禁用'}")
    
    # 2. 检查设备状态
    device = next(model.parameters()).device
    print(f"✓ 模型设备: {device}")
    
    # 3. 检查视觉组件
    if hasattr(model, 'vision_encoder') and model.vision_encoder is not None:
        print("✓ 视觉编码器: 已加载")
        vision_params = sum(p.numel() for p in model.vision_encoder.parameters())
        print(f"  视觉编码器参数量: {vision_params:,}")
    else:
        print("⚠ 视觉编码器: 未加载")
    
    # 4. 检查参数状态
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 总参数量: {total_params:,}")
    print(f"✓ 可训练参数: {trainable_params:,}")
    
    # 5. 测试前向传播
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(**sample_input)
        print("✓ 前向传播: 正常")
        print(f"  输出形状: {outputs.logits.shape}")
        
        # 检查输出数值
        if torch.isnan(outputs.logits).any():
            print("❌ 输出包含NaN值")
        elif torch.isinf(outputs.logits).any():
            print("❌ 输出包含Inf值")
        else:
            print("✓ 输出数值: 正常")
            
    except Exception as e:
        print(f"❌ 前向传播失败: {str(e)}")
    
    # 6. 内存使用检查
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"✓ GPU内存使用: {memory_allocated:.1f}MB / {memory_reserved:.1f}MB")
        
        if memory_allocated > memory_reserved * 0.9:
            print("⚠ GPU内存使用率过高，可能导致OOM")

# 诊断工具使用示例
sample_input = {
    'input_ids': torch.randint(0, 1000, (1, 50)),
    'pixel_values': torch.randn(1, 1, 3, 224, 224)
}

diagnose_model_issues(model, sample_input)
```

---

## 📚 API参考

### 核心类和方法

#### `VLLMconfig`
多模态语言模型配置类

**参数:**
- `image_special_token`: 图像占位符字符串
- `image_ids`: 图像token ID列表  
- 其他参数继承自`llmconfig`

#### `MiniMindVLM`
主要的多模态视觉语言模型类

**主要方法:**

##### `__init__(params, vision_model_path)`
- `params`: VLLMconfig配置对象
- `vision_model_path`: CLIP模型路径

##### `forward(input_ids, pixel_values=None, attention_mask=None, ...)`
- `input_ids`: 输入token序列 [B, L]
- `pixel_values`: 图像像素值 [B, N, 3, 224, 224]
- `attention_mask`: 注意力掩码 [B, L]
- `past_key_values`: KV缓存
- `use_cache`: 是否启用KV缓存
- `logits_to_keep`: 保留的logits数量

**返回:** `ModelOutput`包含logits、hidden_states、past_key_values等

##### 静态方法

- `get_vision_model(model_path)`: 加载CLIP模型
- `image2tensor(image, processor)`: 图像转张量  
- `get_image_embeddings(image_tensors, vision_model)`: 提取图像特征

#### `VisionEncoder`
视觉特征投影器

**参数:**
- `ve_hidden_size`: 视觉编码器输出维度
- `hidden_size`: 目标隐藏层维度

这份文档涵盖了MiniMindVLM的核心使用方法、高级功能配置、性能优化技巧和故障排除指南，为开发者提供了完整的使用参考。