# Zero2LLMV 语言模型架构解析

## 模型概览

Zero2LLMV 是一个基于 Transformer 的多模态大语言模型，支持文本生成和视觉语言理解。模型采用解码器架构，具备以下核心特性：

- **基础模型**: CausalLM (因果语言模型)
- **视觉扩展**: VLM (视觉语言模型) 
- **高级特性**: MoE (专家混合)、GQA (分组查询注意力)、RoPE (旋转位置编码)

## 整体架构流程图

```mermaid
graph TB
    subgraph "输入层 Input Layer"
        A[Token IDs<br/>形状: batch_size × seq_len] 
        B[Pixel Values<br/>形状: batch_size × num_images × 3 × 224 × 224]
        C[Attention Mask<br/>形状: batch_size × seq_len]
    end
    
    subgraph "嵌入层 Embedding Layer"
        D[Token Embedding<br/>形状: batch_size × seq_len × hidden_size]
        E[Dropout<br/>概率: 0.1]
    end
    
    subgraph "视觉处理 Vision Processing"
        F[CLIP Vision Encoder<br/>冻结参数]
        G[Vision Features<br/>形状: batch_size × 196 × 768]
        H[Vision Projector<br/>线性层: 768 → hidden_size]
        I[Projected Vision Features<br/>形状: batch_size × 196 × hidden_size]
    end
    
    subgraph "多模态融合 Multimodal Fusion"
        J[特征融合函数<br/>count_vision_proj]
        K[融合后隐藏状态<br/>形状: batch_size × seq_len × hidden_size]
    end
    
    subgraph "位置编码 Position Encoding"
        L[RoPE 预计算<br/>freqs_cos, freqs_sin]
        M[位置嵌入<br/>形状: seq_len × head_dim]
    end
    
    subgraph "Transformer 层级联 N × Transformer Layers"
        N[Layer i Input<br/>形状: batch_size × seq_len × hidden_size]
        
        subgraph "注意力机制 Attention Mechanism"
            O[RMSNorm<br/>输入归一化]
            P[Q/K/V 投影<br/>Q: batch_size × seq_len × num_heads × head_dim<br/>K,V: batch_size × seq_len × num_kv_heads × head_dim]
            Q[RoPE 位置编码<br/>应用旋转变换]
            R[KV Cache<br/>历史键值缓存]
            S[注意力计算<br/>Scaled Dot-Product Attention]
            T[输出投影<br/>形状: batch_size × seq_len × hidden_size]
        end
        
        subgraph "前馈网络 Feed Forward"
            U[RMSNorm<br/>注意力后归一化]
            V{MoE 模式?}
            W[标准 FFN<br/>SwiGLU 激活]
            X[专家混合 MoE<br/>Gate + Experts]
            Y[FFN 输出<br/>形状: batch_size × seq_len × hidden_size]
        end
        
        Z[残差连接 + Layer i+1]
    end
    
    subgraph "输出层 Output Layer"
        AA[最终 RMSNorm<br/>形状: batch_size × seq_len × hidden_size]
        BB[Language Model Head<br/>线性层: hidden_size → vocab_size]
        CC[Logits<br/>形状: batch_size × seq_len × vocab_size]
        DD[概率分布<br/>Softmax 后的下一词概率]
    end
    
    %% 数据流连接
    A --> D
    B --> F
    F --> G
    G --> H
    H --> I
    
    D --> E
    E --> J
    I --> J
    J --> K
    
    L --> M
    
    K --> N
    M --> Q
    
    N --> O
    O --> P
    P --> Q
    Q --> R
    R --> S
    S --> T
    T --> U
    
    U --> V
    V -->|标准模式| W
    V -->|MoE模式| X
    W --> Y
    X --> Y
    Y --> Z
    
    Z --> AA
    AA --> BB
    BB --> CC
    CC --> DD
    
    %% 样式定义
    classDef inputClass fill:#e1f5fe
    classDef embeddingClass fill:#f3e5f5
    classDef visionClass fill:#e8f5e8
    classDef fusionClass fill:#fff3e0
    classDef positionClass fill:#fce4ec
    classDef transformerClass fill:#e3f2fd
    classDef attentionClass fill:#f1f8e9
    classDef ffnClass fill:#fff8e1
    classDef outputClass fill:#efebe9
    
    class A,B,C inputClass
    class D,E embeddingClass
    class F,G,H,I visionClass
    class J,K fusionClass
    class L,M positionClass
    class N,Z transformerClass
    class O,P,Q,R,S,T attentionClass
    class U,V,W,X,Y ffnClass
    class AA,BB,CC,DD outputClass
```

## 关键组件详细分析

### 1. 注意力机制详解

```mermaid
graph TB
    subgraph "多头注意力 Multi-Head Attention"
        A["输入 X\n形状: batch_size × seq_len × hidden_size"]
        B["Q 投影\n形状: batch_size × seq_len × num_heads × head_dim"]
        C["K 投影\n形状: batch_size × seq_len × num_kv_heads × head_dim"]
        D["V 投影\n形状: batch_size × seq_len × num_kv_heads × head_dim"]
        
        E["RoPE 位置编码\nQ' = apply_rotary_pos_emb(Q)\nK' = apply_rotary_pos_emb(K)"]
        
        F["KV 重复\nrepeat_kv: 适配分组查询注意力\nK', V': batch_size × seq_len × num_heads × head_dim"]
        
        G["注意力分数\nScores = Q' @ K'^T / √head_dim\n形状: batch_size × num_heads × seq_len × seq_len"]
        
        H["因果掩码\n上三角矩阵填充 -∞"]
        
        I["Softmax 归一化\n形状: batch_size × num_heads × seq_len × seq_len"]
        
        J["注意力输出\nOutput = Attention @ V'\n形状: batch_size × num_heads × seq_len × head_dim"]
        
        K["重排 + 投影\n形状: batch_size × seq_len × hidden_size"]
    end
    
    A --> B
    A --> C
    A --> D
    B --> E
    C --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
```

### 2. MoE 专家混合架构

```mermaid
graph TB
    subgraph "专家混合 Mixture of Experts"
        A[输入特征<br/>形状: batch_size × seq_len × hidden_size]
        
        B[门控网络 Gate<br/>线性层: hidden_size → n_experts]
        C[专家评分<br/>Softmax 概率分布]
        D[Top-K 选择<br/>选择最优的 k 个专家]
        
        E[专家网络池<br/>n_routed_experts 个 FFN]
        F[专家1<br/>SwiGLU FFN]
        G[专家2<br/>SwiGLU FFN]
        H[专家N<br/>SwiGLU FFN]
        
        I[共享专家池<br/>n_shared_experts 个 FFN]
        J[共享专家1<br/>所有 token 使用]
        K[共享专家2<br/>所有 token 使用]
        
        L[加权输出<br/>专家输出 × 门控权重]
        M[辅助损失<br/>负载均衡约束]
        N[最终输出<br/>路由专家 + 共享专家]
    end
    
    A --> B
    B --> C
    C --> D
    
    A --> E
    E --> F
    E --> G
    E --> H
    
    A --> I
    I --> J
    I --> K
    
    D --> L
    F --> L
    G --> L
    H --> L
    
    C --> M
    L --> N
    J --> N
    K --> N
```

### 3. 视觉语言融合流程

```mermaid
graph TB
    subgraph "多模态特征融合 Multimodal Feature Fusion"
        A[图像输入<br/>PIL Image 224×224]
        B[CLIP 预处理<br/>Normalize + Tensor]
        C[CLIP Vision Encoder<br/>ViT-Base-Patch16]
        D[Patch 特征提取<br/>196 个 patch 特征<br/>形状: 196 × 768]
        
        E[视觉投影器<br/>线性层: 768 → hidden_size]
        F[投影后特征<br/>形状: 196 × hidden_size]
        
        G[文本 Token 序列<br/>包含图像占位符 @...@]
        H[Token 嵌入<br/>形状: seq_len × hidden_size]
        
        I[占位符检测<br/>查找图像 token ID 位置]
        J[特征替换<br/>用视觉特征替换占位符]
        K[融合序列<br/>文本 + 视觉统一表示]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    
    G --> H
    H --> I
    F --> J
    I --> J
    J --> K
```

## 张量形状变换总览

| 阶段 | 输入形状 | 输出形状 | 关键操作 |
|------|----------|----------|----------|
| Token Embedding | `[B, L]` | `[B, L, H]` | 词嵌入查找 |
| Vision Encoding | `[B, 3, 224, 224]` | `[B, 196, 768]` | CLIP ViT 编码 |
| Vision Projection | `[B, 196, 768]` | `[B, 196, H]` | 线性变换 |
| Q/K/V Projection | `[B, L, H]` | `[B, L, N_h, D_h]` | 多头分割 |
| Attention Scores | `[B, N_h, L, D_h]` | `[B, N_h, L, L]` | Q@K^T / √D_h |
| FFN Gate | `[B, L, H]` | `[B, L, I]` | 线性投影 |
| FFN Up/Down | `[B, L, H]` | `[B, L, H]` | SwiGLU 变换 |
| LM Head | `[B, L, H]` | `[B, L, V]` | 词汇表投影 |

**符号说明:**
- B: batch_size
- L: seq_len (序列长度)
- H: hidden_size (隐藏层维度)
- N_h: num_heads (注意力头数)
- D_h: head_dim (每头维度)
- I: intermediate_size (FFN中间维度)
- V: vocab_size (词汇表大小)

## 模型参数量估算

### 基础 LLM 参数量

```mermaid
graph LR
    subgraph "参数分布 Parameter Distribution"
        A[Token Embedding<br/>V × H]
        B[Position Encoding<br/>预计算，无参数]
        C[N × Transformer Layers]
        D[Attention Weights<br/>4 × H × H 每层]
        E[FFN Weights<br/>3 × H × I 每层]
        F[LayerNorm<br/>2 × H 每层]
        G[LM Head<br/>共享 Embedding 权重]
    end
    
    H[总参数量<br/>≈ V×H + N×(4×H² + 3×H×I + 2×H)]
```

### 关键超参数配置

| 参数名称 | 默认值 | 影响 |
|----------|--------|------|
| `hidden_size` | 768 | 模型表达能力，影响所有层的参数量 |
| `num_hidden_layers` | 12 | 模型深度，影响总参数量和计算复杂度 |
| `num_attention_heads` | 12 | 注意力机制的并行度 |
| `num_key_value_heads` | 4 | GQA 中的 KV 头数，减少 KV Cache 内存 |
| `intermediate_size` | 3072 | FFN 中间层大小，影响前馈网络容量 |
| `vocab_size` | 30000 | 词汇表大小，影响嵌入层参数量 |
| `max_position_embeddings` | 2048 | 最大序列长度 |

### MoE 扩展参数

| 参数名称 | 默认值 | 说明 |
|----------|--------|------|
| `use_moe` | False | 是否启用专家混合 |
| `n_routed_experts` | 8 | 可路由专家数量 |
| `n_shared_experts` | 2 | 共享专家数量 |
| `num_experts_per_token` | 2 | 每个 token 激活的专家数 |
| `aux_loss_alpha` | 0.01 | 辅助损失权重 |

## 实现特色

1. **内存优化**: 
   - KV Cache 缓存机制
   - 分组查询注意力 (GQA)
   - 可选的 Flash Attention

2. **多模态融合**:
   - CLIP 视觉编码器
   - 灵活的特征投影机制
   - 统一的序列建模

3. **可扩展架构**:
   - 模块化的专家混合
   - 可配置的模型规模
   - 标准的 HuggingFace 接口

4. **训练稳定性**:
   - RMSNorm 归一化
   - 梯度裁剪
   - 权重共享策略