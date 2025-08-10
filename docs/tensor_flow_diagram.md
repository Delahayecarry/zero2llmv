# Zero2LLMV 张量流程图

本文档详细描述了 Zero2LLMV 模型中张量在各个组件间的流动过程，包括形状变换和数学运算。

## 完整模型张量流程图

```mermaid
flowchart TD
    %% 输入层
    A["输入 Token IDs<br/>📦 [batch_size, seq_len]<br/>🔢 dtype: int64"] --> B["词嵌入层 (Embedding)<br/>📦 [batch_size, seq_len, hidden_size]<br/>🔢 dtype: float32"]
    
    %% 位置编码预计算
    B --> C["位置编码预计算<br/>🎯 precompute_freqs_cis()<br/>📦 cos: [max_seq_len, head_dim]<br/>📦 sin: [max_seq_len, head_dim]"]
    
    %% 多层Transformer块
    B --> D["Transformer 层 0"]
    C --> D
    
    subgraph Layer["🧠 MiniMindBlock 详细流程"]
        direction TB
        
        %% 层输入
        L1["层输入 x<br/>📦 [batch_size, seq_len, hidden_size]"] 
        
        %% 第一个RMSNorm
        L1 --> L2["RMSNorm (注意力前)<br/>📦 [batch_size, seq_len, hidden_size]<br/>🔬 rms = √(mean(x²) + ε)<br/>📊 output = x / rms * weight"]
        
        %% 注意力机制
        L2 --> L3["🎯 Attention 模块"]
        
        subgraph Attention["🎯 多头注意力详细流程"]
            direction TB
            
            %% Q, K, V投影
            A1["输入<br/>📦 [batch_size, seq_len, hidden_size]"] 
            A1 --> A2["Q 投影<br/>📦 [batch_size, seq_len, n_heads * head_dim]"]
            A1 --> A3["K 投影<br/>📦 [batch_size, seq_len, n_kv_heads * head_dim]"]
            A1 --> A4["V 投影<br/>📦 [batch_size, seq_len, n_kv_heads * head_dim]"]
            
            %% 重塑为多头格式
            A2 --> A5["Q reshape<br/>📦 [batch_size, seq_len, n_heads, head_dim]"]
            A3 --> A6["K reshape<br/>📦 [batch_size, seq_len, n_kv_heads, head_dim]"]
            A4 --> A7["V reshape<br/>📦 [batch_size, seq_len, n_kv_heads, head_dim]"]
            
            %% 应用RoPE位置编码
            A5 --> A8["应用 RoPE 到 Q<br/>🔄 apply_rotary_pos_emb()<br/>📦 [batch_size, seq_len, n_heads, head_dim]"]
            A6 --> A9["应用 RoPE 到 K<br/>🔄 apply_rotary_pos_emb()<br/>📦 [batch_size, seq_len, n_kv_heads, head_dim]"]
            A7 --> A10["V (无变化)<br/>📦 [batch_size, seq_len, n_kv_heads, head_dim]"]
            
            %% GQA: Key-Value重复
            A9 --> A11["K 重复 (GQA)<br/>🔁 repeat_kv()<br/>📦 [batch_size, seq_len, n_heads, head_dim]"]
            A10 --> A12["V 重复 (GQA)<br/>🔁 repeat_kv()<br/>📦 [batch_size, seq_len, n_heads, head_dim]"]
            
            %% KV缓存处理
            A11 --> A13{"KV 缓存?"}
            A12 --> A13
            A13 -->|是| A14["拼接历史 KV<br/>📦 K: [batch_size, total_len, n_heads, head_dim]<br/>📦 V: [batch_size, total_len, n_heads, head_dim]"]
            A13 -->|否| A15["使用当前 KV"]
            A14 --> A16["更新 KV 缓存"]
            A15 --> A16
            
            %% 注意力计算
            A8 --> A17["注意力分数计算<br/>🔢 scores = Q @ K.transpose(-2, -1)<br/>📦 [batch_size, n_heads, seq_len, total_len]"]
            A16 --> A17
            
            A17 --> A18["缩放<br/>🔢 scores = scores / √head_dim<br/>📦 [batch_size, n_heads, seq_len, total_len]"]
            
            A18 --> A19["应用因果掩码<br/>🎭 mask = tril(ones)<br/>🔢 scores = scores.masked_fill(mask==0, -inf)"]
            
            A19 --> A20["Softmax<br/>🔢 attn_weights = softmax(scores)<br/>📦 [batch_size, n_heads, seq_len, total_len]"]
            
            A20 --> A21["注意力输出<br/>🔢 out = attn_weights @ V<br/>📦 [batch_size, n_heads, seq_len, head_dim]"]
            
            A21 --> A22["重塑输出<br/>📦 [batch_size, seq_len, hidden_size]"]
            
            A22 --> A23["输出投影<br/>📦 [batch_size, seq_len, hidden_size]"]
        end
        
        L3 --> L4["残差连接<br/>🔗 x = x + attn_out<br/>📦 [batch_size, seq_len, hidden_size]"]
        
        %% 第二个RMSNorm
        L4 --> L5["RMSNorm (FFN前)<br/>📦 [batch_size, seq_len, hidden_size]"]
        
        %% 前馈网络分支
        L5 --> L6{"使用 MoE?"}
        
        %% 标准FFN分支
        L6 -->|否| L7["🔥 FeedForward 标准FFN"]
        
        subgraph FFN["🔥 标准前馈网络"]
            direction TB
            F1["输入<br/>📦 [batch_size, seq_len, hidden_size]"]
            F1 --> F2["Gate 投影<br/>📦 [batch_size, seq_len, intermediate_size]"]
            F1 --> F3["Up 投影<br/>📦 [batch_size, seq_len, intermediate_size]"]
            
            F2 --> F4["SiLU 激活<br/>🔢 silu(gate_proj)<br/>📦 [batch_size, seq_len, intermediate_size]"]
            
            F4 --> F5["逐元素相乘<br/>🔢 activated_gate * up_proj<br/>📦 [batch_size, seq_len, intermediate_size]"]
            F3 --> F5
            
            F5 --> F6["Down 投影<br/>📦 [batch_size, seq_len, hidden_size]"]
        end
        
        %% MoE分支
        L6 -->|是| L8["🎯 MOEFeedForward"]
        
        subgraph MoE["🎯 MoE前馈网络详细流程"]
            direction TB
            
            M1["输入<br/>📦 [batch_size, seq_len, hidden_size]"]
            M1 --> M2["重塑为token序列<br/>📦 [batch_size * seq_len, hidden_size]"]
            
            %% MoE门控
            M2 --> M3["🚪 MoE Gate"]
            
            subgraph Gate["🚪 MoE门控网络"]
                G1["输入<br/>📦 [total_tokens, hidden_size]"]
                G1 --> G2["门控投影<br/>📦 [total_tokens, n_experts]"]
                G2 --> G3["Softmax<br/>📦 [total_tokens, n_experts]"]
                G3 --> G4["TopK 选择<br/>📦 topk_idx: [total_tokens, k]<br/>📦 topk_weight: [total_tokens, k]"]
                G4 --> G5["权重归一化 (可选)<br/>📦 [total_tokens, k]"]
                G5 --> G6["辅助损失计算<br/>🔢 aux_loss = load_balance_loss"]
            end
            
            M3 --> M4["专家路由<br/>📊 根据topk_idx分发token"]
            
            %% 专家网络并行处理
            M4 --> M5["专家 0<br/>🧠 SwiGLU FFN"]
            M4 --> M6["专家 1<br/>🧠 SwiGLU FFN"]
            M4 --> M7["专家 ...<br/>🧠 SwiGLU FFN"]
            M4 --> M8["专家 N-1<br/>🧠 SwiGLU FFN"]
            
            M5 --> M9["专家输出聚合<br/>🔗 根据topk_weight加权求和"]
            M6 --> M9
            M7 --> M9
            M8 --> M9
            
            %% 共享专家（如果有）
            M2 --> M10["共享专家处理<br/>🤝 所有token都经过"]
            M10 --> M11["共享专家输出<br/>📦 [total_tokens, hidden_size]"]
            
            M9 --> M12["MoE + 共享专家<br/>🔗 路由输出 + 共享输出"]
            M11 --> M12
            
            M12 --> M13["重塑回序列<br/>📦 [batch_size, seq_len, hidden_size]"]
        end
        
        L7 --> L9["FFN残差连接<br/>🔗 x = x + ffn_out"]
        L8 --> L9
        L9 --> L10["层输出<br/>📦 [batch_size, seq_len, hidden_size]"]
    end
    
    %% 多层堆叠
    D --> E["Transformer 层 1<br/>📦 相同的张量流程"]
    E --> F["Transformer 层 2<br/>📦 ..."]
    F --> G["Transformer 层 N-1<br/>📦 最后一层"]
    
    %% 最终层归一化
    G --> H["最终 RMSNorm<br/>📦 [batch_size, seq_len, hidden_size]"]
    
    %% 语言模型头
    H --> I["语言模型头 (LM Head)<br/>📦 [batch_size, seq_len, vocab_size]<br/>🔢 logits = hidden_states @ lm_head.weight.T"]
    
    %% 输出
    I --> J["🎯 最终输出<br/>📦 logits: [batch_size, seq_len, vocab_size]<br/>🔢 可用于生成下一个token"]
    
    %% 样式定义
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef attentionStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef ffnStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef moeStyle fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef outputStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class A,B inputStyle
    class L3,Attention attentionStyle
    class L7,FFN ffnStyle
    class L8,MoE,M3,Gate moeStyle
    class I,J outputStyle
```

## RoPE 位置编码详细流程

```mermaid
flowchart TD
    subgraph RoPE["🔄 旋转位置编码 (RoPE) 详细流程"]
        direction TB
        
        R1["位置编码预计算<br/>🔢 θᵢ = 10000^(-2i/d) for i ∈ [0, d/2]<br/>📦 freqs: [max_seq_len]"] 
        
        R1 --> R2["计算角度<br/>🔢 angles = pos_ids[:, None] * freqs[None, :]<br/>📦 [seq_len, head_dim//2]"]
        
        R2 --> R3["计算 cos/sin<br/>🔢 cos_vals = cos(angles)<br/>🔢 sin_vals = sin(angles)<br/>📦 [seq_len, head_dim//2]"]
        
        R3 --> R4["扩展到完整维度<br/>📦 cos: [seq_len, head_dim]<br/>📦 sin: [seq_len, head_dim]"]
        
        R4 --> R5["应用到 Query/Key<br/>🔄 apply_rotary_pos_emb()"]
        
        subgraph Apply["应用旋转变换"]
            A1["输入张量 x<br/>📦 [batch_size, seq_len, n_heads, head_dim]"]
            A1 --> A2["分离奇偶维度<br/>📦 x1 = x[..., 0::2]  # 偶数位<br/>📦 x2 = x[..., 1::2]  # 奇数位"]
            
            A2 --> A3["旋转变换<br/>🔢 x1_new = x1 * cos - x2 * sin<br/>🔢 x2_new = x1 * sin + x2 * cos"]
            
            A3 --> A4["重新组合<br/>📦 输出: [batch_size, seq_len, n_heads, head_dim]<br/>🎯 保持向量模长不变"]
        end
        
        R5 --> Apply
    end
```

## MoE 门控网络详细流程

```mermaid
flowchart TD
    subgraph MoEGate["🚪 MoE门控网络详细张量变换"]
        direction TB
        
        MG1["输入<br/>📦 [batch_size, seq_len, hidden_size]<br/>🔢 例: [2, 16, 512]"]
        
        MG1 --> MG2["Flatten tokens<br/>📦 [batch_size * seq_len, hidden_size]<br/>🔢 例: [32, 512]"]
        
        MG2 --> MG3["门控投影<br/>🔢 gate_logits = input @ weight.T<br/>📦 [total_tokens, n_experts]<br/>🔢 例: [32, 8]"]
        
        MG3 --> MG4["Softmax 归一化<br/>🔢 probs = softmax(gate_logits)<br/>📦 [total_tokens, n_experts]<br/>🎯 每行和为1"]
        
        MG4 --> MG5["TopK 选择<br/>🔢 topk_values, topk_indices = torch.topk(probs, k)<br/>📦 topk_idx: [total_tokens, k]<br/>📦 topk_weight: [total_tokens, k]"]
        
        MG5 --> MG6{"权重归一化?"}
        MG6 -->|是| MG7["权重归一化<br/>🔢 topk_weight = topk_weight / topk_weight.sum(-1, keepdim=True)<br/>📦 [total_tokens, k]<br/>🎯 每行和为1"]
        MG6 -->|否| MG8["保持原权重"]
        MG7 --> MG9["辅助损失计算"]
        MG8 --> MG9
        
        subgraph AuxLoss["辅助损失计算"]
            AL1["专家频率统计<br/>📊 expert_counts = scatter_add(topk_indices)"]
            AL1 --> AL2["负载均衡损失<br/>🔢 aux_loss = α * CV(expert_counts)²<br/>🎯 鼓励专家负载均衡"]
        end
        
        MG9 --> MG10["输出<br/>📦 topk_idx: [total_tokens, k]<br/>📦 topk_weight: [total_tokens, k]<br/>📦 aux_loss: scalar"]
    end
```

## 注意力机制张量变换详图

```mermaid
flowchart TD
    subgraph AttentionDetail["🎯 注意力机制张量变换详图"]
        direction TB
        
        AD1["输入<br/>📦 [batch_size, seq_len, hidden_size]<br/>🔢 例: [4, 32, 768]"]
        
        AD1 --> AD2["QKV 投影"]
        
        subgraph QKV["QKV 投影详细"]
            Q1["Q 投影<br/>🔢 Q = input @ Wq<br/>📦 [4, 32, 768]<br/>🎯 n_heads=12, head_dim=64"]
            K1["K 投影 (GQA)<br/>🔢 K = input @ Wk<br/>📦 [4, 32, 384]<br/>🎯 n_kv_heads=6, head_dim=64"]  
            V1["V 投影 (GQA)<br/>🔢 V = input @ Wv<br/>📦 [4, 32, 384]<br/>🎯 n_kv_heads=6, head_dim=64"]
        end
        
        AD2 --> AD3["重塑为多头格式"]
        
        subgraph Reshape["重塑张量"]
            RS1["Q reshape<br/>📦 [4, 32, 12, 64]<br/>🔄 view(batch_size, seq_len, n_heads, head_dim)"]
            RS2["K reshape<br/>📦 [4, 32, 6, 64]<br/>🔄 view(batch_size, seq_len, n_kv_heads, head_dim)"]
            RS3["V reshape<br/>📦 [4, 32, 6, 64]<br/>🔄 view(batch_size, seq_len, n_kv_heads, head_dim)"]
        end
        
        AD3 --> AD4["应用 RoPE"]
        AD4 --> AD5["GQA Key-Value 重复"]
        
        subgraph GQA["GQA 重复机制"]
            GQ1["K 重复<br/>🔁 repeat_kv(K, n_heads//n_kv_heads)<br/>📦 [4, 32, 12, 64]<br/>🎯 每个KV头重复2次"]
            GQ2["V 重复<br/>🔁 repeat_kv(V, n_heads//n_kv_heads)<br/>📦 [4, 32, 12, 64]<br/>🎯 每个KV头重复2次"]
        end
        
        AD5 --> AD6["转置为注意力格式<br/>📦 Q: [4, 12, 32, 64]<br/>📦 K: [4, 12, 32, 64]<br/>📦 V: [4, 12, 32, 64]<br/>🔄 transpose(1, 2)"]
        
        AD6 --> AD7["计算注意力分数<br/>🔢 scores = Q @ K.transpose(-2, -1)<br/>📦 [4, 12, 32, 32]<br/>🎯 (seq_len, seq_len) 注意力矩阵"]
        
        AD7 --> AD8["缩放<br/>🔢 scores = scores / √64<br/>📦 [4, 12, 32, 32]<br/>🎯 防止梯度消失"]
        
        AD8 --> AD9["应用因果掩码<br/>🎭 mask = tril(ones(32, 32))<br/>🔢 scores.masked_fill(mask==0, -inf)<br/>🎯 确保因果性"]
        
        AD9 --> AD10["Softmax<br/>🔢 attn_weights = softmax(scores, dim=-1)<br/>📦 [4, 12, 32, 32]<br/>🎯 每行和为1"]
        
        AD10 --> AD11["加权求和<br/>🔢 output = attn_weights @ V<br/>📦 [4, 12, 32, 64]<br/>🎯 上下文向量"]
        
        AD11 --> AD12["重塑输出<br/>📦 [4, 32, 768]<br/>🔄 transpose(1,2).contiguous().view(...)"]
        
        AD12 --> AD13["输出投影<br/>🔢 output = output @ Wo<br/>📦 [4, 32, 768]"]
    end
```

## KV 缓存机制流程图

```mermaid
flowchart TD
    subgraph KVCache["💾 KV缓存机制详细流程"]
        direction TB
        
        KV1["首次推理<br/>📦 input_ids: [1, 64]<br/>🎯 编码阶段"]
        
        KV1 --> KV2["计算 K, V<br/>📦 K: [1, 64, n_kv_heads, head_dim]<br/>📦 V: [1, 64, n_kv_heads, head_dim]"]
        
        KV2 --> KV3["保存到缓存<br/>💾 past_key: [1, 64, n_kv_heads, head_dim]<br/>💾 past_value: [1, 64, n_kv_heads, head_dim]<br/>🎯 存储历史上下文"]
        
        KV3 --> KV4["正常注意力计算<br/>📦 attention_output: [1, 64, hidden_size]"]
        
        KV4 --> KV5["生成阶段<br/>📦 new_token: [1, 1]<br/>🎯 逐步生成"]
        
        KV5 --> KV6["计算新的 K, V<br/>📦 new_K: [1, 1, n_kv_heads, head_dim]<br/>📦 new_V: [1, 1, n_kv_heads, head_dim]"]
        
        KV6 --> KV7["拼接历史缓存<br/>🔗 K = cat([past_key, new_K], dim=1)<br/>🔗 V = cat([past_value, new_V], dim=1)<br/>📦 K: [1, 65, n_kv_heads, head_dim]<br/>📦 V: [1, 65, n_kv_heads, head_dim]"]
        
        KV7 --> KV8["更新缓存<br/>💾 past_key = K<br/>💾 past_value = V<br/>🎯 为下次生成准备"]
        
        KV8 --> KV9["高效注意力<br/>🎯 只需计算 Q @ K.T 一次<br/>📦 scores: [1, n_heads, 1, 65]<br/>⚡ 避免重复计算历史K,V"]
        
        KV9 --> KV10["持续生成<br/>🔄 重复步骤5-9<br/>📦 序列长度逐步增长<br/>⚡ 显著提升生成速度"]
        
        subgraph Performance["性能对比"]
            P1["无缓存: O(n²) 每步<br/>🐌 需要重新计算所有K,V"]
            P2["有缓存: O(n) 每步<br/>⚡ 只计算新token的K,V"]
            P1 -.-> P2
        end
    end
```

## 数据类型和设备管理

```mermaid
flowchart LR
    subgraph DataFlow["📊 数据类型和设备管理"]
        direction TB
        
        DT1["输入数据<br/>📦 input_ids: int64<br/>🖥️ device: cpu/cuda"]
        
        DT1 --> DT2["嵌入查找<br/>📦 embeddings: float32/float16<br/>🎯 转换为浮点数"]
        
        DT2 --> DT3["模型计算<br/>📊 保持数据类型一致性<br/>🖥️ 确保所有张量在同一设备"]
        
        DT3 --> DT4["混合精度 (可选)<br/>📊 forward: float16<br/>📊 loss: float32<br/>🎯 平衡速度和精度"]
        
        DT4 --> DT5["输出<br/>📦 logits: float32/float16<br/>📦 loss: float32"]
        
        subgraph Precision["🎯 精度管理"]
            PR1["FP32: 高精度<br/>🐌 较慢但数值稳定"]
            PR2["FP16: 快速推理<br/>⚡ 2倍速度提升"]
            PR3["BF16: 训练友好<br/>⚖️ 平衡速度和稳定性"]
        end
        
        subgraph Memory["💾 内存管理"]
            MEM1["梯度累积<br/>🔄 分批计算梯度"]
            MEM2["梯度检查点<br/>💾 重计算换内存"]
            MEM3["KV缓存<br/>⚡ 时间换空间"]
        end
    end
```

## 关键张量形状总结表

| 组件 | 输入形状 | 输出形状 | 关键参数 |
|------|----------|----------|----------|
| **Embedding** | `[B, L]` | `[B, L, H]` | vocab_size, hidden_size |
| **RMSNorm** | `[B, L, H]` | `[B, L, H]` | hidden_size, eps |
| **RoPE** | `[B, L, Nh, Dh]` | `[B, L, Nh, Dh]` | head_dim, max_seq_len |
| **Attention** | `[B, L, H]` | `[B, L, H]` | n_heads, n_kv_heads |
| **FeedForward** | `[B, L, H]` | `[B, L, H]` | hidden_size, intermediate_size |
| **MoE Gate** | `[B*L, H]` | `[B*L, K], [B*L, K], scalar` | n_experts, topk |
| **Expert FFN** | `[tokens, H]` | `[tokens, H]` | 每个专家独立的FFN |
| **LM Head** | `[B, L, H]` | `[B, L, V]` | hidden_size, vocab_size |

**符号说明:**
- B: batch_size (批次大小)
- L: seq_len (序列长度)  
- H: hidden_size (隐藏层维度)
- Nh: n_heads (注意力头数)
- Nkv: n_kv_heads (KV头数)
- Dh: head_dim (每个头的维度 = H/Nh)
- I: intermediate_size (FFN中间层维度)
- V: vocab_size (词汇表大小)
- K: num_experts_per_token (每token选择的专家数)
- E: n_routed_experts (总专家数)

这个详细的张量流程图展示了Zero2LLMV模型中每个组件的精确张量变换过程，包括形状变化、数学运算和关键的架构特性如MoE、GQA和KV缓存机制。