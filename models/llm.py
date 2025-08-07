import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from transformers.activations import ACT2FN

from configs.llmconfig import llmconfig

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    相比于LayerNorm，去除了平均值的计算，只保留方差的规一化
    计算公式: y = x / rms(x) * weight, 其中rms(x) = sqrt(mean(x^2))
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 防止除零的小常数
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def _norm(self, x):
        """计算RMS规一化"""
        # 计算均方根(RMS): sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Args:
            x: 输入张量，任意形状
        Returns:
            规一化后的张量，保持原始数据类型
        """
        # 将输入转为float32进行计算，然后转回原始类型，提高数值稳定性
        return self.weight * self._norm(x.float()).type_as(x)

#
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预计算旋转位置编码的频率
    
    Args:
        dim: 注意力头的维度
        end: 最大序列长度
        theta: RoPE的基础频率参数
    
    Returns:
        freqs_cos, freqs_sin: 余弦和正弦频率张量，形状为 (end, dim)
    """
    # 计算频率：1 / (theta ^ (2i/d)) for i in [0, dim//2)
    # 生成偶数位置的频率索引
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成位置索引 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # 计算每个位置和每个频率的乘积：pos * freq
    freqs = torch.outer(t, freqs).float()
    # 计算cos和sin值，并复制以匹配完整的head_dim
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    将旋转位置编码应用到query和key张量上
    
    Args:
        q: query张量，形状 (bsz, seq_len, num_heads, head_dim)
        k: key张量，形状 (bsz, seq_len, num_kv_heads, head_dim)
        cos: 余弦频率张量
        sin: 正弦频率张量
        position_ids: 位置ID（暂未使用）
        unsqueeze_dim: 需要增加维度的位置
    
    Returns:
        q_embed, k_embed: 应用了位置编码的query和key张量
    """
    def rotate_half(x):
        """将张量的后半部分取负并与前半部分交换位置"""
        # 将最后一个维度分成两半，后半部分取负号放到前面
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 应用旋转位置编码公式：x' = x*cos + rotate_half(x)*sin
    # 在指定维度上增加维度以便广播
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复key-value张量以匹配query头的数量（用于分组查询注意力）
    等效于 torch.repeat_interleave(x, dim=2, repeats=n_rep)
    
    Args:
        x: 输入张量，形状 (bsz, seq_len, num_kv_heads, head_dim)
        n_rep: 重复次数
    
    Returns:
        重复后的张量，形状 (bsz, seq_len, num_kv_heads * n_rep, head_dim)
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 在第4个维度插入新维度并扩展，然后重新整形
    # 这样每个kv头会被重复n_rep次
    return (
        x[:, :, :, None, :]  # 形状: (bs, slen, num_kv_heads, 1, head_dim)
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)  # 扩展新维度
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)  # 重新整形
    )


class Attention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention with Grouped Query Attention)
    支持:
    - 缓存机制 (KV Cache)
    - 旋转位置编码 (RoPE)
    - 分组查询注意力 (GQA) 或 多查询注意力 (MQA)
    - Flash Attention支持
    """
    def __init__(self, args: llmconfig):
        super().__init__()
        # 注意力头配置
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        self.n_local_heads = args.num_attention_heads      # query头数量
        self.n_local_kv_heads = args.num_key_value_heads  # key/value头数量  
        self.rep = self.n_local_heads // self.n_local_kv_heads  # 每个kv头对应的query头数量
        self.head_dim = args.hidden_dim // args.num_attention_heads  # 每个头的维度
        
        # 线性投影层
        self.q_proj = nn.Linear(args.hidden_dim, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_dim, args.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_dim, args.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_dim, bias=False)
        
        # Dropout层
        self.atten_dropout = nn.Dropout(args.dropout)  # 注意力权重dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 输出dropout
        self.dropout = args.dropout
        
        # 检查是否可以使用Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn


    def forward(
            self,
            x: torch.Tensor,  # 输入张量: (bsz, seq_len, hidden_dim)
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos_freqs, sin_freqs)
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 历史KV缓存
            use_cache: bool = False,  # 是否返回KV缓存
            attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码: (bsz, seq_len)
    ):
        # 获取输入张量的形状：(bsz, seq_len, hidden_dim)
        bsz, seq_len, _ = x.shape
        
        # 通过线性投影层计算query、key、value
        xq = self.q_proj(x)  # (bsz, seq_len, n_local_heads * head_dim)
        xk = self.k_proj(x)  # (bsz, seq_len, n_local_kv_heads * head_dim)
        xv = self.v_proj(x)  # (bsz, seq_len, n_local_kv_heads * head_dim)

        # 重新整形为多头注意力的格式
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用一下 旋转位置编码
        cos, sin = position_embeddings
        # 应用旋转位置编码，增强模型的位置感知能力
        # xq: (bsz, seq_len, n_local_heads, head_dim)
        # xk: (bsz, seq_len, n_local_kv_heads, head_dim)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len], position_ids=None, unsqueeze_dim=1)

        # KV缓存机制：将历史的key和value与当前的拼接
        # 这样可以在生成过程中重复使用之前计算的注意力状态，提高效率
        if past_key_value is not None:
            # 在序列长度维度上拼接历史和当前的key
            # past_key_value[0]: (bsz, past_seq_len, n_local_kv_heads, head_dim)
            # xk: (bsz, seq_len, n_local_kv_heads, head_dim)
            xk = torch.cat([
                past_key_value[0],
                xk
            ], dim=1)

            # 同样拼接value张量
            # past_key_value[1]: (bsz, past_seq_len, n_local_kv_heads, head_dim)
            # xv: (bsz, seq_len, n_local_kv_heads, head_dim)
            xv = torch.cat([
                past_key_value[1],
                xv
            ], dim=1)
        # 如果需要缓存，保存当前的key和value用于下一次前向传播
        past_kv = (xk, xv) if use_cache else None

        # 重新排列张量维度以适应注意力计算
        # 将seq_len和num_heads维度交换: (bsz, seq_len, num_heads, head_dim) -> (bsz, num_heads, seq_len, head_dim)
        xq, xk, xv = (
            xq.transpose(1, 2),  # query: (bsz, n_local_heads, seq_len, head_dim)
            repeat_kv(xk, self.rep).transpose(1, 2),  # key: (bsz, n_local_heads, total_seq_len, head_dim)
            repeat_kv(xv, self.rep).transpose(1, 2)   # value: (bsz, n_local_heads, total_seq_len, head_dim)
        )
        # Flash Attention分支（目前禁用）
        # Flash Attention可以显著降低内存使用和加速计算
        if False and self.flash and seq_len != 1:
            # 设置训练时的dropout概率
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attn_mask is not None:
                # 扩展注意力掩码维度
                attention_mask = attn_mask.view(bsz, 1, 1, -1).expand(
                    bsz, self.n_local_heads, seq_len, -1
                )
                attention_mask = attention_mask.bool()

            # 使用PyTorch原生Flash Attention
            output = F.scaled_dot_product_attention(
                xq, xk, xv, 
                attn_mask=attention_mask, 
                dropout_p=dropout_p, 
                is_causal=True  # 启用因果掩码
            )
        else:
            # 手动实现注意力机制
            # 计算注意力分数: Q * K^T / sqrt(d_k)
            # scores: (bsz, n_local_heads, seq_len, total_seq_len)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 添加因果掩码，确保当前位置只能看到之前的位置（自回归特性）
            # 创建上三角掩码矩阵，对角线上方设为负无穷
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1  # 对角线上方的元素设为-inf
            ).unsqueeze(0).unsqueeze(0)  # 扩展维度以匹配scores的形状
            scores = scores + causal_mask

            # 添加用户提供的注意力掩码（如padding mask）
            if attention_mask is not None:
                # 扩展掩码维度以匹配scores张量
                # attention_mask: (bsz, seq_len) -> (bsz, 1, 1, seq_len)
                extended_attention_mask = attention_mask.unsqueeze(-1).unsqueeze(2)
                # 将掩码值转换：1->0（可见），0->-1e9（不可见）
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # 计算注意力权重并应用到value上
            # 对最后一个维度（key的序列长度）进行softmax归一化
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
            attn_weights = self.atten_dropout(attn_weights)  # 应用dropout防止过拟合
            # 注意力权重与value相乘得到最终输出
            # output: (bsz, n_local_heads, seq_len, head_dim)
            output = attn_weights @ xv

        # 重新排列输出张量的维度
        # (bsz, n_local_heads, seq_len, head_dim) -> (bsz, seq_len, n_local_heads, head_dim)
        # 然后reshape为 (bsz, seq_len, hidden_dim)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 通过输出投影层并应用残差dropout
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv

class FeedForward(nn.Module):
    """
    SwiGLU前馈神经网络
    使用SwiGLU激活函数: SwiGLU(x) = Swish(gate_proj(x)) * up_proj(x)
    这种结构在许多大型语言模型中显示出优越性能
    """
    def __init__(self, config: llmconfig):
        super().__init__()
        # 计算中间层维度（经验值：~8/3 * hidden_size）
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 对齐64的倍数，优化GPU内存访问
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # SwiGLU的三个投影层
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 门控投影
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)   # 输出投影
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)     # 上采样投影
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数（通常为gelu或swish）

    def forward(self, x: torch.Tensor):
        """
        SwiGLU的前向传播
        
        Args:
            x: 输入张量 (bsz, seq_len, hidden_size)
            
        Returns:
            输出张量 (bsz, seq_len, hidden_size)
        """
        # SwiGLU: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        # 1. gate_proj(x) -> (bsz, seq_len, intermediate_size)
        # 2. act_fn(门控) -> 激活后的门控信号
        # 3. up_proj(x) -> (bsz, seq_len, intermediate_size) 
        # 4. 门控 * 上采样 -> 元素级乘法
        # 5. down_proj -> 投影回原始维度
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))





