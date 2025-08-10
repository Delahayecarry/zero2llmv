import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Union
from transformers import PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
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
        # 确保num_key_value_heads不为None且大于0
        if args.num_key_value_heads is None or args.num_key_value_heads <= 0:
            self.num_key_value_heads = args.num_attention_heads
        else:
            self.num_key_value_heads = args.num_key_value_heads
            
        self.n_local_heads = args.num_attention_heads      # query头数量
        self.n_local_kv_heads = self.num_key_value_heads  # key/value头数量
        
        # 确保query头数量是kv头数量的整数倍（用于分组查询注意力）
        assert self.n_local_heads % self.n_local_kv_heads == 0, \
            f"num_attention_heads ({self.n_local_heads}) must be divisible by num_key_value_heads ({self.n_local_kv_heads})"
            
        self.rep = self.n_local_heads // self.n_local_kv_heads  # 每个kv头对应的query头数量
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个头的维度
        
        # 线性投影层
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 修复：使用正确的kv头数量
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 修复：使用正确的kv头数量
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        # Dropout层
        self.atten_dropout = nn.Dropout(args.dropout)  # 注意力权重dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 输出dropout
        self.dropout = args.dropout
        
        # 检查是否可以使用Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn


    def forward(
            self,
            x: torch.Tensor,  # 输入张量: (bsz, seq_len, hidden_size)
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos_freqs, sin_freqs)
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 历史KV缓存
            use_cache: bool = False,  # 是否返回KV缓存
            attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码: (bsz, seq_len)
    ):
        # 获取输入张量的形状：(bsz, seq_len, hidden_size)
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
            # 获取总的key序列长度（包括历史KV缓存）
            total_seq_len = xk.shape[2]  # xk的形状是(bsz, n_heads, total_seq_len, head_dim)
            
            if total_seq_len > seq_len:
                # 使用KV缓存的情况：当前query只能看到所有历史key + 当前及之前的key
                # 创建因果掩码：(seq_len, total_seq_len)
                causal_mask = torch.zeros((seq_len, total_seq_len), device=scores.device)
                # 对于当前序列的每个位置i，它可以看到：
                # 1. 所有历史位置：[0, total_seq_len - seq_len)
                # 2. 当前序列中的位置：[total_seq_len - seq_len, total_seq_len - seq_len + i]
                for i in range(seq_len):
                    causal_mask[i, total_seq_len - seq_len + i + 1:] = float("-inf")
            else:
                # 标准情况：没有KV缓存，创建标准的上三角掩码
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                    diagonal=1
                )
            
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # 扩展维度以匹配scores的形状
            scores = scores + causal_mask

            # 添加用户提供的注意力掩码（如padding mask）
            if attention_mask is not None:
                # 扩展掩码维度以匹配scores张量
                # attention_mask可能是 (bsz, total_seq_len) 或 (bsz, seq_len)
                # 我们需要处理KV缓存情况下的序列长度不匹配
                total_seq_len = xk.shape[1]  # 获取实际的key序列长度
                
                if attention_mask.shape[1] != total_seq_len:
                    # 如果掩码长度与key长度不匹配，需要调整
                    # 通常在使用KV缓存时会发生这种情况
                    if attention_mask.shape[1] < total_seq_len:
                        # 扩展掩码以匹配总长度（历史+当前）
                        pad_length = total_seq_len - attention_mask.shape[1]
                        attention_mask = F.pad(attention_mask, (0, pad_length), value=1.0)
                
                # attention_mask: (bsz, total_seq_len) -> (bsz, 1, seq_len, total_seq_len)
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                # 扩展到匹配scores的维度：(bsz, n_heads, seq_len, total_seq_len)
                extended_attention_mask = extended_attention_mask.expand(bsz, self.n_local_heads, seq_len, total_seq_len)
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
        # 然后reshape为 (bsz, seq_len, hidden_size)
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


class MoEGate(nn.Module):
    """
    专家混合(Mixture of Experts)门控网络
    
    负责决定每个token应该使用哪些专家以及各专家的权重分配。
    门控网络是MoE架构的核心组件，它通过学习如何路由不同的输入
    到最适合处理该类型输入的专家模型上。
    
    工作原理：
    1. 输入token经过线性变换得到专家评分
    2. 使用softmax获得各专家概率分布
    3. 选择top-k个最优专家及其权重
    4. 计算负载均衡的辅助损失
    """
    def __init__(self, config: llmconfig):
        super().__init__()
        self.config = config
        # 每个token选择的专家数量(通常为1-8)
        self.topk = config.num_experts_per_token
        # 可路由专家的总数量(通常为8-64个专家)
        self.n_routed_experts = config.n_routed_experts

        # 专家评分函数类型(softmax/gumbel_softmax等)
        self.scoring_func = config.scoring_func
        # 辅助损失的权重系数，用于负载均衡
        self.alpha = config.aux_loss_alpha
        # 是否使用序列级辅助损失计算
        self.seq_aux = config.seq_aux

        # 是否对top-k概率进行归一化，确保权重和为1
        self.norm_topk_prob = config.norm_topk_prob
        # 门控网络的输入维度，通常等于模型隐藏层维度
        self.gating_dim = config.hidden_size
        # 门控权重矩阵：(专家数量, 隐藏维度)
        # 每一行代表一个专家的门控向量
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        初始化门控权重参数
        使用Kaiming均匀分布初始化，适合ReLU类激活函数
        """
        import torch.nn.init as init
        # 使用Kaiming均匀分布初始化权重，保证梯度方差稳定
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))


    def forward(self, x: torch.Tensor):
        """
        门控网络的前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, hidden_size)
            
        Returns:
            topk_idx: 选中的专家索引，形状 (batch_size*seq_len, topk)
            topk_weight: 对应专家的权重，形状 (batch_size*seq_len, topk)
            aux_loss: 负载均衡的辅助损失值
        """
        bsz, seq_len, hidden_size = x.shape

        # 将输入重整为二维：(batch_size*seq_len, hidden_size)
        # 这样每个token都会独立进行专家选择
        x = x.view(-1, hidden_size)
        
        # 计算每个token对所有专家的亲和度得分
        # F.linear(input, weight, bias) 等价于 input @ weight.T + bias
        # 这里weight形状为(n_experts, hidden_size)，所以输出为(tokens, n_experts)
        logits = F.linear(x, self.weight, None)
        
        # 将专家得分转换为概率分布
        if self.scoring_func == "softmax":
            # 使用softmax确保所有专家权重和为1
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'{self.scoring_func} is not implemented')

        # 为每个token选择top-k个最优专家
        # sorted=False: 不对结果排序，节省计算资源
        # 返回值：权重和对应的专家索引
        topk_weight, topk_idx = torch.topk(scores, k=self.topk, dim=1, sorted=False)
        
        # 归一化top-k权重，确保选中专家的权重和为1
        # 这对于多专家情况很重要，保证权重分布的合理性
        if self.topk > 1 and self.norm_topk_prob:
            # 加上很小的常数避免除零错误
            denominator = topk_weight.sum(dim=1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 计算负载均衡的辅助损失
        # 目的：防止所有token都选择相同的专家，促进专家使用的均匀分布
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.topk
            # 重新整形为按batch组织的索引：(batch_size, seq_len)
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                # 序列级辅助损失：在每个序列内部计算负载均衡
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # 统计每个batch中每个专家被选择的次数
                ce = torch.zeros(bsz, self.n_routed_experts, device=x.device)
                # 修复：确保索引维度正确匹配
                # topk_idx_for_aux_loss: (bsz, seq_len * topk) -> (bsz, seq_len, topk) -> (bsz, -1)
                idx_flat = topk_idx_for_aux_loss.view(bsz, -1)  # 确保是 (bsz, seq_len*topk)
                ones_flat = torch.ones(bsz, idx_flat.shape[1], device=x.device)  # 匹配的权重张量
                ce.scatter_add_(1, idx_flat, ones_flat).div_(
                    seq_len * aux_topk / self.n_routed_experts  # 归一化因子
                )
                # 辅助损失 = 专家使用频率 * 专家门控概率的平均值
                # 鼓励高概率的专家被更频繁使用，低概率的专家使用频率降低
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 全局辅助损失：在整个batch上计算负载均衡
                # 创建one-hot编码表示专家选择情况
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1),
                    num_classes=self.n_routed_experts
                )
                # ce: 每个专家被选择的频率 (importance)
                ce = mask_ce.float().mean(0)
                # Pi: 每个专家的平均门控概率 (load)
                Pi = scores_for_aux.mean(0)
                # 负载均衡损失：importance * load，鼓励均匀分布
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0

        return topk_idx, topk_weight, aux_loss

class MOEFeedForward(nn.Module):
    """
    专家混合前馈网络 (Mixture of Experts Feed Forward Network)
    
    与Transformer中的标准FFN不同，MoE FFN包含多个并行的专家网络。
    每个token通过门控机制选择部分专家进行处理，从而在不显著增加计算成本的
    情况下大幅提升模型容量。
    
    架构特点：
    - 路由专家：根据输入动态选择的专家
    - 共享专家：所有token都会使用的专家，保证基础能力
    - 负载均衡：通过辅助损失促进专家使用均衡
    """
    def __init__(self, config: llmconfig):
        super().__init__()
        self.config = config
        # 创建可路由专家网络列表，每个专家都是一个独立的FFN
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 门控网络，负责决定每个token使用哪些专家
        self.gate = MoEGate(config)
        # 共享专家：所有token都会使用的专家，提供基础能力保证
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        """
        MoE前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, hidden_size)
            
        Returns:
            输出张量，形状 (batch_size, seq_len, hidden_size)
        """
        # 保存原始输入，用于共享专家的残差连接
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # 使用门控机制选择专家，获得专家索引、权重和辅助损失
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # 将输入重整为二维：(batch_size*seq_len, hidden_size)
        x = x.view(-1, x.shape[-1])
        # 将专家索引展平，便于处理
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            # 训练模式：使用所有选中的专家
            # 为每个token复制num_experts_per_token份，对应不同的专家
            x = x.repeat_interleave(self.config.num_experts_per_token, dim=0)
            # 初始化输出张量，使用float16节省内存
            y = torch.empty_like(x, dtype=torch.float16)
            
            # 遍历所有专家，让对应的token通过对应的专家
            for i, expert in enumerate(self.experts):
                # 只处理分配给当前专家的token
                mask = (flat_topk_idx == i)
                if mask.any():  # 只有当有token分配给该专家时才计算
                    y[mask] = expert(x[mask]).to(y.dtype)  # 确保类型一致
            
            # 将专家输出按权重加权平均，得到最终输出
            # 重整为(batch_size*seq_len, num_experts_per_token, hidden_size)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # 恢复原始形状
            y = y.view(*orig_shape)
        else:
            # 推理模式：优化的专家路由策略
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 添加共享专家的输出（如果存在）
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                # 共享专家使用原始输入，所有token都会通过共享专家
                y = y + expert(identity)
        
        # 保存辅助损失，用于后续的损失计算
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        MoE推理模式的优化实现
        
        在推理时，为了提高效率，我们按专家组织token，让每个专家批量处理分配给它的所有token。
        这样可以最大化GPU利用率并减少内存消耗。
        
        Args:
            x: 输入token，形状 (total_tokens, hidden_size)
            flat_expert_indices: 每个token对应的专家索引
            flat_expert_weights: 每个token对应的专家权重
            
        Returns:
            expert_cache: 专家输出的加权和，形状 (total_tokens, hidden_size)
        """
        # 初始化输出缓存
        expert_cache = torch.zeros_like(x)
        
        # 按专家索引排序，这样相同专家的token会聚集在一起
        idxs = flat_expert_indices.argsort()
        
        # 计算每个专家分配到的token数量的累积和
        # bincount: 统计每个专家被分配到的token数量
        # cumsum: 计算累积和，得到每个专家的结束位置
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        
        # 计算每个专家对应的token索引（考虑每个token可能被多个专家处理）
        token_idxs = idxs // self.config.num_experts_per_token
        
        # 遍历每个专家，批量处理分配给该专家的所有token
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前专家的起始和结束位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue  # 该专家没有分配到token，跳过
            
            expert = self.experts[i]
            # 获取当前专家需要处理的token索引
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 提取对应的token
            expert_tokens = x[exp_token_idx]
            # 通过专家网络处理token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 乘以对应的专家权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将专家输出积累到缓存中对应的token位置
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

class LLMBlock(nn.Module):
    """
    Transformer解码器层 (Transformer Decoder Layer)
    
    每个解码器层包含：
    1. 多头自注意力机制 (Multi-Head Self-Attention)
    2. 前馈网络 (Feed Forward Network) 或 专家混合网络 (MoE)
    3. 层归一化 (RMSNorm) 和 残差连接 (Residual Connection)
    
    这是现代大语言模型的核心组件，采用Pre-Norm的设计模式。
    """
    def __init__(self, layer_id: int, config: llmconfig):
        super().__init__()
        # 基本模型参数
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # 多头自注意力机制
        self.self_attn = Attention(config)

        # 层ID，用于区分不同的层
        self.layer_id = layer_id
        
        # Pre-Norm设计：在注意力和前馈网络之前进行归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 根据配置选择使用标准FFN或MoE FFN
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        Transformer层的前向传播
        
        Args:
            hidden_states: 输入隐藏状态，形状 (batch_size, seq_len, hidden_size)
            position_embeddings: 位置编码，包含(cos_freqs, sin_freqs)
            past_key_value: 历史KV缓存，用于生成任务的加速
            use_cache: 是否保存KV缓存供下次使用
            attention_mask: 注意力掉码，主要用于处理padding
            
        Returns:
            hidden_states: 输出隐藏状态，形状保持不变
            present_key_value: 当前KV缓存，用于下次生成
        """
        # 保存残差连接的输入
        residual = hidden_states
        
        # 第一个子层：Pre-Norm + 多头自注意力 + 残差连接
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # Pre-Norm设计
            position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        # 残差连接：将注意力输出与原始输入相加
        hidden_states += residual
        
        # 第二个子层：Pre-Norm + FFN/MoE + 残差连接
        # 这里使用了简化写法，直接在一行内完成残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        
        return hidden_states, present_key_value


class LLM(nn.Module):
    """
    大语言模型主体架构 (Large Language Model Core)
    
    这是一个基于Transformer的解码器架构，包含以下核心组件：
    1. 词嵌入层 (Token Embeddings)：将输入token转换为高维向量表示
    2. 多个解码器层 (Decoder Layers)：包含自注意力机制和前馈网络的堆叠
    3. 最终层归一化 (Final Layer Norm)：对最后的隐藏状态进行归一化
    4. 旋转位置编码 (RoPE)：增强模型的位置感知能力
    
    支持的高级特性：
    - KV缓存 (KV Cache)：在生成任务中复用历史计算结果
    - 专家混合 (MoE)：通过条件计算提升模型容量
    - 分组查询注意力 (GQA)：在保持性能的同时降低推理成本
    """
    def __init__(self, config: llmconfig):
        super().__init__()
        # 存储模型配置参数
        self.config = config
        # 词汇表大小和隐藏层数量
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        
        # 词嵌入层：将token ID转换为dense向量表示
        # 形状：(vocab_size, hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Dropout层：防止过拟合，在训练时随机设置部分神经元为0
        self.dropout = nn.Dropout(config.dropout)
        
        # 构建多层Transformer解码器
        # 每层包含自注意力机制和前馈网络（或MoE）
        self.layers = nn.ModuleList([LLMBlock(l, config) for l in range(self.num_hidden_layers)])
        
        # 最终层归一化：稳定训练过程，改善梯度流动
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算旋转位置编码 (RoPE) 的频率
        # RoPE能够有效编码位置信息，支持更长的序列长度
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,  # 每个注意力头的维度
            end=config.max_position_embeddings,  # 支持的最大序列长度
            theta=config.rope_theta  # RoPE的基础频率参数
        )
        
        # 将频率注册为缓冲区，不参与梯度更新但会随模型保存/加载
        # persistent=False：在state_dict中不保存，推理时重新计算
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """
        LLM的前向传播过程
        
        Args:
            input_ids: 输入的token ID序列，形状 (batch_size, seq_length)
            attention_mask: 注意力掩码，标识有效token和padding，形状 (batch_size, seq_length)
                           1表示有效token，0表示padding token
            past_key_values: 历史的KV缓存，用于生成任务的增量解码
                           列表长度为层数，每个元素是(key_cache, value_cache)的元组
            use_cache: 是否保存当前计算的KV状态供下次使用（生成模式必需）
            **kwargs: 其他可选参数
            
        Returns:
            hidden_states: 最终的隐藏状态，形状 (batch_size, seq_length, hidden_size)
            presents: 当前层的KV缓存列表，用于下次前向传播
            aux_loss: MoE的辅助损失（如果使用MoE），用于负载均衡
        """
        # 获取输入维度信息
        batch_size, seq_length = input_ids.shape
        
        # 初始化KV缓存：如果没有提供历史缓存，为每层创建None占位符
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # 计算当前序列在整个对话中的起始位置
        # 如果有KV缓存，起始位置 = 历史缓存的长度；否则从位置0开始
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 词嵌入：将token ID转换为向量表示，并应用dropout正则化
        # input_ids: (batch_size, seq_length) -> hidden_states: (batch_size, seq_length, hidden_size)
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 准备位置编码：从预计算的频率中选取当前序列对应的部分
        # 支持KV缓存：start_pos确保位置编码的连续性
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],  # 当前序列的余弦频率
            self.freqs_sin[start_pos:start_pos + seq_length]   # 当前序列的正弦频率
        )

        # 初始化存储当前层KV缓存的列表
        presents = []
        
        # 逐层前向传播：通过所有Transformer解码器层
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # 每一层的前向传播包括：
            # 1. 多头自注意力机制（使用历史KV缓存加速）
            # 2. 前馈网络（FFN或MoE）
            # 3. 残差连接和层归一化
            hidden_states, present = layer(
                hidden_states,          # 当前隐藏状态
                position_embeddings,    # 位置编码信息
                past_key_value=past_key_value,  # 该层的历史KV缓存
                use_cache=use_cache,    # 是否保存当前KV状态
                attention_mask=attention_mask   # 注意力掩码
            )
            # 收集当前层的KV缓存，用于下次生成
            presents.append(present)

        # 最终层归一化：标准化输出隐藏状态，稳定数值
        hidden_states = self.norm(hidden_states)

        # 计算MoE的辅助损失（如果使用了MoE层）
        # 辅助损失用于促进专家负载均衡，防止少数专家承担大部分计算
        aux_loss = sum(
            layer.mlp.aux_loss          # 获取每层MoE的辅助损失
            for layer in self.layers    # 遍历所有层
            if isinstance(layer.mlp, MOEFeedForward)  # 仅考虑MoE层
        )

        return hidden_states, presents, aux_loss


class CausalLM(PreTrainedModel, GenerationMixin):
    """
    因果语言模型 (Causal Language Model)
    
    这是一个自回归的语言模型，用于文本生成任务。它继承自HuggingFace的PreTrainedModel，
    具备以下核心功能：
    
    1. 因果建模：模型只能看到当前位置之前的token，无法看到未来的token
    2. 文本生成：支持各种解码策略（贪婪搜索、束搜索、采样等）
    3. 增量生成：利用KV缓存机制提高生成效率
    4. 权重共享：输入词嵌入与输出投影层共享参数，减少参数量
    
    架构组成：
    - LLM模型主体：处理输入序列，输出隐藏状态
    - 语言模型头：将隐藏状态投影到词汇表大小，计算下一个token的概率分布
    
    适用场景：
    - 文本续写和补全
    - 对话系统
    - 代码生成
    - 创意写作等自然语言生成任务
    """
    # 指定配置类，用于模型的加载和保存
    config_class = llmconfig

    def __init__(self, config: llmconfig = None):
        """
        初始化因果语言模型
        
        Args:
            config: 模型配置对象，如果为None则使用默认配置
        """
        # 设置配置，如果未提供则使用默认配置
        self.config = config or llmconfig()
        # 调用父类初始化，注册模型为HuggingFace兼容的PreTrainedModel
        super().__init__(self.config)
        
        # 创建模型主体：负责编码输入序列并输出隐藏状态
        self.model = LLM(self.config)
        
        # 语言模型头：将隐藏状态映射到词汇表概率分布
        # 不使用偏置项，减少参数量并提高数值稳定性
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # 权重共享 (Weight Tying)：输入词嵌入层与输出投影层共享参数
        # 这种设计的优势：
        # 1. 显著减少参数数量（约减少vocab_size * hidden_size个参数）
        # 2. 提高训练效率和泛化能力
        # 3. 在词汇表大的模型中特别有效
        self.model.embed_tokens.weight = self.lm_head.weight
        
        # 初始化输出对象，用于存储模型前向传播的结果
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        因果语言模型的前向传播
        
        Args:
            input_ids: 输入的token ID序列，形状 (batch_size, seq_length)
            attention_mask: 注意力掩码，形状 (batch_size, seq_length)
            past_key_values: 历史KV缓存，用于增量生成
            use_cache: 是否保存KV缓存供下次使用
            logits_to_keep: 控制输出logits的范围，用于内存优化
                          - int类型：保留最后N个位置的logits
                          - Tensor类型：指定具体位置的索引
            **args: 其他传递给底层模型的参数
            
        Returns:
            CausalLMOutputWithPast对象，包含：
            - logits: 下一个token的概率分布，形状 (batch_size, kept_seq_len, vocab_size)
            - past_key_values: 当前的KV缓存
            - last_hidden_state: 最终隐藏状态
            - aux_loss: MoE辅助损失（如果使用MoE）
        """
        # 通过LLM主体获得隐藏状态、KV缓存和辅助损失
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        # 选择需要计算logits的序列位置，用于内存优化
        # 在生成过程中，通常只需要最后一个位置的logits
        if isinstance(logits_to_keep, int):
            # 如果是整数，取最后N个位置（N=logits_to_keep）
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            # 如果是张量，直接用作索引
            slice_indices = logits_to_keep
            
        # 通过语言模型头计算词汇表上的logits
        # h[:, slice_indices, :]: 选择指定位置的隐藏状态
        # logits: (batch_size, kept_positions, vocab_size)
        logits = self.lm_head(h[:, slice_indices, :])
        
        # 填充输出对象的各个字段
        self.OUT.__setitem__('last_hidden_state', h)           # 完整的隐藏状态
        self.OUT.__setitem__('logits', logits)                 # 输出logits
        self.OUT.__setitem__('aux_loss', aux_loss)             # MoE辅助损失  
        self.OUT.__setitem__('past_key_values', past_kvs)      # KV缓存
        
        return self.OUT






