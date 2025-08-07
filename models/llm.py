import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.nn.functional import dropout
import torch.nn.functional as F

from configs.llmconfig import llmconfig

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

#
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: llmconfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        self.n_local_heads = args.numattention_heads
        self.n_local_kv_heads = args.num_key_value_heads
        self.rep = self.n_local_heads // self.n_local_kv_heads # 为了
        self.head_dim = args.hidden_dim // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_dim, args.num_attention_heads * self.head_dim, bias=False) # 保证输出形状
        self.k_proj = nn.Linear(args.hidden_dim, args.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_dim, args.num_attention_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_dim, bias=False)
        self.atten_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 检查是否有flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn


    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        # x: (bsz, seq_len, hidden_dim)
        bsz, seq_len, _ = x.shape
        # 过线性层
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # xq:(bsz, seq_len, num_attn_heads * hidden_dim)
        # xk...

        # 展开一下
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用一下 旋转位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len], position_ids=None, unsqueeze_dim=self.head_dim)

        # kv_cache
        if past_key_value is not None:
            xk = torch.cat([
                past_key_value[0],
                xk
            ], dim=-1
            )

            xv = torch.cat([
                past_key_value[1],
                xv
            ], dim=-1)

        #
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if False and self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attn_mask is not None:
                attention_mask = attn_mask.view(bsz, 1, 1, -1).expand(
                    bsz, self.n_local_heads, seq_len, -1
                )
                attention_mask = attention_mask.bool()

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attention_mask, dropout_p=dropout_p, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)









