import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            attention_heads: int,
            conv1d_filter_size: int,
            conv1d_kernel_size: int,
            conv_dropout: float,
            attn_weights_dropout: float,
            attn_out_dropout: float,
            rope_base: float,
            rope_max_seq_len: int,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.multi_head_attention = MultiHeadSelfAttention(
            hidden_dim,
            attention_heads,
            attn_weights_dropout,
            rope_base,
            rope_max_seq_len,
        )
        self.norm2 = RMSNorm(hidden_dim)
        self.conv1 = Conv1D(hidden_dim, conv1d_filter_size, conv1d_kernel_size)
        self.conv2 = Conv1D(conv1d_filter_size, hidden_dim, 1)
        self.attn_out_dropout = nn.Dropout(attn_out_dropout)
        self.conv_dropout = nn.Dropout(conv_dropout)

    def forward(
            self,
            x,    # [B, T, D]
            mask  # [B, T, 1] bool
    ):
        float_mask = mask.to(x.dtype)

        attn_out = self.multi_head_attention(self.norm1(x), mask)
        x = x + self.attn_out_dropout(attn_out)
        x = x * float_mask

        ffn_out = self.conv1(self.norm2(x), mask)
        ffn_out = F.silu(ffn_out)
        ffn_out = self.conv2(ffn_out, mask)

        x = x + self.conv_dropout(ffn_out)
        x = x * float_mask

        return x

class RMSNorm(nn.Module):
    """x_norm = x / sqrt(mean(x^2) + eps) * gamma"""
    def __init__(
            self,
            hidden_dim: int,
            eps: float = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim)) # gamma

    def forward(
            self,
            x       # [B, T, D]
    ):
        dtype = x.dtype
        x_f = x.float() # upcast to fp32 for stable mean(x^2)
        rms_recip = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f * rms_recip).to(dtype) * self.weight


"""
##### RoPE #####
m: token position (0, 1, 2, ...)   d: head dim (even)   i: pair index (0 .. d/2-1)

theta_i = 10000^(-2i/d)    # base frequency for pair i
alpha   = m * theta_i      # rotation angle for position m, pair i

(x, y) = (x_m_i, y_m_i)
x' = x*cos(alpha) - y*sin(alpha)
y' = x*sin(alpha) + y*cos(alpha)

Precompute cos/sin as [max_seq_len, d/2] (e.g. head_dim=64 → 32 pairs); at runtime slice
to seq_len and rotate. Follows Karpathy's nanoChat: RoPE on contiguous halves (sliced),
not interleaved pairs → avoids rotate_half shuffle.
"""

class RotaryEmbedding(nn.Module):
    def __init__(
            self, 
            head_dim: int, 
            base: float = 10000.0, 
            max_seq_len: int = 3000
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        cos, sin = self._precompute_rotary_embeddings(head_dim, base, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(
            self, 
            head_dim: int, 
            base: float, 
            max_seq_len: int
    ):
        pair_indices = torch.arange(0, head_dim, 2, dtype=torch.float32) # equals already 2i
        thetas = 1.0 / (base ** (pair_indices / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        
        alphas = torch.outer(positions, thetas) # [max_seq_len x head_dim/2]
        
        cos = alphas.cos()
        sin = alphas.sin()

        cos = rearrange(cos, 't d -> 1 1 t d') # [1, 1, max_seq_len, head_dim/2])
        sin = rearrange(sin, 't d -> 1 1 t d') # [1, 1, max_seq_len, head_dim/2]
        
        return cos, sin

    def forward(
            self, 
            seq_len: int
    ):
        return self.cos[:, :, :seq_len, :], self.sin[:, :, :seq_len, :] # each [1, 1, seq_len, head_dim/2]

def apply_rotary_embeddings(
        x,    # [B, num_heads, T, head_dim]
        cos,  # [1, 1, T, head_dim/2]
        sin,  # [1, 1, T, head_dim/2]
):
    x1, x2 = x.chunk(2, dim=-1)  # each [B, num_heads, T, head_dim/2]

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    return torch.cat([y1, y2], dim=-1).to(x.dtype)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            attention_heads: int,
            attn_weights_dropout: float,
            rope_base: float,
            rope_max_seq_len: int,
    ):
        super().__init__()
        self.num_heads = attention_heads
        self.head_dim = hidden_dim // attention_heads
        self.attn_weights_dropout = attn_weights_dropout
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.rotary_embedding = RotaryEmbedding(
            head_dim =self.head_dim,
            base = rope_base,
            max_seq_len=rope_max_seq_len,
        )

    def forward(
            self, 
            x,      # [B = batch_size, T = seq_len, D = hidden_dim]
            mask    # [B, T, 1]
    ):
        qkv = self.to_qkv(x)            # [B, T, 3 * D]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, T, D]

        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

        cos, sin = self.rotary_embedding(x.shape[1]) # each [1, 1, T, head_dim/2]
        q = apply_rotary_embeddings(q, cos, sin) # [B, num_heads, T, head_dim]
        k = apply_rotary_embeddings(k, cos, sin) # [B, num_heads, T, head_dim]

        # SDPA adds attn_mask to scores: 0 for valid tokens, -inf for padding
        attn_mask = rearrange(mask, 'b t 1 -> b 1 1 t')
        attn_mask = torch.where(attn_mask, 0.0, float('-inf'))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_weights_dropout if self.training else 0.0,
            is_causal=False,  # Encoder = bidirectional
        )

        out = rearrange(out, 'b h t d -> b t (h d)') # [B, T, D]
        out = self.to_out(out)
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            attention_heads: int,
            attn_weights_dropout: float,
    ):
        super().__init__()
        self.num_heads = attention_heads
        self.head_dim = hidden_dim // attention_heads
        self.attn_weights_dropout = attn_weights_dropout
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
            self,
            x_q,        # [B, T, D]
            x_kv,       # [B, T, D]
            kv_mask,    # [B, T, 1] or None
    ):
        q = self.to_q(x_q)
        kv = self.to_kv(x_kv)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

        if kv_mask is None:
            attn_mask = None
        else:
            attn_mask = rearrange(kv_mask, 'b t 1 -> b 1 1 t')
            attn_mask = torch.where(attn_mask, 0.0, float('-inf'))
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_weights_dropout if self.training else 0.0,
            is_causal=False,
        )

        out = rearrange(out, 'b h t d -> b t (h d)') # [B, T, D]
        out = self.to_out(out)
        return out




class Conv1D(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            filter_size: int,
            kernel_size: int,
            dilation: int = 1,
            bias: bool = False,
    ):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1d = nn.Conv1d(
            hidden_dim,
            filter_size,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(
            self,
            x,    # [B, T, D]
            mask  # [B, T, 1] bool
    ):
        x = x * mask.to(x.dtype)
        # empty_strided + copy_ forces channels-first materialization; .contiguous() after
        # permute is silently elided by Inductor's clone-elimination (PyTorch 2.9.1).
        x_view = x.permute(0, 2, 1)
        B, D, T = x_view.shape
        x = torch.empty_strided((B, D, T), (D * T, T, 1), dtype=x.dtype, device=x.device)
        x.copy_(x_view)
        x = self.conv1d(x)
        return x.permute(0, 2, 1)
