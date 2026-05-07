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
    """
    x_norm = x / sqrt(mean(x^2) + eps) * gamma
    """
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
        x_f = x.float() # upcast to float32 for stable mean(x^2) calculation
        rms_recip = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f * rms_recip).to(dtype) * self.weight


"""
##### RoPE #####
m: Token position (index 0, 1, 2, ...)
d: Head-embedding dimension (must be even)
i: Pair index (0, 1, 2, ..., d/2-1)

theta_i = 10000^( -2i/d )    #base frequency for pair i
alpha = m * theta_i          #angle for token position m and pair i

(x, y) = (x_m_i, y_m_i)

x' = x * cos(alpha) - y * sin(alpha)
y' = x * sin(alpha) + y * cos(alpha)

=> (x', y')

Example:
dim_head = 64 => 32 pairs of (x, y)
max_seq_len = 2048
[2048 x 32]

- calculate theta_i for i in [0, 31]
- calculate alpha = m * theta_i for m in [0, 2047]
=> [2048 x 32] matrix with alpha values
- calculate cos(alpha) and sin(alpha) matrices
=> [2048 x 32] matrices for cos and sin => [2048 x 64]
Therefore we can precompute the cos and sin matrices for a given max_seq_len and dim_head.
During inference for a given input sequence length, slice the precomputed cos and sin matrices,
and use them to calculate the rotated (x', y') values.

Implementation follows Andrej Karpathy's nanoChat approach, applying RoPE on contiguous halves (sliced)
rather than interleaved pairs, to avoid the rotate_half shuffle overhead.
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

        cos, sin = self.rotary_embedding(x.shape[1]) # # [1, T, 1, head_dim/2]
        q = apply_rotary_embeddings(q, cos, sin) # [B, num_heads, T, head_dim]
        k = apply_rotary_embeddings(k, cos, sin) # [B, num_heads, T, head_dim]

        # scaled_dot_product_attention adds attn_mask to the scores
        # 0. 0 for valid tokens, -inf for padding
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
        x = rearrange(x, 'b t d -> b d t')
        x = self.conv1d(x)
        return rearrange(x, 'b d t -> b t d')
