import torch
from torch import nn
import torch.nn.functional as F

class DurationPredictor(nn.Module):
    def __init__(
            self,
            dim_hidden: int = 512,
            conv1d_layers: int = 30,
            conv1d_kernel_size: int = 3,
            attention_layers: int = 10,
            attention_heads: int = 8,
            dropout: float = 0.5,
    ):
        super().__init__()

    def forward():
        pass


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-Attention:
      Q  <- phoneme_encodings   [B, dim_hidden, P]
      K,V<- prompt_encodings    [B, dim_hidden, F]
    Output:
      out -> [B, dim_hidden, P]
    """
    def __init__(
            self,
            dim_hidden: int,
            attention_heads: int,
            dropout: float
    ):
        super().__init__()
        assert dim_hidden % attention_heads == 0, "dim_hidden must be divisible by attention_heads"
        self.heads = attention_heads
        self.dim_head = dim_hidden // attention_heads
        self.dropout = dropout

        self.to_q  = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim_hidden, dim_hidden * 2, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim_hidden, bias=False)

    def forward(
        self,
        phoneme_encodings,        # [B, dim_hidden, P]
        phoneme_encodings_mask,   # [B, 1, P]
        prompt_encodings,         # [B, dim_hidden, F]
        prompt_encodings_mask,    # [B, 1, F]
    ):
        # [B, dim_hidden, sequence] -> [B, sequence, dim_hidden]
        x_q  = phoneme_encodings.transpose(1, 2)   # [B, P, dim_hidden]
        x_kv = prompt_encodings.transpose(1, 2)    # [B, F, dim_hidden]

        B, P, dim_hidden = x_q.shape
        _, F, _ = x_kv.shape

        q = self.to_q(x_q)                # [B, P, dim_hidden]
        kv = self.to_kv(x_kv)             # [B, F, 2*dim_hidden]
        k, v = kv.chunk(2, dim=-1)        # each [B, F, dim_hidden]

        # [B, sequence, dim_hidden] -> [B, heads, sequence, dim_head]
        q = q.view(B, P, self.heads, self.dim_head).transpose(1, 2)   # [B, heads, P, dim_head]
        k = k.view(B, F, self.heads, self.dim_head).transpose(1, 2)   # [B, heads, F, dim_head]
        v = v.view(B, F, self.heads, self.dim_head).transpose(1, 2)   # [B, heads, F, dim_head]


        attn_mask = prompt_encodings_mask[:, :, None, :]   # [B, 1, 1, F]
        attn_mask = torch.where(attn_mask, 0.0, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        # merge heads: [B, heads, P, dim_head] -> [B, P, dim_hidden]
        out = out.transpose(1, 2).contiguous().view(B, P, dim_hidden)
        out = self.to_out(out)  # [B, P, dim_hidden]

        q_mask = phoneme_encodings_mask.to(out.dtype)  # [B, 1, P]
        out = out * q_mask.transpose(1, 2)             # [B, P, 1] broadcast

        # zurück zu channel-first: [B, P, dim_hidden] -> [B, dim_hidden, P]
        return out.transpose(1, 2)