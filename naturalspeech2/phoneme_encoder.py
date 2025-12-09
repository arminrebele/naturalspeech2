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

import torch
from torch import nn
import torch.nn.functional as F


class PhonemeEncoder(nn.Module):
    """
    Token-IDs -> Embedding -> N x TransformerEncoderLayer -> Output
    """
    def __init__(
            self,
            token_vocabulary_size: int = None,
            dim_hidden: int = 512,
            transformer_layers: int = 6,
            attention_heads: int = 8,
            conv1d_filter_size: int = 2048,
            conv1d_kernel_size: int = 9,
            dropout: float = 0.2,
            rope_base: float = 10000.0,
            rope_max_seq_len: int = 3000,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.token_embedding = nn.Embedding(token_vocabulary_size, dim_hidden, padding_idx=0)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim_hidden,
                attention_heads,
                conv1d_filter_size, 
                conv1d_kernel_size, 
                dropout,
                rope_base,
                rope_max_seq_len,
                )
            for _ in range(transformer_layers)
        ])

        self.final_norm = RMSNorm(dim_hidden)

    def forward(
            self,
            phoneme_tokens: torch.Tensor,
            phoneme_tokens_mask: torch.Tensor,
            phoneme_tokens_lengths: torch.Tensor,
    ):
        x = self.token_embedding(phoneme_tokens) # [B, P, dim_hidden]

        for layer in self.transformer_layers:
            x = layer(x, phoneme_tokens_mask)
        
        x = self.final_norm(x)

        x = x.transpose(1, 2)  # [B, dim_hidden, P]
        
        return x

class TransformerEncoderLayer(nn.Module):
    """
    x -> RMSNorm -> MultiHeadSelfAttention -> + (Residual)
          -> RMSNorm -> Conv1DFeedForward -> + (Residual)
    """
    def __init__(
            self,
            dim_hidden: int,
            attention_heads: int,
            conv1d_filter_size: int,
            conv1d_kernel_size: int,
            dropout: float,
            rope_base: float,
            rope_max_seq_len: int,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim_hidden)

        self.multi_head_attention = MultiHeadSelfAttention(
            dim_hidden,
            attention_heads,
            dropout,
            rope_base,
            rope_max_seq_len,
        )

        self.norm2 = RMSNorm(dim_hidden)

        self.conv1d_feed_forward = Conv1DFeedForward(
            dim_hidden,
            conv1d_filter_size,
            conv1d_kernel_size,
            dropout,
        )

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, phoneme_tokens_mask):
        attn_out = self.multi_head_attention(self.norm1(x), phoneme_tokens_mask)
        x = x + self.dropout(attn_out)

        ffn_out = self.conv1d_feed_forward(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x

class RMSNorm(nn.Module):
    """
    x_norm = x / sqrt(mean(x^2) + eps) * gamma
    """
    def __init__(self, dim_hidden: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim_hidden)) # gamma
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim_head: int, base: float = 10000.0, max_seq_len: int = 3000):
        super().__init__()
        self.dim_head = dim_head
        self.base = base
        self.max_seq_len = max_seq_len

        cos, sin = self._precompute_rotary_embeddings(dim_head, base, max_seq_len)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, dim_head: int, base: float, max_seq_len: int):
        pair_indices = torch.arange(0, dim_head, 2, dtype=torch.float32) # equals already 2i
        thetas = 1.0 / (base ** (pair_indices / dim_head))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        alphas = torch.outer(positions, thetas) # [max_seq_len x dim_head/2]

        cos = alphas.cos()
        sin = alphas.sin()

        cos = cos[None, :, None, :] # [1, max_seq_len, 1, dim_head/2]
        sin = sin[None, :, None, :] # [1, max_seq_len, 1, dim_head/2]
        
        return cos, sin

    def forward(self, seq_len: int):
        return self.cos[:, :seq_len], self.sin[:, :seq_len]

def apply_rotary_embeddings(
        x,    # [B, P, heads, dim_head]
        cos,  # [1, P, 1, dim_head/2]
        sin,  # [1, P, 1, dim_head/2]
):
    d = x.shape[-1] // 2
    x1 = x[..., :d]  # [B, P, heads, dim_head/2]
    x2 = x[..., d:]  # [B, P, heads, dim_head/2]

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    return torch.cat([y1, y2], dim=-1).to(x.dtype)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            dim_hidden: int,
            attention_heads: int,
            dropout: float,
            rope_base: float,
            rope_max_seq_len: int,
    ):
        super().__init__()
        self.heads = attention_heads
        self.dim_head = dim_hidden // attention_heads
        self.dropout = dropout
        self.to_qkv = nn.Linear(dim_hidden, dim_hidden * 3, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim_hidden, bias=False)

        self.rotary_embedding = RotaryEmbedding(
            dim_head=self.dim_head,
            base = rope_base,
            max_seq_len=rope_max_seq_len,
        )

    def forward(self, x, mask):
        B, P, dim_hidden = x.shape # [B, P, dim_hidden]
        qkv = self.to_qkv(x) # [B, P, 3 * dim_hidden]
        q, k, v = qkv.chunk(3, dim=-1) # each [B, P, dim_hidden]

        # reshape for multi-head attention
        # [B, P, dim_hidden] -> [B, P, heads, dim_head]
        q = q.view(B, P, self.heads, self.dim_head)
        k = k.view(B, P, self.heads, self.dim_head)
        v = v.view(B, P, self.heads, self.dim_head)

        # apply RoPE to Q and K
        cos, sin = self.rotary_embedding(P)
        q = apply_rotary_embeddings(q, cos, sin) # [B, P, heads, dim_head]
        k = apply_rotary_embeddings(k, cos, sin) # [B, P, heads, dim_head]

        # transpose for scaled_dot_product_attention
        # [B, P, heads, dim_head] -> [B, heads, P, dim_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # mask: [B, 1, P] mit True = gültig, False = Padding
        # scaled_dot_product_attention adds attn_mask to the scores
        # Also: 0. 0 for valid tokens, -inf for padding
        # [B, 1, P] -> [B, 1, 1, P] for broadcasting over heads and query positions
        attn_mask = mask[:, :, None, :]  # [B, 1, 1, P]
        attn_mask = torch.where(attn_mask, 0.0, float('-inf'))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # Encoder = bidirectional
        )

        # [B, heads, P, dim_head] -> [B, P, heads, dim_head] -> [B, P, dim_hidden]
        out = out.transpose(1, 2).contiguous().view(B, P, -1)
        
        out = self.to_out(out)

        return out

class Conv1DFeedForward(nn.Module):
    def __init__(
            self,
            dim_hidden: int,
            conv1d_filter_size: int,
            conv1d_kernel_size: int,
            dropout: float,
    ):
        super().__init__()
        padding = (conv1d_kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(dim_hidden, conv1d_filter_size, conv1d_kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(conv1d_filter_size, dim_hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: [B, P, dim_hidden]
        # mask: [B, 1, P]

        x = x.transpose(1, 2)      # [B, dim_hidden, P]
        x = x.masked_fill(~mask, 0.0)  # mask: [B, 1, P]
        
        x = self.conv1(x)
        x = F.silu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        
        x = x.transpose(1, 2)  # [B, P, dim_hidden]
        
        return x


if __name__ == "__main__":
    from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = PhonemeTokenizer()
    vocab_size = tokenizer.token_vocabulary_size
    print(f"Vocabulary size: {vocab_size}")

    phoneme_encoder = PhonemeEncoder(
        token_vocabulary_size=vocab_size,
        dim_hidden=512,
        transformer_layers=6,
        attention_heads=8,
        conv1d_filter_size=2048,
        conv1d_kernel_size=9,
        dropout=0.2,
    ).to(device)

    texts = [
        "Hello, world!",
        "This is a test.",
    ]
    
    # Tokenize
    token_ids_list = [tokenizer(text) for text in texts]
    lengths = [len(ids) for ids in token_ids_list]
    max_len = max(lengths)
    
    # Pad sequences
    padded_tokens = []
    for ids in token_ids_list:
        padded = ids + [0] * (max_len - len(ids))  # 0 = <pad>
        padded_tokens.append(padded)
    
    # Create tensors
    phoneme_tokens = torch.tensor(padded_tokens, device=device)  # [B, P]
    phoneme_tokens_lengths = torch.tensor(lengths, device=device)  # [B]
    
    # Create mask: True = valid, False = padding
    token_idx = torch.arange(max_len, device=device).unsqueeze(0)  # [1, P]
    phoneme_tokens_mask = (token_idx < phoneme_tokens_lengths.unsqueeze(1)).unsqueeze(1)  # [B, 1, P]
    
    print(f"\nInput shapes:")
    print(f"  phoneme_tokens: {phoneme_tokens.shape}")
    print(f"  phoneme_tokens_mask: {phoneme_tokens_mask.shape}")
    print(f"  phoneme_tokens_lengths: {phoneme_tokens_lengths}")
    
    # Forward pass
    phoneme_encoder.eval()
    with torch.no_grad():
        output = phoneme_encoder(phoneme_tokens, phoneme_tokens_mask, phoneme_tokens_lengths)
    
    print(f"\nOutput shape: {output.shape}")  # Expected: [B, dim_hidden, P]
    
    # Verify output values
    print(f"\nOutput statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    
    # Parameter count
    total_params = sum(p.numel() for p in phoneme_encoder.parameters())
    trainable_params = sum(p.numel() for p in phoneme_encoder.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
