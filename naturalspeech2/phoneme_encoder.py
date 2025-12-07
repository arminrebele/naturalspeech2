import torch
from torch import nn
import torch.nn.functional as F

class PhonemeEncoder(nn.Module):
    def __init__(
            self,
            token_vocabulary_size: int = None,
            dim_hidden: int = 512,
            transformer_layers: int = 6,
            attention_heads: int = 8,
            conv1d_filter_size: int = 2048,
            conv1d_kernel_size: int = 9,
            dropout: float = 0.2,
    ):
        super().__init__()

    def forward(
            self,
            phoneme_tokens: torch.Tensor,
            phoneme_tokens_mask: torch.Tensor,
            phoneme_tokens_lengths: torch.Tensor,
    ):
        pass

class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            dim_hidden: int,
            attention_heads: int,
            conv1d_filter_size: int,
            conv1d_kernel_size: int,
            dropout: float,
    ):
        super().__init__()
    
    def forward(self):
        pass

class RMSNorm(nn.Module):
    def __init__(self, dim_hidden: int, eps: float = 1e-8):
        super().__init__()
    
    def forward(self):
        pass

class RotaryEmbedding(nn.Module):
    def __init__(self, dim_hidden: int):
        super().__init__()
    
    def forward(self):
        pass

class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            dim_hidden: int,
            attention_heads: int,
            dropout: float,
    ):
        super().__init__()
    
    def forward(self):
        pass

class Conv1DFeedForward(nn.Module):
    def __init__(
            self,
            dim_hidden: int,
            conv1d_filter_size: int,
            conv1d_kernel_size: int,
            dropout: float,
    ):
        super().__init__()
    
    def forward(self):
        pass
