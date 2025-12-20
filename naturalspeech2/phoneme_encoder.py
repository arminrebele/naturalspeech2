import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from naturalspeech2.modules import TransformerEncoderLayer, RMSNorm

class PhonemeEncoder(nn.Module):
    def __init__(
            self,
            token_vocabulary_size: int = None,
            hidden_dim: int = 512,
            transformer_layers: int = 6,
            attention_heads: int = 8,
            conv1d_filter_size: int = 2048,
            conv1d_kernel_size: int = 9,
            dropout: float = 0.2,
            rope_base: float = 10000.0,
            rope_max_seq_len: int = 3000,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(token_vocabulary_size, hidden_dim, padding_idx=0)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim,
                attention_heads,
                conv1d_filter_size, 
                conv1d_kernel_size, 
                dropout,
                rope_base,
                rope_max_seq_len,
                )
            for _ in range(transformer_layers)
        ])

        self.final_norm = RMSNorm(hidden_dim)

    def forward(
            self,
            phoneme_tokens: torch.Tensor,           # [B, P]
            phoneme_tokens_mask: torch.Tensor,      # [B, P, 1]
            phoneme_tokens_lengths: torch.Tensor,   # [B]
    ):
        phoneme_tokens_emb = self.token_embedding(phoneme_tokens) # [B, P, hidden_dim]

        for layer in self.transformer_layers:
            phoneme_tokens_emb = layer(phoneme_tokens_emb, phoneme_tokens_mask)
        
        phoneme_tokens_emb = self.final_norm(phoneme_tokens_emb)

        phoneme_tokens_emb = phoneme_tokens_emb * phoneme_tokens_mask
        
        return phoneme_tokens_emb # [B, P, hidden_dim]


if __name__ == "__main__":
    from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = PhonemeTokenizer()
    vocab_size = tokenizer.token_vocabulary_size
    print(f"Vocabulary size: {vocab_size}")

    phoneme_encoder = PhonemeEncoder(
        token_vocabulary_size=vocab_size,
        hidden_dim=512,
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
    
    print(f"\nOutput shape: {output.shape}")  # Expected: [B, hidden_dim, P]
    
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
