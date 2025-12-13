import torch
from torch import nn
import torch.nn.functional as F
from naturalspeech2.transformer_encoder_layer import TransformerEncoderLayer, RMSNorm

class SpeechPromptEncoder(nn.Module):
    """
    Prompt Encodings -> Projection -> N x TransformerEncoderLayer -> Output
    """
    def __init__(
            self,
            hidden_dim: int = 512,
            latent_dim: int = 128,
            transformer_layers: int = 6,
            attention_heads: int = 8,
            conv1d_filter_size: int = 2048,
            conv1d_kernel_size: int = 9,
            dropout: float = 0.2,
            rope_base: float = 10000.0,
            rope_max_seq_len: int = 3000,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)

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
            prompt_latents: torch.Tensor,         # [B, latent_dim, F]
            prompt_latents_mask: torch.Tensor,    # [B, 1, F]
            prompt_latents_lengths: torch.Tensor,
    ):
        x = self.input_projection(prompt_latents)  # [B, hidden_dim, F]
        x = x.transpose(1, 2)  # [B, F, hidden_dim]

        for layer in self.transformer_layers:
            x = layer(x, prompt_latents_mask)

        x = self.final_norm(x)

        x = x.transpose(1, 2)  # [B, hidden_dim, F]

        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    B = 4
    F = 50
    latent_dim = 128
    hidden_dim = 512
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    speech_prompt_encoder = SpeechPromptEncoder(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        transformer_layers=6,
        attention_heads=8,
        conv1d_filter_size=2048,
        conv1d_kernel_size=9,
        dropout=0.2,
    ).to(device)

    prompt_latents = torch.randn(B, latent_dim, F, device=device)
    prompt_latents_lengths = torch.randint(low=int(F*0.5), high=F + 1, size=(B,), device=device)
    
    # Create mask
    frame_idx = torch.arange(F, device=device).unsqueeze(0)  # [1, F]
    prompt_latents_mask = (frame_idx < prompt_latents_lengths.unsqueeze(1)).unsqueeze(1)  # [B, 1, F]

    # Apply mask to input for a more realistic test (zero out padded parts)
    prompt_latents = prompt_latents * prompt_latents_mask

    print(f"Input Shapes:")
    print(f"  prompt_latents: {prompt_latents.shape}")
    print(f"  prompt_latents_mask: {prompt_latents_mask.shape}")
    print(f"  prompt_latents_lengths: {prompt_latents_lengths}")
    
    # --- Forward Pass ---
    speech_prompt_encoder.eval()
    with torch.no_grad():
        output = speech_prompt_encoder(
            prompt_latents,
            prompt_latents_mask,
            prompt_latents_lengths
        )

    print(f"  Output shape: {output.shape}")  # Expected: [B, hidden_dim, F]
    assert output.shape == (B, hidden_dim, F)
    
    # --- Output Verification ---
    print(f"\nOutput Statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    
    # --- Parameter Count ---
    total_params = sum(p.numel() for p in speech_prompt_encoder.parameters())
    trainable_params = sum(p.numel() for p in speech_prompt_encoder.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")