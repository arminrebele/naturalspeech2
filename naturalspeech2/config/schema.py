from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import OmegaConf


@dataclass
class MelConfig:
    n_fft: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: Optional[float] = None


@dataclass
class PhonemeEncoderConfig:
    transformer_layers: int = 6
    attention_heads: int = 8
    conv1d_filter_size: int = 2048
    conv1d_kernel_size: int = 9
    conv_dropout: float = 0.2
    attn_weights_dropout: float = 0.2
    attn_out_dropout: float = 0.2


@dataclass
class AlignerConfig:
    attn_channels: int = 80
    temperature: float = 0.0005
    prior_w: float = 1.0
    dropout: float = 0.1


@dataclass
class SpeechPromptEncoderConfig:
    transformer_layers: int = 6
    attention_heads: int = 8
    conv1d_filter_size: int = 2048
    conv1d_kernel_size: int = 9
    conv_dropout: float = 0.2
    attn_weights_dropout: float = 0.2
    attn_out_dropout: float = 0.2


@dataclass
class DurationPredictorConfig:
    conv1d_layers: int = 30
    conv1d_kernel_size: int = 3
    attention_layers: int = 10
    attention_heads: int = 8
    conv_dropout: float = 0.5
    attn_weights_dropout: float = 0.5
    attn_out_dropout: float = 0.5


@dataclass
class PitchPredictorConfig:
    conv1d_layers: int = 30
    conv1d_kernel_size: int = 5
    attention_layers: int = 10
    attention_heads: int = 8
    conv_dropout: float = 0.5
    attn_weights_dropout: float = 0.5
    attn_out_dropout: float = 0.5


@dataclass
class DiffusionModelConfig:
    wavenet_layers: int = 40
    wavenet_kernel_size: int = 3
    wavenet_dilation: int = 2
    wavenet_filter_size: int = 1024
    attention_heads: int = 8
    query_tokens: int = 32
    attn_weights_dropout: float = 0.2
    attn_out_dropout: float = 0.2
    wavenet_attn_weights_dropout: float = 0.2
    wavenet_attn_out_dropout: float = 0.2
    wavenet_gate_dropout: float = 0.2
    time_dim: int = 128
    beta_min: float = 0.1
    beta_max: float = 20.0
    sampling_steps: int = 150
    sampling_temperature: float = 1.44
    score_loss_weight: float = 0.1
    score_eps: float = 0.05
    ce_rvq_loss_weight: float = 0.1
    timestep_eps: float = 1e-3


@dataclass
class ModelConfig:
    hidden_dim: int = 512
    latent_dim: int = 128
    rope_base: float = 10000.0
    rope_max_seq_len: int = 3000
    min_prompt_pct: float = 0.2
    max_prompt_pct: float = 0.5

    mel: MelConfig = field(default_factory=MelConfig)
    phoneme_encoder: PhonemeEncoderConfig = field(default_factory=PhonemeEncoderConfig)
    aligner: AlignerConfig = field(default_factory=AlignerConfig)
    speech_prompt_encoder: SpeechPromptEncoderConfig = field(default_factory=SpeechPromptEncoderConfig)
    duration_predictor: DurationPredictorConfig = field(default_factory=DurationPredictorConfig)
    pitch_predictor: PitchPredictorConfig = field(default_factory=PitchPredictorConfig)
    diffusion_model: DiffusionModelConfig = field(default_factory=DiffusionModelConfig)

    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "diffusion_loss": 1.0,
        "duration_predictor_loss": 1.0,
        "pitch_predictor_loss": 1.0,
        "aligner_loss": 1.0,
        "forward_sum_loss": 1.0,
        "bin_loss": 1.0,
    })
    loss_warmup_steps: dict[str, int] = field(default_factory=lambda: {
        "diffusion_loss": 1000,
        "duration_predictor_loss": 1000,
        "pitch_predictor_loss": 1000,
        "aligner_loss": 0,
        "forward_sum_loss": 0,
        "bin_loss": 0,
    })


def model_cfg_from_omegaconf(cfg: Any) -> ModelConfig:
    # OmegaConf.merge validates `cfg` against the dataclass schema (raises on
    # unknown keys / type mismatches). to_object returns a real ModelConfig
    # instance, so call sites are Hydra-agnostic from here on.
    schema = OmegaConf.structured(ModelConfig)
    merged = OmegaConf.merge(schema, cfg)
    return OmegaConf.to_object(merged)
