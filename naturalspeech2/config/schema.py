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
    score_eps: float = 0.05
    timestep_eps: float = 1e-3
    min_snr_gamma: float = 5.0


@dataclass
class AlignerLossWeights:
    group_weight: float = 1.0
    forward_sum_loss: float = 1.0
    bin_loss: float = 1.0


@dataclass
class DiffusionLossWeights:
    group_weight: float = 1.0
    data_loss: float = 1.0
    score_loss: float = 1.0
    ce_rvq_loss: float = 0.1


@dataclass
class LossWeights:
    duration_predictor_loss: float = 1.0
    pitch_predictor_loss: float = 1.0
    aligner_loss: AlignerLossWeights = field(default_factory=AlignerLossWeights)
    diffusion_loss: DiffusionLossWeights = field(default_factory=DiffusionLossWeights)


@dataclass
class AlignerLossWarmups:
    group_warmup: int = 0
    forward_sum_loss: int = 0
    bin_loss: int = 0


@dataclass
class DiffusionLossWarmups:
    group_warmup: int = 1000
    data_loss: int = 0
    score_loss: int = 0
    ce_rvq_loss: int = 0


@dataclass
class LossWarmups:
    duration_predictor_loss: int = 1000
    pitch_predictor_loss: int = 1000
    aligner_loss: AlignerLossWarmups = field(default_factory=AlignerLossWarmups)
    diffusion_loss: DiffusionLossWarmups = field(default_factory=DiffusionLossWarmups)


@dataclass
class ModelConfig:
    hidden_dim: int = 512
    latent_dim: int = 128
    rope_base: float = 10000.0
    rope_max_seq_len: int = 3000
    prompt_seconds: float = 3.0
    min_target_seconds: float = 1.0

    mel: MelConfig = field(default_factory=MelConfig)
    phoneme_encoder: PhonemeEncoderConfig = field(default_factory=PhonemeEncoderConfig)
    aligner: AlignerConfig = field(default_factory=AlignerConfig)
    speech_prompt_encoder: SpeechPromptEncoderConfig = field(default_factory=SpeechPromptEncoderConfig)
    duration_predictor: DurationPredictorConfig = field(default_factory=DurationPredictorConfig)
    pitch_predictor: PitchPredictorConfig = field(default_factory=PitchPredictorConfig)
    diffusion_model: DiffusionModelConfig = field(default_factory=DiffusionModelConfig)

    loss_weights: LossWeights = field(default_factory=LossWeights)
    loss_warmup_steps: LossWarmups = field(default_factory=LossWarmups)


def model_cfg_from_omegaconf(cfg: Any) -> ModelConfig:
    # OmegaConf.merge validates `cfg` against the dataclass schema (raises on
    # unknown keys / type mismatches). to_object returns a real ModelConfig
    # instance, so call sites are Hydra-agnostic from here on.
    schema = OmegaConf.structured(ModelConfig)
    merged = OmegaConf.merge(schema, cfg)
    return OmegaConf.to_object(merged)
