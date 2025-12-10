import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

from naturalspeech2.data.vctk import VCTKDataset, vctk_collate_fn
from naturalspeech2.model import NaturalSpeech2Model
from naturalspeech2.data.phoneme_tokenizer import PhonemeTokenizer

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):

    if cfg.training.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = cfg.training.device
    print(f"Using device: {device}")

    tokenizer = PhonemeTokenizer()
    token_vocabulary_size = tokenizer.token_vocabulary_size 

    dataset = VCTKDataset(
        sampling_rate=cfg.data.sampling_rate,
        num_proc=cfg.data.num_proc
    )

    loader = DataLoader(
        dataset, 
        batch_size=cfg.data.batch_size, 
        shuffle=cfg.data.shuffle, 
        collate_fn=vctk_collate_fn
    )

    model_args = {
        'device': device,
        'token_vocabulary_size': token_vocabulary_size,
        'dim_hidden': cfg.model.dim_hidden,
        'sampling_rate': cfg.data.sampling_rate,
        'rope_base': cfg.model.rope_base,
        'rope_max_seq_len': cfg.model.rope_max_seq_len,
        'min_prompt_pct': cfg.model.min_prompt_pct,
        'max_prompt_pct': cfg.model.max_prompt_pct,

        # Log Mel Spectrogram parameters
        'n_fft': cfg.model.mel.n_fft,
        'hop_length': cfg.model.mel.hop_length,
        'n_mels': cfg.model.mel.n_mels,
        'f_min': cfg.model.mel.f_min,
        'f_max': cfg.model.mel.f_max,

        # Phoneme Encoder parameters
        'phoneme_encoder_layers': cfg.model.phoneme_encoder.transformer_layers,
        'phoneme_encoder_heads': cfg.model.phoneme_encoder.attention_heads,
        'phoneme_encoder_filter_size': cfg.model.phoneme_encoder.conv1d_filter_size,
        'phoneme_encoder_kernel_size': cfg.model.phoneme_encoder.conv1d_kernel_size,
        'phoneme_encoder_dropout': cfg.model.phoneme_encoder.dropout,

        # Aligner parameters
        'aligner_attn_channels': cfg.model.aligner.attn_channels,
        'aligner_temperature': cfg.model.aligner.temperature,
        'prior_w': cfg.model.aligner.prior_w,

        # Speech Prompt Encoder parameters
        'speech_prompt_encoder_layers': cfg.model.speech_prompt_encoder.transformer_layers,
        'speech_prompt_encoder_heads': cfg.model.speech_prompt_encoder.attention_heads,
        'speech_prompt_encoder_filter_size': cfg.model.speech_prompt_encoder.conv1d_filter_size,
        'speech_prompt_encoder_kernel_size': cfg.model.speech_prompt_encoder.conv1d_kernel_size,
        'speech_prompt_encoder_dropout': cfg.model.speech_prompt_encoder.dropout,
    }

    model = NaturalSpeech2Model(**model_args).to(device)

    


if __name__ == "__main__":
    # train()
    pass