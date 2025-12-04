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

    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['token_vocabulary_size'] = token_vocabulary_size
    model_config['device'] = device
    model_config['sampling_rate'] = cfg.data.sampling_rate

    model = NaturalSpeech2Model(**model_config).to(device)

    


if __name__ == "__main__":
    # train()
    pass