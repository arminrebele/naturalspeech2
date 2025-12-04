import torch
from torch.utils.data import DataLoader


from naturalspeech2.data.vctk import VCTKDataset, vctk_collate_fn
from naturalspeech2.model import NaturalSpeech2Model


def train():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {} # TODO: define model config
    
    dataset = VCTKDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=vctk_collate_fn)

    model = NaturalSpeech2Model(config).to(device)


    


if __name__ == "__main__":
    # train()
    pass