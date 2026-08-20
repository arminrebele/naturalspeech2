# NaturalSpeech 2 (Unofficial Implementation)

Unofficial PyTorch implementation of NaturalSpeech 2 [1]: zero-shot text-to-speech via
latent diffusion over Encodec [2] latents, with speech-prompted speaker transfer and
duration/pitch prediction. The phoneme aligner is trained end-to-end with the model
(monotonic alignment search adapted from Glow-TTS [3]) — no external forced aligner or
precomputed alignments required.

## Setup

Requires Linux and an NVIDIA GPU (the lockfile pins CUDA 12.8 wheels), plus
[uv](https://docs.astral.sh/uv/) and the espeak-ng library
(Debian/Ubuntu: `apt install espeak-ng`).

    git clone <repo-url> && cd naturalspeech2
    uv sync

Put your Weights & Biases key in `.env` (`WANDB_API_KEY=...`), or run with `wandb.log=false`.

## Usage

Run from the repo root:

    # preprocess the dataset (MLS English [4]; downloads + F0-extracts on first use)
    uv run python scripts/preprocess.py +experiment=10M

    # train
    uv run python scripts/train.py +experiment=10M run_name=my_run

    # generate audio + metrics from a trained checkpoint
    uv run python scripts/eval_checkpoint.py checkpoint=models/checkpoints/<group>/<run>/ckpt.pt

Batch-size defaults are tuned for the reference setup: an NVIDIA RTX PRO 6000
Blackwell (96 GB) for training, with an RTX 4090 (24 GB) picked up automatically
as second GPU for parallel eval.

## References

[1] Shen et al., *NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot
    Speech and Singing Synthesizers* (2023). [arXiv:2304.09116](https://arxiv.org/abs/2304.09116)

[2] Défossez et al., *High Fidelity Neural Audio Compression* (2022).
    [arXiv:2210.13438](https://arxiv.org/abs/2210.13438) — Encodec

[3] Kim et al., *Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment
    Search* (NeurIPS 2020). [arXiv:2005.11129](https://arxiv.org/abs/2005.11129) — vendored
    monotonic-align kernel (MIT)

[4] Pratap et al., *MLS: A Large-Scale Multilingual Dataset for Speech Research* (2020).
    [arXiv:2012.03411](https://arxiv.org/abs/2012.03411) — training data
