# NaturalSpeech 2 (Unofficial Implementation)

This repository contains an unofficial PyTorch implementation of the **NaturalSpeech 2** [1] text-to-speech (TTS) system.

### Model Architecture (Training)

Below is the complete forward pass through the NaturalSpeech 2 architecture during training. 
*(Click the image to expand and zoom in)*

[![NaturalSpeech 2 Training Architecture](docs/architecture/training-architecture.png)](docs/architecture/training-architecture.png)

*Color Coding:* The sub-modules with individual loss terms are colored the same as their respective terms at the top of the picture. Additionally, all yellow sub-modules represent pre-trained models that aren't trained together with the diffusion model.

**Modifications from the original paper:**

*Open-source substitutions* — the paper relies on proprietary components that we replaced with public equivalents:
- **Audio codec:** Encodec [2] (`facebook/encodec_24khz`, frozen) in place of SoundStream.
- **Phonemizer:** espeak-ng (via the `phonemizer` library) in place of the proprietary Microsoft phonemizer.
- **Aligner:** beta-binomial prior + CTC forward-sum with Viterbi decoding ("One TTS Alignment to Rule Them All" [3]) in place of the proprietary internal Microsoft alignment tool — fully end-to-end.
- **Sample rate:** 24 kHz (hop 320, 75 Hz frame rate) instead of 16 kHz, forced by the Encodec choice.

*Modernised Transformer building blocks* — the paper inherits FastSpeech-era conventions; we use current standards:
- **RoPE** [4] rotary positional embeddings instead of sinusoidal absolute embeddings — better length generalisation, no learned parameters.
- **RMSNorm** [5] in place of LayerNorm, applied **pre-norm** (`norm → op → residual add`) instead of post-norm — materially more stable at the depth of the 30-layer predictor stacks.
- **SiLU** [6] activations in FFNs instead of ReLU.

*Training-stability tweaks:*
- **MSE on log-durations / log-f0** for the duration and pitch predictors, instead of L1 — numerically stabler and standard in follow-up TTS work.

The core latent denoiser is a 40-block WaveNet-style stack that interleaves dilated convolutions with Q-K-V cross-attention and FiLM conditioning:

[![Diffusion Model Architecture (Training)](docs/architecture/diffusion-model-training.png)](docs/architecture/diffusion-model-training.png)

---

## Getting Started

This project is designed to run seamlessly inside a Docker container. We provide a [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml) to handle all system dependencies, C++ extensions, and Python packages via Poetry.

### 1. Build and Start the Environment

Ensure you have Docker and the NVIDIA Container Toolkit installed. You'll also need to create a `docker-compose.override.yml` (gitignored, host-specific) to supply the settings the committed [`docker-compose.yml`](docker-compose.yml) deliberately leaves out — at minimum a **shared-memory size** for the PyTorch DataLoader and a **huggingface cache** mount:

```yaml
services:
  ns2:
    # PyTorch DataLoader ships batches worker→main through /dev/shm. Docker's 64 MB
    # default is far too small for our ~100 MB audio batches → "Bus error (core dumped)".
    shm_size: "8gb"
    volumes:
      # Persist HF dataset/model downloads across containers (avoids re-fetching MLS).
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
```

> **Note:** This is the minimal override. [§3](#3-dataset-and-custom-mounts) below extends the *same* file with `MERGERFS_DISKS` if your dataset spans multiple disks.

Since [`entrypoint.sh`](entrypoint.sh) defaults to dropping you into a bash shell, you can build the image and start an interactive session immediately with:

```bash
docker compose run --rm --build ns2
```

*You are now inside the container's interactive shell*

### 2. Running the Training

We use Hydra for hierarchical configuration management. Configurations are defined in the [`config/`](config/) directory, where [`config/config.yaml`](config/config.yaml) is the default configuration.

You can start the training process and dynamically override config values directly from the terminal. This includes individual values like the Learning Rate, but also full sets of configurations like switching to a predefined experiment.

For example:

```bash
python scripts/train.py training.learning_rate=1e-4 +experiment=overfit_test
```

Running the [training script](scripts/train.py) will automatically initiate our [Preprocessing-Pipeline](naturalspeech2/data/dataset.py) and prepare the dataset, before the Train-Loop starts. 

**Evaluation (decoupled eval daemon).** Every `setup.eval_interval` steps the model is evaluated — held-out loss over the dev+test pool, audio generation on fixed reference clips, and objective metrics (WER intelligibility, SIM-o speaker similarity). With a **second GPU**, this whole eval block is off-loaded to a persistent subprocess pinned to **GPU 1**: the trainer (GPU 0) only does a fast weight-snapshot copy and keeps stepping, so training **never pauses** for evaluation, generation, or metrics. The subprocess is fully isolated — if it crashes it cannot take down training, and the trainer respawns it. It reports held-out loss on **both the live and the EMA weights** (best-checkpoint selection stays on EMA dev loss) and owns the `ema_best.safetensors` write; results stream back to the trainer and are logged to W&B against a dedicated `eval/snapshot_step` x-axis. On a **single-GPU** host — or with `setup.eval_daemon.enabled=false` — evaluation transparently falls back to the original **in-process** path (runs synchronously on the training card between steps, exactly as before). Diagnostic modes (`overfit_test`, `loss_analysis`, `gradient_analysis`, dropout trials) always evaluate in-process.

| `config/setup/base.yaml` key | Default | Purpose |
|---|---|---|
| `eval_daemon.enabled` | `true` | Use the daemon when a 2nd GPU is present; `false` → always in-process. |
| `eval_daemon.compile` | `true` | `torch.compile` the daemon's eval forward (the in-process forward was already compiled). |
| `eval_daemon.num_workers` | `4` | Dataloader workers per daemon loader — independent of `dataloader.num_workers`, so the main run isn't slowed. |
| `eval_daemon.snapshot_dir` | `null` | RAM-backed weight-handoff dir; `null` → `/dev/shm/ns2_eval/<run>`. |
| `eval_metrics` | `["wer"]` | Objective metrics on the fixed refs: `wer` (HuBERT-CTC, no LM) and/or `sim_o` (WavLM-Large-SV speaker cosine vs. the prompt). Enable both with `["wer","sim_o"]`. |
| `metric_device` | `auto` | Metric-model device (WER + SIM-o): `auto` → idle 2nd GPU if present, else CPU (never silently the training card; override e.g. `cuda:0`). Daemon forces its own card. |

---

### 3. Dataset and Custom Mounts

Following the paper [1], we use the english subset of [**Multilingual LibriSpeech (MLS)**](https://huggingface.co/datasets/parler-tts/mls_eng) [7] as our training dataset, which the [Preprocessing-Pipeline](naturalspeech2/data/dataset.py) will load per default. 

> **Note:** This downloads **705 GB** of Parquet (cached in your mounted `~/.cache/huggingface`), deserializing to **~800 GB** of Arrow. The Parquet cache persists *alongside* the Arrow, and during preprocessing the intermediate `.map` outputs transiently double the Arrow before the old shards are reclaimed — so the **peak is ≈ 705 GB Parquet + ~2× 800 GB Arrow ≈ 2.3 TB**, settling back to ~800 GB processed + 705 GB Parquet cache afterwards.
> Additionally, `resample_on_the_fly=False` (e.g. if your CPU node can't resample fast enough during loading) writes pre-resampled FLAC for another **~3.8 TB**.

For fast development iteration, the `+experiment=overfit_test`, `+experiment=loss_analysis`, and `+experiment=gradient_analysis` configs override the train split to MLS' small `dev` split, so they don't trigger preprocessing of the full ~800 GB train data. The default no-arg invocation (the `full_run` composition) uses the train split as usual.

If your dataset spans multiple physical disks, the provided [`entrypoint.sh`](entrypoint.sh) natively supports MergerFS to seamlessly pool them into the `/workspace/data` directory expected by the codebase.
You can configure this by adding the following lines to your `docker-compose.override.yml` file on your host machine to supply the `MERGERFS_DISKS` environment variable and mount the required drives:

```yaml
services:
  ns2:
    environment:
      - MERGERFS_DISKS=/disk1:/disk2
    volumes:
      # Host system disk mounts
      - /path/to/your/first/drive:/disk1
      - /path/to/your/second/drive:/disk2
    devices:
      # Required for mergerfs
      - /dev/fuse:/dev/fuse
    cap_add:
      - SYS_ADMIN
```

---

### 4. Benchmarking and Hardware Optimization

To maximize GPU VRAM utilization, enable efficient `torch.compile` graph caching, and mitigate memory fragmentation, this repository includes an automated benchmarking pipeline. 

We highly recommend running these steps before starting a full training run on a new dataset or hardware configuration.

**1. Finding Optimal Buckets**

Dynamic batch bucketing relies on grouping sequences by length. Fixed buckets are crucial for maximizing VRAM utilization, utilizing `torch.compile`, and reducing memory fragmentation. This [script](scripts/benchmarks/dataloader/find_optimal_buckets.py) determines the optimal buckets for dynamic batch-bucketing of a specific dataset.

It ultimately logs the found optimal buckets (defined by `audio_samples` and `phoneme_tokens`) for different choices of the total number of buckets *K*. The choice of *K* is a trade-off between the number of different shapes that have to be compiled, and the respective resulting padding waste. 

> **Note:** Paste the found bucket-boundaries for your choice of *K* into the respective [config-files](config/dataloader/) before continuing.

```bash
python scripts/benchmarks/dataloader/find_optimal_buckets.py
```

**2. Finding Maximum Batch Sizes**

Once your bucket boundaries are defined, you need to assign the respective buckets their [optimal batch size](scripts/benchmarks/dataloader/find_max_batch_sizes.py), as at that point, the buckets only contain the optimal sequence length.

```bash
python scripts/benchmarks/dataloader/find_max_batch_sizes.py
```

> **Note:** Paste the bucket-boundaries into the respective [config-files](config/dataloader/) again, to override your values from the previous step. These will represent the *(almost)* final bucket-mapping, since it now includes the found maximum batch-size as well.

**3. Stress-Testing Memory Fragmentation**

Even though batch sizes might be stable individually, dynamically jumping between different shapes during training could trigger memory fragmentation. This [stress-test](scripts/benchmarks/dataloader/stress_test_fragmentation.py) makes sure the batch sizes don't lead to an OOM deep into training when specific shapes/buckets follow each other.

```bash
python scripts/benchmarks/dataloader/stress_test_fragmentation.py
```

*If it passes, the output will yield a safe bucket mapping configuration that you should paste directly into your Hydra [config](config/dataloader/), in case the setting from the previous step led to an OOM during the stress-test.*

**4. Dataloader Optimization**  

Determine the most efficient data loading strategy (resampling during pre-processing vs. on-the-fly) and test worker configurations for your specific hardware-setup. 

This can be done by running [this bash script](scripts/benchmarks/dataloader/run_benchmark_dataloader.sh), which will [benchmark](scripts/benchmarks/dataloader/benchmark_dataloader.py) the dataloader across multiple configurations. You might have to update the tested `WORKER_COUNTS` and `ASSUMED_MAX_BATCH_SIZE` to match your machine. 
Per default, it only evaluates, if the resampling on-the-fly option will run efficiently on your hardware, but you can use command-line arguments to test other options as well. 

`--full` tests both the otf-resampling and the pre-resampling option, `--pre-resample` tests only the pre-resampling option (e.g. if you ran the default otf-setting, but your dataloader starved the GPU).

```bash
python scripts/benchmarks/dataloader/run_benchmark_dataloader.sh
```

> **Note:** Technically, resampling always happens during pre-processing, since this is necessary to determine the pitch, but the option to resample on-the-fly discards the resampled audio and therefore saves disk space.

---

### 5. Final Steps before Starting Training
> **Note:** If you just want to recreate our results, and run your training on the same settings, you can start training right away, since the following tests are completely hardware-agnostic.

We ran the mentioned tests to **verify the correctness** of our implementation, and to obtain **reasonable hyperparameter-defaults** for the main training run.

**1. Initialization**

To preserve variance **locally**, we initialize all *Linear*, *Conv1D*, and *Embedding* layers via a Gaussian Distribution with:

$$E[w] = 0$$ 

$$\mathbf{Var[w] = 0.02^2}$$ 

where **Biases** are initialized as **0**. 

> **Note:** While older architectures relied on He-Initialization ($\frac{2}{n_{in}}$) [8], modern Pre-Norm Transformer/Diffusion stacks with zero-initialized residuals naturally prevent variance explosion globally. We use a conservative fixed standard deviation of $\mathcal{N}(0, 0.02)$ to prevent artificially huge updates in narrow low-dimensional layers (like 1D pitch projections) on the first optimizer step.

Per default, we initialize the **scale-parameters** of *RMSNorm-Blocks* as **1**. 

Additionally, to safely route gradients through the deep predictor and WaveNet stacks, we strictly apply Fixup/ReZero principles. We initialize the final operation of every *Residual-Branch* (and FiLM projection) to exactly **0**, forcing the entire network into a perfect Identity-Function at step 0.

> **Note:** Because we use SiLU activations, which have a non-zero derivative at $x=0$, zeroing the weights of a 1-layer residual branch does not kill the gradient, allowing the branch to safely "wake up" during training.

**2. Overfit-Test**

To verify that there are no fundamental errors preventing the model from learning, we intentionally overfit it by running the [Train-Loop](scripts/train.py) on a single batch for multiple iterations.

```bash
python scripts/train.py +experiment=overfit_test
```

**3. Loss-Analysis**

Since our Loss comprises multiple individual Loss-Terms, we run a few iterations of the [Train-Loop](scripts/train.py) to estimate their magnitude. 

We then scale the Loss-Terms accordingly, so they influence the shared parameters equally (actually we might still want to introduce intentional biases towards individual Terms afterwards).

```bash
python scripts/train.py +experiment=loss_analysis
```

**4. Gradient-Analysis**

In order to verify, if the Loss-Balancing from the previous step actually worked, we run the [Train-Loop](scripts/train.py) again, but perform the backward passes individually for each Loss-Term. This allows us to compare the L2-norms of the Loss-Term-gradients of shared parameters, which acts as our measure for impact on the model. 

We additionally calculate cosine-similarities between the gradients, to make sure that individual Loss-Terms do not compete with each other. If that were the case, we would need to introduce further measures, like slowly warming up those specific Loss-Terms.

```bash
python scripts/train.py +experiment=gradient_analysis
```

**5. Hyperparameter-Tuning**

For all Hyperparameters that are explicitly stated in the original paper [1], we use the declared values. The rest is tuned by Optuna, or alternatively set to our best guesses, to perform Hyperparameter-Tuning inside a reasonable time frame. 

For a detailed breakdown, see **this table**.

## Hardware: Reference, Assumptions, and Minimum Baseline

The reference models were trained on the configuration below — but it is our *default*, not a hard requirement. The code targets **any single CUDA-capable NVIDIA GPU**; the per-spec notes that follow say what each default actually buys and how to scale it down.

* **GPU:** NVIDIA RTX PRO 6000 Blackwell Workstation-Edition (96 GB VRAM) — main training card.
* **GPU (optional 2nd):** NVIDIA RTX 4090 (24 GB) — *idle* card that runs the decoupled eval daemon (held-out loss, generation, WER + SIM-o), kept off the training card's VRAM budget so training never pauses for eval.
* **CPU:** 32 cores.
* **RAM:** 252 GB.
* **Disk:** ≈800 GB Arrow for MLS-train (~2× peak during preprocessing; +3.8 TB if `resample_on_the_fly=False`) — see [§3](#3-dataset-and-custom-mounts).

### What our defaults assume

- **GPU / 96 GB VRAM** — encoded *only* in the per-bucket batch sizes in [`config/dataloader/base.yaml`](config/dataloader/base.yaml) (`bucket_mapping`, ~80k audio-frames per batch); nothing else hard-codes VRAM. On a smaller card, re-derive them with the [§4](#4-benchmarking-and-hardware-optimization) benchmarks (`find_optimal_buckets.py` → `find_max_batch_sizes.py`, which measures peak VRAM empirically with a 10 % margin → `stress_test_fragmentation.py`). The sampler hard-fails if a bucket OOMs at batch size 1, so an over-budget config surfaces immediately rather than mid-run.
- **Second GPU (optional)** — when present, runs the **decoupled eval daemon** ([`scripts/eval_daemon.py`](scripts/eval_daemon.py)) so the eval block never blocks training (see [§2](#2-running-the-training)). Auto-detected (`device_count() > 1`); absent → eval runs in-process on the training card, exactly as before. Never asserted, never required. The WER/SIM-o metric device follows `setup.metric_device` (`auto` → idle 2nd GPU else CPU).
- **CPU / 32 cores** — sets the data-pipeline parallelism in [`config/dataloader/base.yaml`](config/dataloader/base.yaml), across two phases. *Training-time:* `num_workers` (loader processes) — tune for your machine with the dataloader benchmark ([§4](#4-benchmarking-and-hardware-optimization)). *One-time preprocessing:* `num_proc_phonemize`/`num_proc_tokenize` are cheap (text / token lists, safe to set high), while `num_proc_pitch` is the heavy one — each worker decodes full audio and runs pyworld F0, so it drives both the preprocessing RAM peak and wall-clock. Lower any of them on fewer cores; it costs speed, not correctness.
- **RAM / 252 GB** — mostly **OS page-cache headroom, not a hard allocation**. The processed dataset is memory-mapped (`load_from_disk` in [`naturalspeech2/data/dataset.py`](naturalspeech2/data/dataset.py)), so spare RAM transparently caches hot shards to keep the loader fed, but that cache is reclaimable. The resident working set is far smaller and has two regimes: training-time DataLoader buffers (`num_workers × prefetch × ~100 MB/batch`, a few GB), and the heavier one-time preprocessing peak — HF batched `.map` holds ~1000 decoded-audio rows per worker, so it scales with `num_proc_pitch` (the audio-decoding stage), **not** `num_workers`. Lower `num_proc_pitch` if preprocessing OOMs.
- **Shared memory (`shm_size`)** — a genuine hard requirement, *not* headroom: the DataLoader ships batches worker→main through `/dev/shm`, and Docker's 64 MB default triggers `Bus error`. Set it to a few GB in your override ([§1](#1-build-and-start-the-environment)).

### Minimum to run the full pipeline (training, preprocessing, and inference)

- **GPU** — one CUDA-capable NVIDIA GPU. Below 96 GB, re-run the [§4](#4-benchmarking-and-hardware-optimization) benchmarks first: the committed batch sizes assume 96 GB and will otherwise OOM.
- **CPU** — any multi-core CPU (lower `num_workers` / `num_proc_*`). Full MLS-train preprocessing is a **multi-day, one-time** CPU job (pyworld-F0-bound, ~linear in `num_proc_pitch`); set `dataset.max_train_clips=N` for a seeded, speaker-diverse subset that cuts this to ~an hour if you don't need the full corpus.
- **RAM** — no hard floor we've measured. Training steady-state is single-digit GB of loader buffers plus reclaimable page-cache; the binding moment is the one-time preprocessing peak, which scales with `num_proc_pitch` (lower it if you OOM) — see the RAM note above. Add a few-GB `shm_size` on top. More RAM only buys dataset page-cache (fewer disk reads), never correctness.
- **Disk** — the binding constraint for the full MLS run; see [§3](#3-dataset-and-custom-mounts). The `dev`-split diagnostic modes (`+experiment=overfit_test|loss_analysis|gradient_analysis`) need only a tiny fraction and are the fastest end-to-end check of a fresh setup.

> **Inference** ([§ Generating Audio](#generating-audio)) is far lighter than training — a single GPU, one clip at a time, no DataLoader workers — so neither `shm_size` nor large RAM is a concern.

---

## Inference and Pre-trained Models

> **Note:** This section is a placeholder. Pre-trained weights and generation scripts will be made available upon the completion of the training runs.

### Inference Architecture

During inference, the model takes a text transcript and a short speech prompt, predicting phoneme durations and pitch, and running the reverse diffusion process to synthesize the final audio latents.

[![NaturalSpeech 2 Inference Architecture](docs/architecture/inference-architecture.png)](docs/architecture/inference-architecture.png)

---

### Generating Audio

*(Instructions on downloading huggingface weights and running the `model.generate()` function will be added here)*

---

## References

*(For academic use, the BibTeX citations for these works can be found in [docs/references.bib](docs/references.bib))*

[1] Shen, Kai, Zeqian Ju, Xu Tan, Eric Liu, Yichong Leng, Lei He, Tao Qin, Sheng Zhao, and Jiang Bian (2024). “NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers”. In: *International Conference on Representation Learning*. Vol. 2024, pp. 698–722.

[2] Défossez, Alexandre, Jade Copet, Gabriel Synnaeve, and Yossi Adi (2023). “High Fidelity Neural Audio Compression”. In: *Transactions on Machine Learning Research*.

[3] Badlani, Rohan, Adrian Łańcucki, Kevin J. Shih, Rafael Valle, Wei Ping, and Bryan Catanzaro (2022). “One TTS Alignment to Rule Them All”. In: *ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 6092–6096.

[4] Su, Jianlin, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu (2024). “RoFormer: Enhanced transformer with Rotary Position Embedding”. In: *Neurocomputing* 568.

[5] Zhang, Biao and Rico Sennrich (2019). “Root Mean Square Layer Normalization”. In: *Advances in Neural Information Processing Systems*. Ed. by H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-Buc, E. Fox, and R. Garnett. Vol. 32. Curran Associates, Inc.

[6] Elfwing, Stefan, Eiji Uchibe, and Kenji Doya (2018). “Sigmoid-weighted linear units for neural network function approximation in reinforcement learning”. In: *Neural Networks* 107. Special issue on deep reinforcement learning, pp. 3–11.

[7] Pratap, Vineel, Qiantong Xu, Anuroop Sriram, Gabriel Synnaeve, and Ronan Collobert (2020). “MLS: A Large-Scale Multilingual Dataset for Speech Research”. In: *Interspeech 2020*, pp. 2757–2761.

[8] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2015). “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification”. In: *2015 IEEE International Conference on Computer Vision (ICCV)*, pp. 1026–1034.