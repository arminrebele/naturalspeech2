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

Ensure you have Docker and the NVIDIA Container Toolkit installed. You'll also need to create a `docker-compose.override.yml` to pass machine-specific settings like your huggingface/hub cache and the shared memory size for PyTorch Dataloader workers.

Since [`entrypoint.sh`](entrypoint.sh) defaults to dropping you into a bash shell, you can build the image and start an interactive session immediately with:

```bash
docker compose run --rm --build ns2
```

*You are now inside the container's interactive shell*

### 2. Running the Training

We use Hydra for hierarchical configuration management. Configurations are defined in the [`config/`](config/) directory, where [`config/config.yaml`](config/config.yaml) is the default configuration.

You can start the training process and dynamically override config values directly from the terminal. This includes individual values like the Learning Rate, but also full sets of configurations like setting up a specific dataset.

For example:

```bash
python scripts/train.py dataset=vctk training.learning_rate=1e-4 wandb=serious_run
```

Running the [training script](scripts/train.py) will automatically initiate our [Preprocessing-Pipeline](naturalspeech2/data/dataset.py) and prepare the dataset, before the Train-Loop starts. 

---

### 3. Dataset and Custom Mounts

Following the paper [1], we use the english subset of [**Multilingual LibriSpeech (MLS)**](https://huggingface.co/datasets/parler-tts/mls_eng) [7] as our training dataset, which the [Preprocessing-Pipeline](naturalspeech2/data/dataset.py) will load per default. 

> **Note:** This will download **705 GB** of Parquet files, which will translate to an additional **~800 GB** worth of Arrow files after deserializing. During preprocessing, the peak disk usage is **~2x** the dataset in arrow format. 
Additionally, if you choose to use `resample_on_the_fly=False` (e.g. if your CPU-node can't handle resampling during dataloading fast enough), the pre-resampled FLAC files will require an additional **~3.8 TB** of disk space.

We also provide the option to instead use the significantly smaller [**VCTK**](https://huggingface.co/datasets/sanchit-gandhi/vctk) dataset, in case you just want to play around with the codebase.

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

*If it passes, the output will yield a safe bucket mapping configuration that you should paste directly into your Hydra `config` setup, in case the setting from the previous step led to an OOM during the stress-test.*

**4. Dataloader Optimization**  

Determine the most efficient data loading strategy (resampling during pre-processing vs. on-the-fly) and test worker configurations for your specific hardware-setup. 

This can be done by running [this bash script](scripts/benchmarks/dataloader/run_benchmark_dataloader.sh), which will [benchmark](scripts/benchmarks/dataloader/benchmark_dataloader.py) the dataloader across multiple configurations. You might have to update the tested `WORKER_COUNTS` and `ASSUMED_MAX_BATCH_SIZE` to match your machine. 
Per default, it only evaluates, if the resampling on-the-fly option will run efficiently on your hardware, but you can use command-line arguments to test other options as well. 

`--full` tests both the otf-resampling and the pre-resampling option, `--pre-resample` tests only the pre-resampling option (e.g. if you ran the default otf-setting, but your dataloader starved the GPU). You can also specify the dataset to be used in the benchmark via `--dataset=vctk` (the default is `mls_eng`).

```bash
python scripts/benchmarks/dataloader/run_benchmark_dataloader.sh
```

> **Note:** Technically, resampling always happens during pre-processing, since this is necessary to determine the pitch, but the option to resample on-the-fly discards the resampled audio and therefore saves disk space.

---

### 5. Final Steps before Starting Training
> **Note:** If you just want to recreate our results, and run your training on the same settings, you can start training right away, since the following tests are completely hardware-agnostic.

We ran the mentioned tests to **verify the correctness** of our implementation, and to obtain **reasonable hyperparameter-defaults** for the main training run.

**1. Initialization**

To preserve variance **locally**, we initialize all *Linear*- and *Conv1D*-Layers via a Gaussian Distribution with:

$$E[w] = 0$$ 

$$\mathbf{Var[w] = \frac{2}{n_{in}}}$$ 

where **Biases** are initialized as **0** (*He-Initialization* [8]).
We assume, the SiLU-activation matches the form of ReLU close enough, for this initialization-scheme to still work sufficiently. 
Per default, we initialize the **scale-parameters** of *RMSNorm-Blocks* as **1**. 

Additionally, to prevent exponential variance-growth **globally**, we initialize the weights of the final operation of every *Residual-Layer* to exactly **0**, which turns each *Residual-Layer* into an Identity-Function during the first iteration.

**2. Overfit-Test**

To verify that there are no fundamental errors preventing the model from learning, we intentionally overfit it by running the [Train-Loop](scripts/train.py) on a single batch for multiple iterations.

```bash
python scripts/train.py setup=overfit_test wandb=overfit_test
```

**3. Loss-Analysis**

Since our Loss comprises multiple individual Loss-Terms, we run a few iterations of the [Train-Loop](scripts/train.py) to estimate their magnitude. 

We then scale the Loss-Terms accordingly, so they influence the shared parameters equally (actually we might still want to introduce intentional biases towards individual Terms afterwards).

```bash
python scripts/train.py setup=loss_analysis wandb=loss_analysis
```

**4. Gradient-Analysis**

In order to verify, if the Loss-Balancing from the previous step actually worked, we run the [Train-Loop](scripts/train.py) again, but perform the backward passes individually for each Loss-Term. This allows us to compare the L2-norms of the Loss-Term-gradients of shared parameters, which acts as our measure for impact on the model. 

We additionally calculate cosine-similarities between the gradients, to make sure that individual Loss-Terms do not compete with each other. If that were the case, we would need to introduce further measures, like slowly warming up those specific Loss-Terms.

```bash
python scripts/train.py setup=gradient_analysis wandb=gradient_analysis
```

**5. Hyperparameter-Tuning**

For all Hyperparameters that are explicitly stated in the original paper [1], we use the declared values. The rest is tuned by Optuna, or alternatively set to our best guesses, to perform Hyperparameter-Tuning inside a reasonable time frame. 

For a detailed breakdown, see **this table**.

## Training Hardware

The reference models in this repository were trained on the following system configuration:

* **GPU:** NVIDIA RTX PRO 6000 Blackwell Workstation-Edition (96 GB VRAM)
* **CPU:** 32 Cores
* **RAM:** 252 GB 

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