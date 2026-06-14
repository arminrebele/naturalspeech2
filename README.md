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

**Evaluation (decoupled eval daemon).** Every `setup.eval_interval` steps the model is evaluated — held-out loss over the dev+test pool, audio generation on fixed reference clips, and objective metrics (WER intelligibility, SIM-o speaker similarity). With a **second GPU**, this whole eval block is off-loaded to a persistent subprocess pinned to **GPU 1**: the trainer (GPU 0) only does a fast weight-snapshot copy and keeps stepping, so training **never pauses** for evaluation, generation, or metrics. The subprocess is fully isolated — if it crashes it cannot take down training, and the trainer respawns it. It reports held-out loss on **both the live and the EMA weights** (best-checkpoint selection stays on EMA val loss) and owns the `ema_best.safetensors` write; results stream back to the trainer and are logged to W&B against a dedicated `eval/snapshot_step` x-axis. On a **single-GPU** host — or with `setup.eval_daemon.enabled=false` — evaluation transparently falls back to the original **in-process** path (runs synchronously on the training card between steps, exactly as before). Diagnostic modes (`overfit_test`, `gradient_analysis`, aligner trials) always evaluate in-process.

| `config/setup/base.yaml` key | Default | Purpose |
|---|---|---|
| `eval_daemon.enabled` | `true` | Use the daemon when a 2nd GPU is present; `false` → always in-process. |
| `eval_daemon.compile` | `true` | `torch.compile` the daemon's eval forward (the in-process forward was already compiled). |
| `eval_daemon.num_workers` | `4` | Dataloader workers per daemon loader — independent of `dataloader.num_workers`, so the main run isn't slowed. |
| `eval_daemon.snapshot_dir` | `null` | RAM-backed weight-handoff dir; `null` → `/dev/shm/ns2_eval/<run>`. |
| `eval_daemon.batch_size_divisor` | `2` | Shrinks the daemon's per-bucket eval batch (÷) on the 2nd GPU for VRAM headroom; grad-accum is ×'d by the same factor, so the logical batch and the metric are unchanged. `1` = full batch. |
| `eval_metrics` | `["wer"]` | Objective metrics on the fixed refs: `wer` (HuBERT-CTC, no LM) and/or `sim_o` (WavLM-Large-SV speaker cosine vs. the prompt). Enable both with `["wer","sim_o"]`. |
| `metric_device` | `auto` | Metric-model device (WER + SIM-o): `auto` → idle 2nd GPU if present, else CPU (never silently the training card; override e.g. `cuda:0`). Daemon forces its own card. |

---

### 3. Dataset and Custom Mounts

Following the paper [1], we use the english subset of [**Multilingual LibriSpeech (MLS)**](https://huggingface.co/datasets/parler-tts/mls_eng) [7] as our training dataset, which the [Preprocessing-Pipeline](naturalspeech2/data/dataset.py) will load per default. 

> **Note:** This downloads **705 GB** of Parquet (cached in your mounted `~/.cache/huggingface`), deserializing to **~800 GB** of Arrow. The Parquet cache persists *alongside* the Arrow, and during preprocessing the intermediate `.map` outputs transiently double the Arrow before the old shards are reclaimed — so the **peak is ≈ 705 GB Parquet + ~2× 800 GB Arrow ≈ 2.3 TB**, settling back to ~800 GB processed + 705 GB Parquet cache afterwards.
> Additionally, `resample_on_the_fly=False` (e.g. if your CPU node can't resample fast enough during loading) writes pre-resampled FLAC for another **~3.8 TB**.

For fast development iteration, `+experiment=overfit_test` (1k clips) and `+experiment=multibatch` (4k clips) cap the **train** split to a small slice via `max_train_clips` (preprocessed before training), so they touch only a tiny fraction of the ~800 GB train data. The default no-arg invocation (the `full_run` composition) uses the full train split.

> **Note — held-out (dev/test) length caps.** Unlike the train split, the dev/test splits aren't sized by `max_train_clips`; their max audio/phoneme lengths are taken from the **largest bucket** in [`config/dataloader/base.yaml`](config/dataloader/base.yaml) and frozen into the processed cache when first built. So if you later widen the buckets — e.g. re-running the [§4](#4-benchmarking-and-hardware-optimization) benchmark on the full corpus after developing against a small subset — held-out caches built under the older, narrower buckets keep the smaller cap and silently drop the now-in-range longer clips. The loader flags this at startup with a loud `PREPROCESSING CONFIGURATION MISMATCH` (naming `max_phoneme_length`). Keeping the stale cache is harmless *within* a single run lineage (the eval set stays fixed → dev loss stays comparable), but **re-preprocess the held-out splits once against the final buckets before the productive run** (delete their `processed_*` directory to trigger it).

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

**1. Bucket Boundaries (`audio_length` + `phoneme_length`)**

Dynamic batch bucketing groups sequences by length into fixed `(audio_length, phoneme_length, batch_size)` buckets ([`config/dataloader/base.yaml`](config/dataloader/base.yaml)) — crucial for maximizing VRAM utilization, keeping the `torch.compile` graph cache hot, and reducing memory fragmentation. `audio_length` is the pad width on the audio axis; `phoneme_length` is **both** a pad width **and a hard cap** (a clip with more phonemes than its audio-bucket's value crashes `BucketedCollateFn`). Derive the two length columns here — for a range of bucket counts *K*, trading compiled shapes against padding waste — then `batch_size` in step 2. Which tool you run depends on whether the data is preprocessed and whether you'll later train on more of it:

| Situation | `audio_length` from | `phoneme_length` from | Re-run later? |
| --- | --- | --- | --- |
| **Final split, already preprocessed** — full corpus, *or* a fixed subset you won't grow | `find_optimal_buckets.py` | same run | No — derived once |
| **Training subsets now, full corpus later** | `find_audio_buckets_from_parquet.py` | `measure_bucket_phoneme_lengths.py` | Phonemes only, on each subset growth (audio is already final) |

- [`find_optimal_buckets.py`](scripts/benchmarks/dataloader/find_optimal_buckets.py) — **both columns at once**, off the cached `audio_length`/`phoneme_length` of a *preprocessed* split. It builds the split through the training path, so `dataset.max_train_clips=N` targets exactly the subset you'll train (full split → final). Use it for a split you won't change.
- [`find_audio_buckets_from_parquet.py`](scripts/benchmarks/dataloader/find_audio_buckets_from_parquet.py) — **audio only**, straight from the parquet headers (`len = ceil(frames · target_sr / native_sr)`, exactly the decode path) with **no preprocessing**. Audio depends only on the length distribution, knowable for the *full corpus* up front, so these boundaries are **final** and stay valid for any subset. Run it to fix audio before the multi-day preprocessing finishes.
- [`measure_bucket_phoneme_lengths.py`](scripts/benchmarks/dataloader/measure_bucket_phoneme_lengths.py) — the **phoneme** column for the *current* subset: it groups that subset through the **real sampler** and prints each bucket's `max` (the value to paste) in a ready-to-paste block, **and** verifies an already-set `phoneme_length`, exiting non-zero on any overflow so it can gate `preprocess && measure && train`. `phoneme_length` may be unset when you run it (pure derive). Re-run on every subset growth.

```bash
python scripts/benchmarks/dataloader/find_optimal_buckets.py                                 # both columns, full split
python scripts/benchmarks/dataloader/find_optimal_buckets.py dataset.max_train_clips=200000  #   ... on a subset
python scripts/benchmarks/dataloader/find_audio_buckets_from_parquet.py                      # audio only, no preprocessing
python scripts/benchmarks/dataloader/measure_bucket_phoneme_lengths.py                       # phoneme only (pass your run's overrides)
```

> **Note — audio is final; re-check only phonemes as the subset grows.** The train split isn't phoneme-filtered, so a bucket's max phoneme count only *grows* with the subset — re-run `measure_bucket` (and raise any flagged cap) whenever `max_train_clips` increases; audio never changes. That subset coverage is the *only* reason to re-derive — **not** a boundary-rounding artefact: the sampler buckets by raw samples and `find_optimal` by frame length rounded up to a ×8 grid, but since the boundaries live on that same ×8 grid the two assignments are provably identical (they diverge only when the data does). Training also self-guards — the sampler **fails loud at startup** if any clip exceeds its bucket's phoneme cap (mirroring the audio-overflow check), so a stale cap can't silently crash mid-run; `measure_bucket` is the pre-flight that hands you the numbers first.

> **Note:** Paste the boundaries for your chosen *K* into [`config/dataloader/`](config/dataloader/) before continuing.

**2. Finding Maximum Batch Sizes**

Once your bucket boundaries are defined, you need to assign the respective buckets their [optimal batch size](scripts/benchmarks/dataloader/find_max_batch_sizes.py), as at that point, the buckets only contain the optimal sequence length.

```bash
python scripts/benchmarks/dataloader/find_max_batch_sizes.py
```

> **Note:** Paste the bucket-boundaries into the respective [config-files](config/dataloader/) again, to override your values from the previous step. These will represent the *(almost)* final bucket-mapping, since it now includes the found maximum batch-size as well.

**3. Stress-Test + Compile-Mode Selection**

This [harness](scripts/benchmarks/dataloader/stress_test_fragmentation.py) cycles every bucket through a real forward+backward step (dummy batches), doing two jobs at once: it (a) **stress-tests the allocator** against back-to-back shape jumps — catching an OOM deep into training that the per-bucket sizing missed — and (b) **A/B-compares `torch.compile` modes** for the training step.

```bash
python scripts/benchmarks/dataloader/stress_test_fragmentation.py
```

For each `(dynamic, mode)` it reports steady-state **ms/step**, **peak VRAM**, the **unique-graph count**, warmup wall-time, and a **plateau check** (a full extra bucket pass must add *no* new graphs = no shape leak). Use the table to choose `setup.compile.{dynamic,mode}` (step 5). Env knobs: `COMPILE_SWEEP=auto|fast|all` (default `all`, including both cudagraph autotune modes), `COMPILE_TIMED_ROUNDS`, `COMPILE_MAX_WARMUP_PASSES`.

> **Note:** `dynamic=False` (a static graph per bucket) and `max-autotune` trade a longer one-off warmup for a faster steady-state step — usually worth it over a 400–600k-step run, but **benchmark it**: the win is hardware-dependent (the big channel dims are already static, so only the batch/length dims are at play). The cudagraph modes (`reduce-overhead`, `max-autotune`) reserve a static memory pool per shape → **higher peak VRAM**; if the mode you pick raises peak VRAM, **re-run step 2** — the bucket batch sizes were tuned against the old peak. An OOM here on the default (`auto`) mode means the bucket batch sizes themselves are over budget and need lowering.

**4. Dataloader Optimization**  

Determine the most efficient data loading strategy (resampling during pre-processing vs. on-the-fly) and test worker configurations for your specific hardware-setup. 

This can be done by running [this bash script](scripts/benchmarks/dataloader/run_benchmark_dataloader.sh), which will [benchmark](scripts/benchmarks/dataloader/benchmark_dataloader.py) the dataloader across multiple configurations. You might have to update the tested `WORKER_COUNTS` and `ASSUMED_MAX_BATCH_SIZE` to match your machine. 
Per default, it only evaluates, if the resampling on-the-fly option will run efficiently on your hardware, but you can use command-line arguments to test other options as well. 

`--full` tests both the otf-resampling and the pre-resampling option, `--pre-resample` tests only the pre-resampling option (e.g. if you ran the default otf-setting, but your dataloader starved the GPU).

```bash
python scripts/benchmarks/dataloader/run_benchmark_dataloader.sh
```

> **Note:** Technically, resampling always happens during pre-processing, since this is necessary to determine the pitch, but the option to resample on-the-fly discards the resampled audio and therefore saves disk space.

**5. `torch.compile` Configuration and Tracking**

The compile call is configured under `setup.compile` (the defaults reproduce a bare `torch.compile(model)`):

| Key | Default | Meaning |
| --- | --- | --- |
| `compile.enabled` | `true` | Compile the training model. `false` runs eager (debugging). |
| `compile.dynamic` | `null` | `null` = automatic (one static graph for a bucket's first shape, then one symbolic graph generalizing the rest); `false` = a static graph per bucket (more graphs, no symbolic-shape overhead); `true` = fully symbolic. |
| `compile.mode` | `default` | `default` / `reduce-overhead` / `max-autotune-no-cudagraphs` / `max-autotune`. The latter two autotune kernels; `reduce-overhead` and both `max-autotune` modes enable CUDA graphs (lower launch overhead, **higher peak VRAM**). |

The daemon's eval forward reuses `compile.{dynamic,mode}` (its on/off is `eval_daemon.compile`), so eval stays consistent with training. Pick the values from the step-3 A/B; changing `mode` to a cudagraph mode can shift peak VRAM (re-run step 2 if so).

**What's tracked at runtime.** Bucketing makes the input shapes airtight, so the compiled graphs should reach a steady state and then stay there. Training logs a per-`log_interval` wandb series — the two signals to watch:

- `Compile/unique_graphs` **must plateau** after warmup. A staircase that keeps climbing at non-eval steps means shapes are *leaking* (recompiling past the *K* buckets); training also prints a `⚠️ LEAKING` warning in that case. (The first in-process eval adds a one-time eval-mode graph family — recognized and logged as benign, not a leak. The 2-GPU daemon evals in a separate process, so it never perturbs the trainer's count.)
- `Compile/n_break_reasons` **must stay flat** at the two intended `@torch.compiler.disable` sites (`encodec.get_latents`, `aligner.maximum_path_indices`); a new break reason is logged the moment it appears. `Compile/cache_size_limit` is logged as a canary.

The daemon reports the same signals independently under `Evaluation: Compile/*` (and pre-warms all eval graphs at startup so the first snapshot isn't slowed). Absolute counts differ between train (forward+backward) and the daemon (forward-only) — only the *plateau*, never the raw number, is the health signal.

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

> **Exception — aligner encoders.** The aligner's mel/text conv stacks use **Xavier-uniform** init (not the global $\mathcal{N}(0, 0.02)$). Its attention is raw squared-L2 ($-\lVert \text{mel}-\text{text}\rVert^2$), which needs $O(1)$ feature magnitudes for a non-degenerate per-frame softmax; under $\mathcal{N}(0, 0.02)$ the features collapse to ~0, the distances to ~0, the softmax to ~uniform, and the forward-sum CTC to all-blank. RAD-TTS uses the same Xavier init for these encoders.

Additionally, to safely route gradients through the deep predictor and WaveNet stacks, we strictly apply Fixup/ReZero principles. We initialize the final operation of every *Residual-Branch* (and FiLM projection) to exactly **0**, forcing the entire network into a perfect Identity-Function at step 0.

> **Note:** Because we use SiLU activations, which have a non-zero derivative at $x=0$, zeroing the weights of a 1-layer residual branch does not kill the gradient, allowing the branch to safely "wake up" during training.

**2. Overfit-Test**

To verify that there are no fundamental errors preventing the model from learning, we intentionally overfit it by running the [Train-Loop](scripts/train.py) on a single batch for multiple iterations.

```bash
python scripts/train.py +experiment=overfit_test
```

**3. Gradient-Analysis**

On weighting itself the paper is almost silent: the only weight it states is λ_ce-rvq = 0.1; all other terms are summed as-is (and the aligner / voicing terms are our additions — they don't exist there). We keep that implied scheme — every weight 1.0, ce_rvq 0.1 — as the default and deviate from it only on a measured pathology, judged by **gradient impact, not loss magnitude**: magnitude proved a poor proxy for impact in our runs (two terms of near-identical magnitude differed ~30× in gradient norm on shared parameters).

The decision instrument is therefore a per-term gradient diagnostic. We run the [Train-Loop](scripts/train.py) again, but perform the backward passes individually for each (weighted) Loss-Term. This lets us compare the L2-norms of the Loss-Term-gradients on shared parameters, which acts as our measure for impact on the model.

We additionally calculate cosine-similarities between the gradients to expose competing Loss-Terms. A weight deviates from its default only on a measured pathology: a term that conflicts with others (negative cosine) while dominating them in norm, or a term being starved on a parameter region it must train. Orthogonal pulls (cosine ≈ 0) of different sizes are accepted — they coexist rather than fight. Escalation order: slowly warming up the offending term (the built-in **hold-then-ramp** schedule — `loss_warmup_hold_steps` steps at weight 0, then a linear 0→target ramp over `loss_warmup_steps`), then a static down-weight.

For the **alignment and duration** losses specifically, we instead **stop-gradient** their inputs (`detach_aligner_input` / `detach_duration_predictor_input`, default on) so they don't reach the shared phoneme encoder at all. The **duration** detach is the paper recipe (Glow-TTS's `sg[·]` on the duration input, Eq 6; RAD-TTS `torch.detach`s the duration's text input). The **aligner** detach shields the encoder: RAD-TTS achieves the same shielding *structurally* — it computes alignment off the **raw pre-encoder embedding** (`text_embeddings`, not the encoder output `text_enc`), so the CTC/bin gradient never reaches its contextual encoder. Our aligner reads the **post-encoder** `phoneme_encodings`, so we detach for the equivalent shielding; the truly RAD-faithful alternative — tapping the aligner off the phoneme **embedding** (pre-transformer) — is the unbuilt "upstream-tap" option. (Pitch is left **attached**, matching RAD-TTS's F0 predictor, which reads non-detached `text_enc`.) When one task's gradient both dominates and destabilizes a shared backbone, decoupling it is cleaner than re-weighting; the encoder is then shaped only by the generative path (+ pitch). Set the flags `false` to A/B the coupled form.

The aligner/duration **warmups** follow a fs → bin → duration curriculum (fixed absolute steps, anchored to examples-seen — our ~80k-frame/step batch is ~10× the alignment papers', so RAD-TTS's `bin@18k` scales to ~2k): `forward_sum` is full from step 0; `bin` holds to 2k then ramps to full at 8k (premature hardening locks in ~uniform durations before the soft alignment differentiates phoneme lengths — RAD-TTS delays `bin` for the same reason); `duration` holds to 6k then ramps to full at 15k, after `bin` has sharpened the Viterbi path its MSE targets.

```bash
python scripts/train.py +experiment=gradient_analysis
```

**4. Hyperparameter-Tuning**

For every hyperparameter the original paper [1] states explicitly, we use the declared value. The rest is set to a sensible default or — where the paper is silent and the choice is sensitive — tuned with **Optuna**.

**Aligner scalars.** Following the reference implementation (RAD-TTS), the alignment distance is **raw squared-L2** (`−‖mel−text‖²`, no temperature — sharpness is set by the aligner's Xavier init, see §5.1). That leaves **two** scalars unspecified: `prior_w` (Beta-Binomial prior strength) and `blank_logit` (CTC blank-vs-label calibration). Because alignment gates everything downstream, we search them with Optuna (TPE sampler) to **minimize the held-out aligner loss** (`forward_sum + bin`):

```bash
python scripts/tuning/run_aligner_optuna.py --n-trials 30
```

Each trial is a fresh `+experiment=aligner_trial` training job (`--max-iters`, default 5000) that dumps per-eval held-out aligner losses; the trial is scored on the mean over its converged tail. Study state persists to SQLite under `research/aligner_optuna/`, so runs are resumable and inspectable. Tune the search ranges in `SEARCH_SPACE` at the top of [the driver](scripts/tuning/run_aligner_optuna.py), and use `--override <hydra.key=value>` to forward extra Hydra overrides to every trial. The best scalars print at the end → paste into [`config/model/base.yaml`](config/model/base.yaml). To probe one setting by hand: `python scripts/train.py +experiment=aligner_trial model.aligner.prior_w=0.1`. The winner still gets a manual `overfit_test` (monotonic durations, intelligible audio) — the loss objective is necessary, not sufficient.

**Learning rate.** The paper's LR is for a different codec / aligner / normalization regime and doesn't transfer directly. Rather than an Optuna sweep over a single scalar, we find it with a **range test** (Smith 2017): one short run that exponentially ramps the LR from `min_lr` to `max_lr`, logs a loss-vs-LR curve, and stops on divergence.

```bash
python scripts/train.py +experiment=lr_range_test
```

The run-end summary prints the steepest-descent LR and the min-loss LR (use ~1 decade below); confirm against the `LR Range Test/loss` vs `/lr` curve in W&B. This is the **last** tuning step — run it on a model that already trains healthily, since the right LR depends on the fixed architecture (loss weights, aligner, normalization), and a plateaued model's LR-invariant loss tells you nothing.

**Dropout.** Zero by default: the paper reports the model still underfitting at 300k steps, so regularization is expected net-negative. We reintroduce it locally only if a run shows overfitting (val/train divergence), most likely in the small duration/pitch predictors.

**5. Alignment & Conditioning Diagnostics**

When a run underfits despite a clean overfit test, two read-only probes — run on any saved checkpoint, logged to the eval W&B project, writing nothing — localize the cause:

- **Alignment heatmap** decomposes the aligner into three per-frame distributions (Beta-Binomial **prior** alone / **learned** scores alone / **posterior**) with the Viterbi path overlaid, plus pooled scalars. Learned ≈ prior (low `align/learned_vs_prior_tv`, `learned_peak` ≈ `1/P`) ⇒ the aligner is riding the prior, not learning; sharp + content-dependent ⇒ the aligner is fine and the bottleneck is downstream.

  ```bash
  python scripts/diagnostics/alignment_heatmap.py checkpoint=models/checkpoints/main_training/ckpt.pt
  ```

- **Conditioning ablation** re-samples from identical noise with the real vs zeroed/shuffled content `condition` and a swapped speaker prompt. A content/speaker RMSE ratio ≪ 1 means the denoiser ignores the phoneme content (a problem downstream of the aligner). It decodes real vs zeroed-condition audio to listen.

  ```bash
  python scripts/diagnostics/condition_ablation.py checkpoint=models/checkpoints/main_training/ckpt.pt
  ```

Both take the architecture from the checkpoint (the resume rule), like the checkpoint benchmark below.

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
- **Disk** — the binding constraint for the full MLS run; see [§3](#3-dataset-and-custom-mounts). The small-`max_train_clips` diagnostic modes (`+experiment=overfit_test|multibatch`) need only a tiny fraction and are the fastest end-to-end check of a fresh setup.

> **Inference** ([§ Generating Audio](#generating-audio)) is far lighter than training — a single GPU, one clip at a time, no DataLoader workers — so neither `shm_size` nor large RAM is a concern.

---

## Inference and Pre-trained Models

> **Note:** Pre-trained weights are not yet published — the Hugging Face repo id below is a placeholder, filled in once the training runs complete. The generation code and workflow are in place and usable today with your own checkpoints.

### Inference Architecture

During inference, the model takes a text transcript and a short speech prompt, predicting phoneme durations and pitch, and running the reverse diffusion process to synthesize the final audio latents.

[![NaturalSpeech 2 Inference Architecture](docs/architecture/inference-architecture.png)](docs/architecture/inference-architecture.png)

---

### Generating Audio

Both workflows funnel through `generate_audio()` — the Layer-3 wrapper that phonemizes the text, resamples the reference, validates inputs, and calls `model.generate()`. Pass text and a reference clip; never raw phonemes.

**Local weights** — your own training run, or anyone who followed the recipe. The CLI is the quickest path:

Checkpoints are written per run under `models/checkpoints/<setup.log_name>/` (e.g. `main_training/`) so a diagnostic run can never clobber the main run's resume point; adjust the paths below to your run's subdir.

```bash
python scripts/inference.py \
    --checkpoint models/checkpoints/main_training/ema_best.safetensors \
    --prompt path/to/reference.wav \
    --text "Hello world." \
    --prompt-seconds 10        # optional: slice the reference to a 10 s window
```

…or call the library directly (e.g. from a notebook):

```python
import soundfile as sf
from naturalspeech2.inference import load_inference_model, generate_audio
from naturalspeech2.modules.encodec import SAMPLING_RATE

model = load_inference_model("models/checkpoints/main_training/ema_best.safetensors", device="cuda")
audio, length = generate_audio(model, "path/to/reference.wav", "Hello world.", prompt_seconds=10)
sf.write("out.wav", audio[:length], samplerate=SAMPLING_RATE)
```

**Recovering `ema_final` from a partial run.** A completed run writes both `ema_best.safetensors` (best dev loss) and `ema_final.safetensors` (final EMA weights). A run that crashed or was stopped early leaves only `ema_best` — it is written in-loop, whereas `ema_final` is a post-loop step that never ran. The final EMA weights are still inside `ckpt.pt`, so re-serialize them into the inference format without retraining:

```bash
python scripts/export_ema.py \
    --checkpoint models/checkpoints/main_training/ckpt.pt \
    --output models/checkpoints/main_training/ema_final.safetensors
```

**From Hugging Face** — once weights are published, pass a repo id instead of a path; the weights, config, and token vocabulary are downloaded and cached automatically:

```python
from naturalspeech2.inference import load_inference_model, generate_audio

model = load_inference_model("<org>/<model-repo>", device="cuda")   # downloads on first use
audio, length = generate_audio(model, "path/to/reference.wav", "Hello world.")
```

**Reference length & limits.** Training uses 3 s prompts, but longer references at inference (≈5–15 s) tend to improve speaker similarity [1]; `prompt_seconds` keeps a leading window of the chosen length (default: the full clip). Inputs are validated at the boundary — empty text, sub-frame audio, and sequences beyond the model's `rope_max_seq_len` ceiling (≈40 s of reference, or its phoneme equivalent) raise a clear error instead of failing deep in the model. A per-phoneme duration cap (`--max-seconds-per-phoneme`, default 4 s) guards against runaway synthesis from an under-trained or out-of-distribution duration prediction.

---

### Benchmarking a Checkpoint

`scripts/inference.py` generates a single clip; to score a checkpoint on the **same objective metrics the training loop reports** — held-out (and train-subset) loss, WER, SIM-o, and sample audio — use the standalone benchmark. It reuses the in-training eval core, so the numbers are directly comparable to a live run's, but it targets any saved checkpoint and logs to its own W&B project (`naturalspeech2-eval`).

```bash
# a resume checkpoint → self-describing; reports both live and EMA metrics
python scripts/eval_checkpoint.py checkpoint=models/checkpoints/main_training/ckpt.pt

# the shipped best-dev artifact → EMA-only (metadata read from the sibling ckpt.pt)
python scripts/eval_checkpoint.py checkpoint=models/checkpoints/main_training/ema_best.safetensors
```

It accepts either checkpoint kind. A `ckpt.pt` carries its own config plus both the live and EMA weights, so it needs no extra arguments and reports both. An `ema_*.safetensors` holds only the EMA weights; it reads the architecture and vocabulary from the sibling `ckpt.pt`, or — for a checkpoint downloaded without one — from `model_config=<yaml> token_vocab=<json>`. The benchmark is **read-only**: it never writes `ema_best`.

Eval *policy* is the normal Hydra config, overridable on the CLI like training — e.g. a larger, steadier sample than the lightweight in-loop eval, scoring held-out splits only:

```bash
python scripts/eval_checkpoint.py checkpoint=.../ema_best.safetensors \
    setup.num_audio_refs=300 eval_train=false out_dir=logs/benchmarks/run_a
```

`eval_train=false` scores held-out (dev+test) only and skips the train loader, so a checkpoint can be benchmarked with just those splits and a vocabulary — no preprocessed train split required (the default, `true`, additionally reports a train-subset loss and needs the train split). Because the reference clips are drawn from a fixed seed, two checkpoints evaluated with the same `seed`, splits, and `num_audio_refs` share an identical eval set and line up for head-to-head comparison in W&B; `out_dir` also dumps the scalars and generated wavs to disk.

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