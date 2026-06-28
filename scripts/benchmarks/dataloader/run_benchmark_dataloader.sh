#!/bin/bash

# Dataloader starvation benchmark — a READ-ONLY consumer of an already-preprocessed train split.
# For the chosen setting it sweeps num_workers and, per config, times the real training step
# (create_dataloader → bucketed loader + NS2 fwd/bwd) to check whether CPU dispatch + dataloading stays
# fully hidden behind GPU compute.
#
# Decoupled from preprocessing, by design. Intended workflow:
#   1. Preprocess the train split (full or a subset) with the setting you intend to ship — for us that
#      is on-the-fly resampling (OTF), the default here. (scripts/preprocess.py)
#   2. Run THIS benchmark on that one setting and read the W&B / log starvation rate + P95.
#   3. If OTF is healthy you are done — there is no reason to benchmark anything else.
#   4. ONLY if it starves: separately preprocess a subset with the alternative setting (pre-resampled
#      FLAC) and re-run with --pre-resample. OTF is strictly easier to operate (no multi-TB FLAC copy,
#      any node can resample on the fly), so it is the default and the expected answer.
#
# The benchmark NEVER preprocesses, moves, or deletes data: if the requested setting is not already
# preprocessed it fails loud (build_if_missing=False). The only state it touches is the OS page cache,
# dropped between configs so each reads cold (unbiased disk I/O); the Arrow shards re-mmap on demand.

DATASET_NAME="mls_eng"
MODE_OVERRIDES=""          # set by --pre-resample (otherwise OTF = the config default)
RUN_TAG="otf"
EXTRA_OVERRIDES=()         # forwarded verbatim to Hydra, e.g. dataset.max_train_clips=200000

while [[ "$#" -gt 0 ]]; do
  case $1 in
    # Secondary setting: single-folder FLAC path (the chunked store requires OTF). Must have been
    # preprocessed separately first — the benchmark fails loud otherwise.
    --pre-resample) MODE_OVERRIDES="dataloader.resample_on_the_fly=false dataset.chunk_size=null"; RUN_TAG="pre"; shift ;;
    --dataset=*) DATASET_NAME="${1#*=}"; shift ;;
    --*) echo "Unknown flag: $1 (valid flags: --pre-resample, --dataset=NAME). Hydra overrides pass through WITHOUT a '--' prefix, e.g. dataset.max_train_clips=200000." >&2; exit 1 ;;
    *) EXTRA_OVERRIDES+=("$1"); shift ;;   # bare Hydra override → forward to the python benchmark
  esac
done

# Output log file
LOG_FILE="logs/benchmarks/dataloader_benchmark.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Measured steps per config (+ warmup, which is skipped). 1000 gives a stable starvation rate (binomial
# standard error <~1.6 %) and a ~50-sample P95 tail — enough to argue healthy/starved confidently
# without burning GPU time on more.
export NUM_BENCHMARK_STEPS=1000
export WARMUP_STEPS=100

# Proactive host-RAM watchdog threshold: the benchmark self-aborts (exit 42 → sweep stops) above this
# %, before the OS OOM killer can fire and kill a co-tenant job on a shared box. Raise/lower per host.
export RAM_ABORT_PERCENT=90

# Swap-thrash threshold (GB): the watchdog also aborts if swap grows by this much during a config. The
# kernel can hold RAM% under the limit by paging out, so this catches the thrashing the RAM % alone
# misses (box laggy/unusable but steps still crawling). Lower it to be stricter on swap-light hosts.
export SWAP_ABORT_GB=2

# num_workers values to sweep (tune to your CPU core count)
WORKER_COUNTS=(8 16 24 30)

echo "Dataloader benchmark | setting=$RUN_TAG | dataset=$DATASET_NAME | $NUM_BENCHMARK_STEPS steps + $WARMUP_STEPS warmup per config" > $LOG_FILE
echo "Read-only consumer: the '$RUN_TAG' setting must already be preprocessed (it is never built or deleted here)." | tee -a $LOG_FILE

for workers in "${WORKER_COUNTS[@]}"; do

  echo -e "\n===========================================================" | tee -a $LOG_FILE
  echo "Testing num_workers=$workers (setting=$RUN_TAG)" | tee -a $LOG_FILE

  # Clear ONLY the OS page cache → each config reads cold (unbiased disk I/O). This frees no
  # preprocessed data; the Arrow shards are simply re-mmapped on first access.
  echo "Clearing OS page cache..." | tee -a $LOG_FILE
  sync; echo 3 > /proc/sys/vm/drop_caches || echo "Warning: drop_caches needs root — without it, warm-cache reads bias the I/O timing low." | tee -a $LOG_FILE

  python scripts/benchmarks/dataloader/benchmark_dataloader.py \
    dataset=$DATASET_NAME \
    $MODE_OVERRIDES \
    "${EXTRA_OVERRIDES[@]}" \
    dataloader.num_workers=$workers \
    wandb=benchmark_dataloader \
    wandb.run_name="workers_${workers}_${RUN_TAG}" \
    2>&1 | tee -a $LOG_FILE
  rc=${PIPESTATUS[0]}   # python's exit code (not tee's)

  # Stop the sweep the moment a config exhausts host RAM, thrashes/stalls, or otherwise crashes. The
  # sweep is ascending in num_workers, so a RAM blow-up only worsens from here; completed configs are
  # already logged. 42 = the benchmark's own RAM/stall watchdog; 137 = the OS OOM killer beat it.
  if [ "$rc" -ne 0 ]; then
    if [ "$rc" -eq 42 ]; then
      echo "ABORTING SWEEP: num_workers=$workers tripped the host-RAM / stall watchdog. Higher worker counts use more RAM and would fail too." | tee -a $LOG_FILE
    elif [ "$rc" -eq 137 ]; then
      echo "ABORTING SWEEP: num_workers=$workers was OOM-killed (exit 137). Higher worker counts would too; consider lowering RAM_ABORT_PERCENT to catch it gracefully first." | tee -a $LOG_FILE
    else
      echo "ABORTING SWEEP: num_workers=$workers exited $rc — see the log above." | tee -a $LOG_FILE
    fi
    break
  fi

  # Let GPU free allocations
  sleep 2
done

echo -e "\nAll configs done. Compare the per-config starvation rate / P95 in $LOG_FILE and W&B."
