#!/bin/bash

# Default arguments
TEST_MODE="otf-resample"
DATASET_NAME="mls_eng"

# Parse all passed arguments in any order
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --full) TEST_MODE="full"; shift ;;
    --pre-resample) TEST_MODE="pre-resample"; shift ;;
    --dataset=*) DATASET_NAME="${1#*=}"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
done

# Output log file
LOG_FILE="research/benchmarks/dataloader_benchmark.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Benchmark Parameters for Dataset Sizing
ASSUMED_MAX_BATCH_SIZE=32
# These are actual iterations without gradient accumulation steps
export NUM_BENCHMARK_STEPS=10000
export WARMUP_STEPS=100
TOTAL_STEPS=$((NUM_BENCHMARK_STEPS + WARMUP_STEPS))
DATASET_SPLIT_SIZE=$((ASSUMED_MAX_BATCH_SIZE * TOTAL_STEPS))

if [[ "$TEST_MODE" == "pre-resample" ]]; then
  echo -e "\nStarting Offline Resampling Benchmark (Mode: $TEST_MODE)..." >> $LOG_FILE
else
  echo "Starting Full I/O and GPU Pipeline Benchmark (Mode: $TEST_MODE)..." > $LOG_FILE
fi

echo "Provisioning $DATASET_SPLIT_SIZE samples to prevent page cache looping (Assumed Max Batch Size: $ASSUMED_MAX_BATCH_SIZE | Total Steps: $TOTAL_STEPS)" | tee -a $LOG_FILE

# Worker counts to test
WORKER_COUNTS=(8 16 24 30)

if [ "$TEST_MODE" == "full" ]; then
  RESAMPLE_FLAGS=("false" "true")
elif [ "$TEST_MODE" == "pre-resample" ]; then
  RESAMPLE_FLAGS=("false")
else
  RESAMPLE_FLAGS=("true")
fi

CLEAN_SPLIT="train_${DATASET_SPLIT_SIZE}"

echo "Wiping benchmark data directory (/workspace/data/$DATASET_NAME/$CLEAN_SPLIT) to ensure a clean start..." | tee -a $LOG_FILE
rm -rf /workspace/data/$DATASET_NAME/$CLEAN_SPLIT

for workers in "${WORKER_COUNTS[@]}"; do
  for resample_flag in "${RESAMPLE_FLAGS[@]}"; do
    
    echo -e "\n===========================================================" | tee -a $LOG_FILE
    echo "Testing Config: Resample On-The-Fly = $resample_flag | Workers = $workers" | tee -a $LOG_FILE

    # Clear OS RAM cache → disk I/O not artificially fast between iterations
    echo "Clearing OS RAM cache..." | tee -a $LOG_FILE
    sync; echo 3 > /proc/sys/vm/drop_caches || echo "Warning: Failed to drop caches (requires root)" | tee -a $LOG_FILE
    
    # Run benchmark on a large dataset subset, append to log
    python scripts/benchmarks/dataloader/benchmark_dataloader.py \
      dataset=$DATASET_NAME \
      dataset.split="train[:${DATASET_SPLIT_SIZE}]" \
      dataloader.num_workers=$workers \
      dataloader.resample_on_the_fly=$resample_flag \
      wandb=benchmark_dataloader \
      wandb.run_name="workers_${workers}_otf_${resample_flag}" \
      2>&1 | tee -a $LOG_FILE
      
    # Let GPU free allocations
    sleep 2
  done
done

echo -e "\nCleaning up benchmark dataset caches (/workspace/data/$DATASET_NAME/$CLEAN_SPLIT) to save disk space..." | tee -a $LOG_FILE
rm -rf /workspace/data/$DATASET_NAME/$CLEAN_SPLIT

echo -e "\nAll benchmarks complete. Check $LOG_FILE for summary."