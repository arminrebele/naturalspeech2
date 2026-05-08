#!/bin/bash

# Output log file
LOG_FILE="scripts/benchmarks/dataloader_benchmark.log"
echo "Starting Full I/O and GPU Pipeline Benchmark..." > $LOG_FILE

# We will test combinations of these workers
WORKER_COUNTS=(8 16 24 30)

for workers in "${WORKER_COUNTS[@]}"; do
  for resample_flag in "false" "true"; do
    
    echo -e "\n===========================================================" | tee -a $LOG_FILE
    echo "Testing Config: Resample On-The-Fly = $resample_flag | Workers = $workers" | tee -a $LOG_FILE

    # Delete the whole DATA_DIR to ensure a completely fresh run
    rm -rf /workspace/data/*
    
    # Run the benchmark and append output to log
    # Assign sufficiently large subset of the dataset
    python scripts/benchmarks/dataloader/benchmark_dataloader.py \
      dataset.split="train[:6000]" \
      dataloader.num_workers=$workers \
      dataloader.resample_on_the_fly=$resample_flag \
      | tee -a $LOG_FILE
      
    # Give the GPU a tiny moment to cool down and free memory allocations
    sleep 2
  done
done

echo -e "\nAll benchmarks complete. Check $LOG_FILE for summary."